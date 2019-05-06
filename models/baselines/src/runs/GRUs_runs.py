## import useful tools
import os
import time
import numpy as np
import pandas as pd
pd.set_option('precision', 4)

## import torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

# import our model and dataloader
from src.utils.argstools import args, createfolder, remove_file, Adam16
from src.dataseters.dataseterGRU import TyDataset, ToTensor, Normalize

# set seed 
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def get_dataloader(args):
    '''
    This function is used to get dataloaders.
    '''
    # transform
    transform = transforms.Compose([ToTensor(), Normalize(args)])
    
    traindataset = TyDataset(args=args, train = True, train_num=args.train_num, transform=transform)
    testdataset = TyDataset(args=args, train = False, train_num=args.train_num, transform=transform)
    # datloader
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.able_cuda else {}
    trainloader = DataLoader(dataset=traindataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testloader = DataLoader(dataset=testdataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    return trainloader, testloader

def train(net, trainloader, testloader, loss_function, args):
    '''
    This function is to train the model.
    '''
    # set file path for saveing some info.
    createfolder(args.result_folder)
    createfolder(args.params_folder)
    
    log_file = os.path.join(args.result_folder, 'log.txt')
    result_file = os.path.join(args.result_folder, 'result.csv')
    params_file = os.path.join(args.result_folder, 'params_counts.csv')
    params_pt = os.path.join(args.params_folder, 'params.pt')

    remove_file(log_file)
    remove_file(result_file)
    remove_file(params_file)
    remove_file(params_pt)

    # set the optimizer (learning rate is from args)
    if args.optimizer.__name__ == 'Adam':
        optimizer = args.optimizer(net.parameters(), lr=args.lr, eps=1e-06, weight_decay=args.weight_decay)
    elif args.optimizer.__name__ == 'Adam16':
        optimizer = args.optimizer(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, device=args.device)
    elif args.optimizer.__name__ == 'SGD':
        optimizer = args.optimizer(net.parameters(), lr=args.lr, momentum=0.6, weight_decay=args.weight_decay)
    else:
        optimizer = args.optimizer(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Set scheduler
    if args.lr_scheduler and args.optimizer.__name__ != 'Adam':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[x for x in range(1, args.max_epochs) if x % 5 == 0], gamma=0.7)
    
    total_batches = len(trainloader)
    
    # To declare a pd.DataFrame to store training, testing loss, and learning rate.
    result = pd.DataFrame([], index=pd.Index(range(1, args.max_epochs+1), name='epoch'), columns=['training_loss', 'testing_loss', 'lr'])

    for epoch in range(args.max_epochs):
        net.train(True)
        time_a = time.time()

        f_log = open(log_file, 'a')
        
        # update the learning rate
        if args.lr_scheduler and args.optimizer.__name__ != 'Adam':
            scheduler.step()
        # show the current learning rate (optimizer.param_groups returns a list which stores several params)
        print('lr: {:.1e}'.format(optimizer.param_groups[0]['lr']))
        # Save the learning rate per epoch
        result.iloc[epoch,2] = optimizer.param_groups[0]['lr']
        f_log.writelines('lr: {:.1e}\n'.format(optimizer.param_groups[0]['lr']))  
        # training process
        train_loss = 0.
        running_loss = 0.

        if epoch == 10:
            net = net.to(device=args.device, dtype=torch.float16)
            if args.model.upper() == 'TRAJGRU':
                args.batch_size = 4
            elif args.model.upper() == 'CONVGRU':
                args.batch_size = 8
            trainloader, testloader = get_dataloader(args)

        for i, data in enumerate(trainloader, 0):
            inputs = data['inputs'].to(device=args.device, dtype=args.value_dtype)  # inputs.shape = [batch_size, input_frames, input_channel, H, W]
            labels = data['targets'].to(device=args.device, dtype=args.value_dtype)  # labels.shape = [batch_size, target_frames, H, W]
            
            optimizer.zero_grad()
            
            outputs = net(inputs)                           # outputs.shape = [batch_size, target_frames, H, W]

            outputs = outputs.view(-1, outputs.shape[1]*outputs.shape[2]*outputs.shape[3])
            labels = labels.view(-1, labels.shape[1]*labels.shape[2]*labels.shape[3])

            if args.normalize_target:
                outputs = (outputs - args.min_values['QPE']) / (args.max_values['QPE'] - args.min_values['QPE'])
            
            # calculate loss function
            loss = loss_function(outputs, labels)
            # breakpoint()
            train_loss += loss.item()/len(trainloader)
            running_loss += loss.item()/40
            # optimize model
            
            # print('MAx outputs: {:.6f}, Loss: {:.6f}'.format(torch.max(outputs).item(), loss.item()))

            loss.backward()
            # 'clip_grad_norm' helps prevent the exploding gradient problem in RNNs or LSTMs.
            if args.clip:
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_max_norm)
            optimizer.step()

            # print training loss per 40 batches.
            if (i+1) % 40 == 0:
                # print out the training results.
                print('{}|  Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'.format(args.model, epoch+1, args.max_epochs, i+1, total_batches, running_loss))
                # print the trainging results to the log file.
                f_log.writelines('{}|  Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}\n'.format(args.model, epoch+1, args.max_epochs, i+1, total_batches, running_loss))
                running_loss = 0.
        
        # save the training results.
        result.iloc[epoch,0] = train_loss
        print('{}|  Epoch [{}/{}], Train Loss: {:8.3f}'.format(args.model, epoch+1, args.max_epochs, train_loss))

        # Save the test loss per epoch
        test_loss = test(net, testloader=testloader, loss_function=loss_function, args=args)
        # print out the testing results.
        print('{}|  Epoch [{}/{}], Test Loss: {:8.3f}'.format(args.model, epoch+1, args.max_epochs, test_loss))
        # save the testing results.
        result.iloc[epoch,1] = test_loss.item()

        # output results per 1 epoch.
        result.to_csv(result_file)

        time_b = time.time()
        print('The computing time of this epoch = {:.3f} sec'.format(time_b-time_a))
        print(('Max allocated memory:{:.3f}GB'.format(int(torch.cuda.max_memory_allocated(device=args.gpu)/1024/1024/1024))))
        f_log.writelines('The computing time of this epoch = {:.3f} sec\n'.format(time_b-time_a))
        f_log.writelines('Max allocated memory:{:.3f}GB\n'.format(int(torch.cuda.max_memory_allocated(device=args.gpu)/1024/1024/1024)))
        f_log.close()

        if (epoch+1) % 10 == 0 or (epoch+1) == args.max_epochs:
            # save the params per 10 epochs.
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss':loss},
                        params_pt
                        )

        if (epoch+1) == args.max_epochs:
            # counts the number of model weightings.
            total_params = sum(p.numel() for p in net.parameters())
            print('\{}|  Total_params: {:.2e}'.format(total_params))
            # save the number of model weightings.
            f_params = open(params_file, 'a')
            f_params.writelines('Total_params: {:.2e}\n'.format(total_params))
            f_params.close()

    print('Training process has finished!')
    
def test(net, testloader, loss_function, args):
    '''
    Arguments: this function is about to test the given model on test data.
    net(nn.Module): trained model
    testloader(Dataloader): the dataloader for test process.
    loss_function: loss function
    device: the device where the training process takes.
    '''
    # set evaluating process
    net.eval()
    loss = 0
    n_batch = len(testloader)

    with torch.no_grad():
        for _, data in enumerate(testloader, 0):
            inputs, labels = data['inputs'].to(args.device, dtype=args.value_dtype), data['targets'].to(args.device, dtype=args.value_dtype)
            outputs = net(inputs)
            if args.normalize_target:
                outputs = (outputs - args.min_values['QPE']) / (args.max_values['QPE'] - args.min_values['QPE'])
            loss += loss_function(outputs, labels)/n_batch

    return loss