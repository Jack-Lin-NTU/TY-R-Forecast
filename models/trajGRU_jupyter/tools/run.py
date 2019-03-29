## import useful tools
import os
import time
import pandas as pd
pd.set_option('precision', 4)

## import torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils


# import our model and dataloader
from tools.datasetGRU import TyDataset, ToTensor, Normalize
from .args_tools import createfolder, remove_file


def get_dataloader(args):
    # transform
    transform = transforms.Compose([ToTensor(), Normalize(max_values=args.max_values, min_values=args.min_values)])

    # dataset
    traindataset = TyDataset(ty_list = args.ty_list,
                             input_frames = args.input_frames,
                             target_frames = args.target_frames,
                             train = True,
                             with_grid = args.input_with_grid,
                             transform = transform)

    testdataset = TyDataset(ty_list = args.ty_list,
                            input_frames = args.input_frames,
                            target_frames = args.target_frames,
                            train = False,
                            with_grid = args.input_with_grid,
                             transform = transform)
    # datloader
    trainloader = DataLoader(dataset=traindataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(dataset=testdataset, batch_size=args.batch_size, shuffle=False)
    
    return trainloader, testloader


def train(net, trainloader, testloader, args):
    '''

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
    optimizer = args.optimizer(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Set scheduler
    if args.lr_scheduler:
        # milestone = [int(((x+1)/10)*50) for x in range(9)]
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[x for x in range(1, args.max_epochs) if x % 5 == 0], gamma=0.4)
    
    total_batches = len(trainloader)
    
    # To declare a pd.DataFrame to store training, testing loss, and learning rate.
    result = pd.DataFrame([], index=pd.Index(range(1, args.max_epochs+1), name='epoch'), columns=['training_loss', 'testing_loss', 'lr'])

    for epoch in range(args.max_epochs):
        time_a = time.time()

        f_log = open(log_file, 'a')
        # set training process
        net.train()
        # update the learning rate
        if args.lr_scheduler:
            scheduler.step()
        # show the current learning rate (optimizer.param_groups returns a list which stores several params)
        print('lr: {:.1e}'.format(optimizer.param_groups[0]['lr']))
        # Save the learning rate per epoch
        result.iloc[epoch,2] = optimizer.param_groups[0]['lr']
        f_log.writelines('lr: {:.1e}\n'.format(optimizer.param_groups[0]['lr']))  
        # training process
        train_loss = 0
        for i, data in enumerate(trainloader):
#             print(i)
            inputs = data['input'].to(args.device, dtype=args.value_dtype)  # inputs.shape = [4,10,1,180,180]
            labels = data['target'].to(args.device, dtype=args.value_dtype)  # labels.shape = [4,18,60,60]

            outputs = net(inputs)                           # outputs.shape = [4, 18, 60, 60]

            outputs = outputs.view(outputs.shape[0], -1)    # outputs.shape = [4, 64800]
            labels = labels.view(labels.shape[0], -1)       # labels.shape = [4, 64800]

            # calculate loss function
            loss = args.loss_function(outputs, labels)
            train_loss += loss.item()

            # optimize model
            optimizer.zero_grad()
            loss.backward()
            # 'clip_grad_norm' helps prevent the exploding gradient problem in RNNs or LSTMs.
            if args.clip:
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_max_norm)
            optimizer.step()

            # print training loss per 40 batches.
            if (i+1) % 40 == 0:
                # print out the training results.
                print('trajGRU|  Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'.format(epoch+1, args.max_epochs, i+1, total_batches, loss.item()))
                # print the trainging results to the log file.
                f_log.writelines('trajGRU|  Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}\n'.format(epoch+1, args.max_epochs, i+1, total_batches, loss.item()))
        
        # calculate average training loss per 1 epoch.
        train_loss = train_loss/len(trainloader)
        # save the training results.
        result.iloc[epoch,0] = train_loss
        print('trajGRU|  Epoch [{}/{}], Train Loss: {:8.3f}'.format(epoch+1, args.max_epochs, train_loss))

        # Save the test loss per epoch
        test_loss = test(net, testloader=testloader, args=args)
        # print out the testing results.
        print('trajGRU|  Epoch [{}/{}], Test Loss: {:8.3f}'.format(epoch+1, args.max_epochs, test_loss))
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
            print('\trajGRU|  Total_params: {:.2e}'.format(total_params))
            # save the number of model weightings.
            f_params = open(params_file, 'a')
            f_params.writelines('Total_params: {:.2e}\n'.format(total_params))
            f_params.close()

    print('Training process has finished!')
    
def test(net, testloader, args):
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
            inputs, labels = data['input'].to(args.device, dtype=args.value_dtype), data['target'].to(args.device, dtype=args.value_dtype)
            outputs = net(inputs)
            outputs = outputs.view(outputs.shape[0], -1)
            labels = labels.view(labels.shape[0], -1)
            loss += args.loss_function(outputs, labels)

        loss = loss/n_batch
    return loss