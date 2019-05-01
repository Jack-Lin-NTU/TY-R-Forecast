## import useful tools
import os
import time
import numpy as np
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
from src.argstools.argstools import args, createfolder, remove_file, loss_rainfall, Adam16
from src.models.trajGRU_same_warp import Model
if args.load_all_data:
    from src.dataseters.trajGRU_all_data import TyDataset, ToTensor, Normalize
else:
    from src.dataseters.trajGRU import TyDataset, ToTensor, Normalize

# set seed 
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def get_dataloader(args):
    '''
    This function is used to get dataloaders.
    '''
    # transform
    transform = transforms.Compose([ToTensor(), Normalize(args)])
    
    traindataset = TyDataset(args=args, train = True, transform=transform)
    testdataset = TyDataset(args=args, train = False, transform=transform)
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
    if args.optimizer is optim.Adam:
        optimizer = args.optimizer(net.parameters(), lr=args.lr, eps=1e-07, weight_decay=args.weight_decay)
    elif args.optimizer is Adam16:
        optimizer = args.optimizer(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, device=args.device)
    else:
        optimizer = args.optimizer(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Set scheduler
    if args.lr_scheduler and args.optimizer is not optim.Adam:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[x for x in range(1, args.max_epochs) if x % 5 == 0], gamma=0.7)
    
    total_batches = len(trainloader)
    
    # To declare a pd.DataFrame to store training, testing loss, and learning rate.
    result = pd.DataFrame([], index=pd.Index(range(1, args.max_epochs+1), name='epoch'), columns=['training_loss', 'testing_loss', 'lr'])

    for epoch in range(args.max_epochs):
        time_a = time.time()

        f_log = open(log_file, 'a')
        # set training process
        net.train()
        
        # update the learning rate
        if args.lr_scheduler and args.optimizer is not optim.Adam:
            scheduler.step()
        # show the current learning rate (optimizer.param_groups returns a list which stores several params)
        print('lr: {:.1e}'.format(optimizer.param_groups[0]['lr']))
        # Save the learning rate per epoch
        result.iloc[epoch,2] = optimizer.param_groups[0]['lr']
        f_log.writelines('lr: {:.1e}\n'.format(optimizer.param_groups[0]['lr']))  
        # training process
        train_loss = 0

        for i, data in enumerate(trainloader, 0):
            inputs = data['inputs'].to(args.device, dtype=args.value_dtype)  # inputs.shape = [batch_size, input_frames, input_channel, H, W]
            labels = data['targets'].to(args.device, dtype=args.value_dtype)  # labels.shape = [batch_size, target_frames, H, W]
            
            outputs = net(inputs)                           # outputs.shape = [batch_size, target_frames, H, W]
            
            if args.normalize_target:
                outputs = (outputs - args.min_values['QPE']) / (args.max_values['QPE'] - args.min_values['QPE'])
            

            # calculate loss function=
            loss = loss_function(outputs, labels)
            train_loss += loss.item()/len(trainloader)
            # optimize model
            optimizer.zero_grad()
            loss.backward()
            # 'clip_grad_norm' helps prevent the exploding gradient problem in RNNs or LSTMs.
            if args.clip:
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_max_norm)

            optimizer.step()

            # print training loss per 40 batches.
            if (i+1) % 10 == 0:
                # print out the training results.
                print('trajGRU|  Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'.format(epoch+1, args.max_epochs, i+1, total_batches, loss.item()))
                # print the trainging results to the log file.
                f_log.writelines('trajGRU|  Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}\n'.format(epoch+1, args.max_epochs, i+1, total_batches, loss.item()))
        
        # save the training results.
        result.iloc[epoch,0] = train_loss
        print('trajGRU|  Epoch [{}/{}], Train Loss: {:8.3f}'.format(epoch+1, args.max_epochs, train_loss))

        # Save the test loss per epoch
        test_loss = test(net, testloader=testloader, loss_function=loss_function, args=args)
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
            loss += loss_function(outputs, labels)

        loss = loss/n_batch
    return loss

if __name__ == '__main__':
    # get trainloader and testloader
    trainloader, testloader = get_dataloader(args)
    # breakpoint()
    # initilize model

    # set the factor of cnn channels
    c = args.channel_factor

    ## construct Traj GRU
    # initialize the parameters of the encoders and forecasters
    rnn_link_size = [13, 13, 9]

    encoder_input_channel = args.input_channels
    encoder_downsample_channels = [4*c,32*c,96*c]
    encoder_rnn_channels = [32*c,96*c,96*c]

    if args.I_shape[0] == args.F_shape[0]*3:
        encoder_downsample_k = [5,4,3]
        encoder_downsample_s = [3,2,2]
        encoder_downsample_p = [1,1,1]
    elif args.I_shape[0] == args.F_shape[0]:
        encoder_downsample_k = [7,5,4]
        encoder_downsample_s = [5,3,2]
        encoder_downsample_p = [1,1,1]

    encoder_rnn_k = [3,3,3]
    encoder_rnn_s = [1,1,1]
    encoder_rnn_p = [1,1,1]
    encoder_n_layers = 6

    forecaster_input_channel = 0
    forecaster_upsample_channels = [96*c,96*c,4*c]
    forecaster_rnn_channels = [96*c,96*c,32*c]

    forecaster_upsample_k = [4,5,7]
    forecaster_upsample_s = [2,3,5]
    forecaster_upsample_p = [1,1,1]

    forecaster_rnn_k = encoder_rnn_k
    forecaster_rnn_s = encoder_rnn_s
    forecaster_rnn_p = encoder_rnn_p
    forecaster_n_layers = encoder_n_layers

    forecaster_output_channels = 1
    forecaster_output_k = 3
    forecaster_output_s = 1
    forecaster_output_p = 1
    forecaster_output_layers = 1

    Net = Model(n_encoders=args.input_frames, n_forecasters=args.target_frames, rnn_link_size=rnn_link_size,
                encoder_input_channel=encoder_input_channel, encoder_downsample_channels=encoder_downsample_channels,
                encoder_rnn_channels=encoder_rnn_channels, encoder_downsample_k=encoder_downsample_k,
                encoder_downsample_s=encoder_downsample_s, encoder_downsample_p=encoder_downsample_p, 
                encoder_rnn_k=encoder_rnn_k,encoder_rnn_s=encoder_rnn_s, encoder_rnn_p=encoder_rnn_p, 
                encoder_n_layers=encoder_n_layers, forecaster_input_channel=forecaster_input_channel, 
                forecaster_upsample_channels=forecaster_upsample_channels, forecaster_rnn_channels=forecaster_rnn_channels,
                forecaster_upsample_k=forecaster_upsample_k, forecaster_upsample_s=forecaster_upsample_s, 
                forecaster_upsample_p=forecaster_upsample_p, forecaster_rnn_k=forecaster_rnn_k, forecaster_rnn_s=forecaster_rnn_s,
                forecaster_rnn_p=forecaster_rnn_p, forecaster_n_layers=forecaster_n_layers, forecaster_output=forecaster_output_channels, 
                forecaster_output_k=forecaster_output_k, forecaster_output_s=forecaster_output_s, 
                forecaster_output_p=forecaster_output_p, forecaster_output_layers=forecaster_output_layers, 
                batch_norm=args.batch_norm, device=args.device, value_dtype=args.value_dtype).to(args.device, dtype=args.value_dtype)

    # train process
    time_s = time.time()

    size = '{}X{}'.format(args.I_shape[0], args.I_shape[1])

    if args.weather_list == []:
        args.result_folder = os.path.join(args.result_folder, size, 'RAD_no_weather')
        args.params_folder = os.path.join(args.params_folder, size, 'RAD_no_weather')
    else:
        args.result_folder = os.path.join(args.result_folder, size, 'RAD_weather')
        args.params_folder = os.path.join(args.params_folder, size, 'RAD_weather')

    args.result_folder = os.path.join(args.result_folder, 'wd{:.5f}_lr{:f}'.format(args.weight_decay, args.lr))
    args.params_folder = os.path.join(args.params_folder, 'wd{:.5f}_lr{:f}'.format(args.weight_decay, args.lr))

    if args.lr_scheduler and args.optimizer is not optim.Adam:
        args.result_folder += '_scheduler'
        args.params_folder += '_scheduler'
    
    if args.optimizer is optim.Adam:
        args.result_folder += '_Adam'
        args.params_folder += '_Adam'

    if args.loss_function == 'BMSE':
        loss_function = loss_rainfall(max_values=args.max_values, min_values=args.min_values).bmse
    elif args.loss_function == 'BMAE':
        loss_function = loss_rainfall(max_values=args.max_values, min_values=args.min_values).bmae

    train(net=Net, trainloader=trainloader, testloader=testloader, loss_function=loss_function, args=args)

    time_e = time.time()
    t = time_e-time_s
    h = int((t)//3600)
    m = int((t-h*3600)//60)
    s = int(t-h*3600-m*60)

    print('The total computing time of this training process: {:d}:{:d}:{:d}'.format(h,m,s))
