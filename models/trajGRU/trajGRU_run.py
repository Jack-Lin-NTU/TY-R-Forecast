## import useful tools
import os
import sys
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
from tools.datasetGRU import ToTensor, Normalize, TyDataset
from tools.loss_function import BMAE, BMSE
from tools.trajGRU_model import model
from tools.args_tools import args, createfolder, remove_file

def train(net, trainloader, testloader, result_folder, params_folder, max_epochs=50, loss_function=BMSE, optimizer=optim.Adam, lr_scheduler=args.lr_scheduler, device=args.device):
    '''
    Arguments: This function is for training process.
    net(nn.module): the training model.
    trainloader(Dataloader): the dataloader for training process.
    testloader(Dataloader): the dataloader for test process.
    result_folder(str): the path of output folder for results.
    params_folder(str): the folder path of params files.
    max_epochs(int): the max epoch of training process.
    loss_function(function): loss function.
    optimizer(torch.nn.optim): the optimizer for training process.
    lr_scheduler: control the learning rate.
    device: the device where the training process takes.
    '''
    # set file path for saveing some info.
    log_file = os.path.join(result_folder, 'log.txt')
    result_file = os.path.join(result_folder, 'result.csv')
    params_file = os.path.join(result_folder, 'params_counts.csv')
    params_pt = os.path.join(params_folder, 'params.pt')

    remove_file(log_file)
    remove_file(result_file)
    remove_file(params_file)
    remove_file(params_pt)
    
    # set the optimizer (learning rate is from args)
    optimizer = optimizer(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Set scheduler
    if lr_scheduler:
        # milestone = [int(((x+1)/10)*50) for x in range(9)]
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20,25,30,35,40,45], gamma=0.4)
    
    total_batches = len(trainloader)
    
    # To declare a pd.DataFrame to store training, testing loss, and learning rate.
    result = pd.DataFrame([], index=range(1,max_epochs+1), columns=['training_loss','testing_loss','lr'])
    result.index.name = 'epoch'

    for epoch in range(max_epochs):
        time_a = time.time()

        f_log = open(log_file, 'a')
        # set training process
        net.train()
        # update the learning rate
        if lr_scheduler:
            scheduler.step()
        # show the current learning rate (optimizer.param_groups returns a list which stores several params)
        print('lr: {:.1e}'.format(optimizer.param_groups[0]['lr']))
        # Save the learning rate per epoch
        result.iloc[epoch,2] = optimizer.param_groups[0]['lr']
        f_log.writelines('lr: {:.1e}\n'.format(optimizer.param_groups[0]['lr']))  
        # training process
        train_loss = 0
        for i, data in enumerate(trainloader,0):
            
            inputs = data['RAD'].to(device, dtype=torch.float)  # inputs.shape = [4,10,1,180,180]
            labels = data['QPE'].to(device, dtype=torch.float)  # labels.shape = [4,18,60,60]
            
            outputs = net(inputs)                           # outputs.shape = [4, 18, 60, 60]

            outputs = outputs.view(outputs.shape[0], -1)    # outputs.shape = [4, 64800]
            labels = labels.view(labels.shape[0], -1)       # labels.shape = [4, 64800]

            # calculate loss function
            loss = loss_function(outputs, labels)
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
                print('ConvGRUv2|  Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'.format(epoch+1, max_epochs, i+1, total_batches, loss.item()))
                # print the trainging results to the log file.
                f_log.writelines('ConvGRUv2|  Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}\n'.format(epoch+1, max_epochs, i+1, total_batches, loss.item()))
        
        # calculate average training loss per 1 epoch.
        train_loss = train_loss/len(trainloader)
        # save the training results.
        result.iloc[epoch,0] = train_loss
        print('ConvGRUv2|  Epoch [{}/{}], Train Loss: {:8.3f}'.format(epoch+1, max_epochs, train_loss))

        # Save the test loss per epoch
        test_loss = test(net, testloader=testloader, loss_function=loss_function, device=device)
        # print out the testing results.
        print('ConvGRUv2|  Epoch [{}/{}], Test Loss: {:8.3f}'.format(epoch+1, max_epochs, test_loss))
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

        if (epoch+1) % 10 == 0 or (epoch+1) == max_epochs:
            # save the params per 10 epochs.
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss':loss},
                        params_pt
                        )

        if (epoch+1) == max_epochs:
            # counts the number of model weightings.
            total_params = sum(p.numel() for p in net.parameters())
            print('\nConvGRUv2|  Total_params: {:.2e}'.format(total_params))
            # save the number of model weightings.
            f_params = open(params_file, 'a')
            f_params.writelines('Total_params: {:.2e}\n'.format(total_params))
            f_params.close()

    print('Training process has finished!')

def test(net, testloader, loss_function=BMSE, device=args.device):
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
        for _, data in enumerate(testloader,0):
            inputs, labels = data['RAD'].to(device, dtype=torch.float), data['QPE'].to(device, dtype=torch.float)
            outputs = net(inputs)
            outputs = outputs.view(outputs.shape[0], -1)
            labels = labels.view(labels.shape[0], -1)
            loss += loss_function(outputs, labels)

        loss = loss/n_batch
    return loss

def get_dataloader(input_frames, output_frames, with_grid=False, normalize_target=False):
    '''
    Arguments: this function divides the data properly with given condition to provide training and testing dataloader.
    input_frames(int): the number of input frames for the constructed model.
    output_frames(int): the number of output frames for the constructed model.
    with_grid(boolean): add grid info to input frames or nor.
    normalize_target(boolean): to normalize output frames or not.
    '''
    # Normalize data
    mean = [7.044] * input_frames
    std = [12.180] * input_frames
    if normalize_target:
        mean += [1.122] * output_frames
        std += [3.858] * output_frames
    transfrom = transforms.Compose([ToTensor(),Normalize(mean=mean, std=std)])

    # set train and test dataset
    traindataset = TyDataset(ty_list_file=args.ty_list_file,
                            root_dir=args.root_dir,
                            input_frames=input_frames,
                            output_frames=output_frames,
                            train=True,
                            with_grid=with_grid,
                            transform = transfrom)
    testdataset = TyDataset(ty_list_file=args.ty_list_file,
                            root_dir=args.root_dir,
                            input_frames=input_frames,
                            output_frames=output_frames,
                            train=False,
                            with_grid=with_grid,
                            transform = transfrom)
    inputs_channels = traindataset[0]['RAD'].shape[1]
    # set train and test dataloader
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 1}
    trainloader = DataLoader(traindataset, **params)
    testloader = DataLoader(testdataset, batch_size=args.batch_size*10, shuffle=False)

    return inputs_channels, trainloader, testloader


def run(result_folder, params_folder, channel_factor, input_frames, output_frames, with_grid=args.input_with_grid,
        loss_function=BMSE, max_epochs=100, batch_norm=args.batch_norm, device=args.device):

    # get dataloader
    inputs_channels, trainloader, testloader = get_dataloader(input_frames, output_frames, with_grid=with_grid, normalize_target=args.normalize_target)

    # set the factor of cnn channels
    c = channel_factor

    ## construct Traj GRU
    # initialize the parameters of the encoders and forecasters

    rnn_link_size = [13,13,9]

    encoder_input_channel = inputs_channels
    encoder_downsample_channels = [2*c,32*c,96*c]
    encoder_rnn_channels = [32*c,96*c,96*c]

    decoder_input_channel = 0
    decoder_upsample_channels = [96*c,96*c,4*c]
    decoder_rnn_channels = [96*c,96*c,32*c]

    if int(args.I_shape[0]/3) == args.F_shape[0]:
        encoder_downsample_k = [5,4,4]
        encoder_downsample_s = [3,2,2]
        encoder_downsample_p = [1,1,1]
    elif args.I_shape[0] == args.F_shape[0]:
        encoder_downsample_k = [3,4,4]
        encoder_downsample_s = [1,2,2]
        encoder_downsample_p = [1,1,1]

    encoder_rnn_k = [3,3,3]
    encoder_rnn_s = [1,1,1]
    encoder_rnn_p = [1,1,1]
    encoder_n_layers = 6

    decoder_upsample_k = [4,4,3]
    decoder_upsample_s = [2,2,1]
    decoder_upsample_p = [1,1,1]

    decoder_rnn_k = [3,3,3]
    decoder_rnn_s = [1,1,1]
    decoder_rnn_p = [1,1,1]
    decoder_n_layers = 6

    decoder_output = 1
    decoder_output_k = 3
    decoder_output_s = 1
    decoder_output_p = 1
    decoder_output_layers = 1

    Net = model(n_encoders=input_frames, n_decoders=output_frames, rnn_link_size=rnn_link_size,  encoder_input_channel=encoder_input_channel, encoder_downsample_channels=encoder_downsample_channels, encoder_rnn_channels=encoder_rnn_channels, encoder_downsample_k=encoder_downsample_k, encoder_downsample_s=encoder_downsample_s, encoder_downsample_p=encoder_downsample_p, encoder_rnn_k=encoder_rnn_k,encoder_rnn_s=encoder_rnn_s, encoder_rnn_p=encoder_rnn_p, encoder_n_layers=encoder_n_layers, decoder_input_channel=decoder_input_channel, decoder_upsample_channels=decoder_upsample_channels, decoder_rnn_channels=decoder_rnn_channels, decoder_upsample_k=decoder_upsample_k, decoder_upsample_s=decoder_upsample_s, decoder_upsample_p=decoder_upsample_p, decoder_rnn_k=decoder_rnn_k, decoder_rnn_s=decoder_rnn_s, decoder_rnn_p=decoder_rnn_p, decoder_n_layers=decoder_n_layers, decoder_output=decoder_output, decoder_output_k=decoder_output_k, decoder_output_s=decoder_output_s, decoder_output_p=decoder_output_p, decoder_output_layers=decoder_output_layers, batch_norm=args.batch_norm).to(device, dtype=torch.float)

    info = '| Channel factor c: {:02d}, Forecast frames: {:02d}, Input frames: {:02d} |'.format(channel_factor, output_frames,input_frames)

    print('='*len(info))
    print(info)
    print('='*len(info))
    train(net=Net, trainloader=trainloader, testloader=testloader, result_folder=result_folder, params_folder=params_folder, max_epochs=max_epochs, loss_function=loss_function, lr_scheduler=args.lr_scheduler, device=device)

if __name__ == '__main__':
    # set the parameters of the experiment
    time_s = time.time()
    output_frames = args.output_frames
    channel_factor = args.channel_factor
    input_frames = args.input_frames

    mother_dir = 'trajGRU_i{:d}_o{:d}_c{:d}'.format(input_frames, output_frames, channel_factor)
    sub_dir = 'I{:d}_F{:d}'.format(args.I_shape[0], args.F_shape[0])

    if args.input_with_grid:
        sub_dir += '_with_grid'
    else:
        sub_dir += '_without_grid'

    if args.clip:
        sub_dir += '_clip'

    if args.batch_norm:
        sub_dir += '_batch_norm'

    args.result_dir = os.path.join(args.result_dir, mother_dir, sub_dir)
    args.params_dir = os.path.join(args.params_dir, mother_dir, sub_dir)
    createfolder(args.result_dir)
    createfolder(args.params_dir)
    ## test weight decay and clip max norm
    # i is a factor of weight decay, j is a factor to control the thresold norm of parameters in the model.
    for i in range(2,6):
        origin_wd = args.weight_decay
        if args.clip:
            for j in [1]:
                args.weight_decay = i*origin_wd
                args.clip_max_norm = int(j*args.clip_max_norm)
                args.result_folder = os.path.join(args.result_dir, 'BMSE_wd{:.4f}_cm{:02d}'.format(args.weight_decay, args.clip_max_norm))
                args.params_folder = os.path.join(args.params_dir, 'BMSE_wd{:.4f}_cm{:02d}'.format(args.weight_decay, args.clip_max_norm))
                print(' [The folder path of reslut files]:', args.result_folder)
                print(' [The folder path of params files]:', args.params_folder)
                createfolder(args.result_folder)
                createfolder(args.params_folder)

                run(result_folder=args.result_folder, params_folder=args.params_folder, channel_factor=channel_factor, input_frames=input_frames, output_frames=output_frames, loss_function=BMSE, max_epochs=args.max_epochs, batch_norm=args.batch_norm, device=args.device)
                time_e = time.time()
                t = time_e-time_s
                h = int((t)//3600)
                m = int((t-h*3600)//60)
                s = int(t-h*3600-m*60)

                print('The total computing time of this training process: {:d}:{:d}:{:d}'.format(h,m,s))
        else:
            args.weight_decay = 10**(-i)
            args.result_folder = os.path.join(args.result_dir, 'BMSE_wd{:.4f}'.format(args.weight_decay))
            args.params_folder = os.path.join(args.params_dir, 'BMSE_wd{:.4f}'.format(args.weight_decay))
            print(' [The folder path of reslut files]:', args.result_folder)
            print(' [The folder path of params files]:', args.params_folder)
            createfolder(args.result_folder)
            createfolder(args.params_folder)
            run(result_folder=args.result_folder, params_folder=args.params_folder, channel_factor=channel_factor, input_frames=input_frames, output_frames=output_frames, loss_function=BMSE, max_epochs=args.max_epochs, batch_norm=args.batch_norm, device=args.device)

            time_e = time.time()
            t = time_e-time_s
            h = int((t)//3600)
            m = int((t-h*3600)//60)
            s = int(t-h*3600-m*60)

            print('The total computing time of this training process: {:d}:{:d}:{:d}'.format(h,m,s))