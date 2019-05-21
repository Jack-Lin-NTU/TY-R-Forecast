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
from src.utils.utils import createfolder, remove_file, Adam16
from src.utils.GRUs_hyperparams import TRAJGRU_HYPERPARAMs, CONVGRU_HYPERPARAMs
from src.dataseters.GRUs import TyDataset, ToTensor, Normalize

def get_dataloader(args, train_num=None):
    '''
    This function is used to get trainloader and testloader.
    '''
    # transform
    transform = transforms.Compose([ToTensor(), Normalize(args)])
    if train_num is None:
        train_num = args.train_num

    traindataset = TyDataset(args=args, train=True, train_num=train_num, transform=transform)
    testdataset = TyDataset(args=args, train=False, train_num=train_num, transform=transform)

    # dataloader
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.able_cuda else {}
    trainloader = DataLoader(dataset=traindataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testloader = DataLoader(dataset=testdataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    return trainloader, testloader


def get_model(args=None):
    if args.model.upper() == 'TRAJGRU':
        from src.operators.trajGRU import Multi_unit_Model as Model
        print('Model: TRAJGRU')
        TRAJGRU = TRAJGRU_HYPERPARAMs(args=args)
        model = Model(n_encoders=args.input_frames, n_forecasters=args.target_frames, gru_link_size=TRAJGRU.gru_link_size,
                encoder_input_channel=TRAJGRU.encoder_input_channel, encoder_downsample_channels=TRAJGRU.encoder_downsample_channels,
                encoder_gru_channels=TRAJGRU.encoder_gru_channels, encoder_downsample_k=TRAJGRU.encoder_downsample_k,
                encoder_downsample_s=TRAJGRU.encoder_downsample_s, encoder_downsample_p=TRAJGRU.encoder_downsample_p, 
                encoder_gru_k=TRAJGRU.encoder_gru_k, encoder_gru_s=TRAJGRU.encoder_gru_s, encoder_gru_p=TRAJGRU.encoder_gru_p, 
                encoder_n_cells=TRAJGRU.encoder_n_cells, forecaster_input_channel=TRAJGRU.forecaster_input_channel, 
                forecaster_upsample_channels=TRAJGRU.forecaster_upsample_channels, forecaster_gru_channels=TRAJGRU.forecaster_gru_channels,
                forecaster_upsample_k=TRAJGRU.forecaster_upsample_k, forecaster_upsample_s=TRAJGRU.forecaster_upsample_s, 
                forecaster_upsample_p=TRAJGRU.forecaster_upsample_p, forecaster_gru_k=TRAJGRU.forecaster_gru_k, forecaster_gru_s=TRAJGRU.forecaster_gru_s,
                forecaster_gru_p=TRAJGRU.forecaster_gru_p, forecaster_n_cells=TRAJGRU.forecaster_n_cells, forecaster_output=TRAJGRU.forecaster_output_channels, 
                forecaster_output_k=TRAJGRU.forecaster_output_k, forecaster_output_s=TRAJGRU.forecaster_output_s, 
                forecaster_output_p=TRAJGRU.forecaster_output_p, forecaster_output_layers=TRAJGRU.forecaster_output_layers, 
                batch_norm=args.batch_norm, device=args.device, value_dtype=args.value_dtype).to(args.device, dtype=args.value_dtype)

    elif args.model.upper() == 'CONVGRU':
        from src.operators.convGRU import Multi_unit_Model as Model
        print('Model: CONVGRU')
        CONVGRU = CONVGRU_HYPERPARAMs(args=args)
        model = Model(n_encoders=args.input_frames, n_forecasters=args.target_frames,
                encoder_input_channel=CONVGRU.encoder_input_channel, encoder_downsample_channels=CONVGRU.encoder_downsample_channels,
                encoder_gru_channels=CONVGRU.encoder_gru_channels, encoder_downsample_k=CONVGRU.encoder_downsample_k,
                encoder_downsample_s=CONVGRU.encoder_downsample_s, encoder_downsample_p=CONVGRU.encoder_downsample_p, 
                encoder_gru_k=CONVGRU.encoder_gru_k,encoder_gru_s=CONVGRU.encoder_gru_s, encoder_gru_p=CONVGRU.encoder_gru_p, 
                encoder_n_cells=CONVGRU.encoder_n_cells, forecaster_input_channel=CONVGRU.forecaster_input_channel, 
                forecaster_upsample_channels=CONVGRU.forecaster_upsample_channels, forecaster_gru_channels=CONVGRU.forecaster_gru_channels,
                forecaster_upsample_k=CONVGRU.forecaster_upsample_k, forecaster_upsample_s=CONVGRU.forecaster_upsample_s, 
                forecaster_upsample_p=CONVGRU.forecaster_upsample_p, forecaster_gru_k=CONVGRU.forecaster_gru_k, forecaster_gru_s=CONVGRU.forecaster_gru_s,
                forecaster_gru_p=CONVGRU.forecaster_gru_p, forecaster_n_cells=CONVGRU.forecaster_n_cells, forecaster_output=CONVGRU.forecaster_output_channels, 
                forecaster_output_k=CONVGRU.forecaster_output_k, forecaster_output_s=CONVGRU.forecaster_output_s, 
                forecaster_output_p=CONVGRU.forecaster_output_p, forecaster_output_layers=CONVGRU.forecaster_output_layers, 
                batch_norm=args.batch_norm, device=args.device, value_dtype=args.value_dtype).to(args.device, dtype=args.value_dtype)
    return model


def train(model, trainloader, testloader, args):
    '''
    This function is to train the model.
    '''
    # set file path for saveing some info.
    createfolder(args.result_folder)
    createfolder(args.params_folder)
    
    log_file = os.path.join(args.result_folder, 'log.txt')
    result_file = os.path.join(args.result_folder, 'result_df.csv')
    params_file = os.path.join(args.result_folder, 'params_counts.csv')
    remove_file(log_file)
    remove_file(result_file)
    remove_file(params_file)
    
    # set optimizer
    if args.optimizer == 'Adam16':
        optimizer = Adam16(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, device=args.device)
    else:
        optimizer = getattr(optim, args.optimizer)
        if args.optimizer == 'Adam16':
            optimizer = optimizer(model.parameters(), lr=args.lr, eps=1e-07, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optimizer(model.parameters(), lr=args.lr, momentum=0.6, weight_decay=args.weight_decay)
        else:
            optimizer = optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Set scheduler
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[x for x in range(1, args.max_epochs) if x % 7 == 0], gamma=0.7)
    
    total_batches = len(trainloader)
    
    # To create a pd.DataFrame to store training, validating loss, and learning rate.
    result_df = pd.DataFrame([], index=pd.Index(range(1, args.max_epochs+1), name='epoch'), columns=['train_loss', 'val_loss', 'lr'])

    for epoch in range(args.max_epochs):
        # turn on train mode
        model.train(True)

        # store time
        time1 = time.time()

        # open the log file in append mode
        f_log = open(log_file, 'a')
        
        # update the learning rate
        if args.lr_scheduler:
            scheduler.step()

        # show the current learning rate (optimizer.param_groups returns a list which stores several hyper-params)
        print('lr: {:.1e}'.format(optimizer.param_groups[0]['lr']))
        # Save the learning rate per epoch
        result_df.iloc[epoch, 2] = optimizer.param_groups[0]['lr']
        f_log.writelines('lr: {:.1e}\n'.format(optimizer.param_groups[0]['lr']))
        
        # initilaize loss
        train_loss = 0.
        running_loss = 0.
        
        ## change the value dtype after training 10 epochs
        if args.change_value_dtype and epoch == 10:
            args.value_dtype = torch.float16
            model = model.to(device=args.device, dtype=args.value_dtype)
            model.modify_value_dtype_(value_dtype=args.value_dtype)
            if args.model.upper() == 'TRAJGRU':
                args.batch_size = 4
            elif args.model.upper() == 'CONVGRU':
                args.batch_size = 8
            trainloader, testloader = get_dataloader(args)

            # set optimizer
            if args.optimizer == 'Adam16':
                optimizer = Adam16(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, device=args.device)
            else:
                optimizer = getattr(optim, args.optimizer)
                if args.optimizer == 'Adam16':
                    optimizer = optimizer(model.parameters(), lr=args.lr, eps=1e-07, weight_decay=args.weight_decay)
                elif args.optimizer == 'SGD':
                    optimizer = optimizer(model.parameters(), lr=args.lr, momentum=0.6, weight_decay=args.weight_decay)
                else:
                    optimizer = optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for idx, data in enumerate(trainloader, 0):
            inputs = data['inputs'].to(device=args.device, dtype=args.value_dtype)  # inputs.shape = [batch_size, input_frames, input_channel, H, W]
            labels = data['targets'].to(device=args.device, dtype=args.value_dtype)  # labels.shape = [batch_size, target_frames, H, W]

            optimizer.zero_grad()
            
            outputs = model(inputs)                           # outputs.shape = [batch_size, target_frames, H, W]

            outputs = outputs.view(-1, outputs.shape[1]*outputs.shape[2]*outputs.shape[3])
            labels = labels.view(-1, labels.shape[1]*labels.shape[2]*labels.shape[3])

            if args.normalize_target:
                outputs = (outputs - args.min_values['QPE']) / (args.max_values['QPE'] - args.min_values['QPE'])
            
            # calculate loss function
            loss = args.loss_function(outputs, labels)
            # breakpoint()
            train_loss += loss.item()/len(trainloader)
            running_loss += loss.item()/40
            # optimize model
            
            # print('Max outputs: {:.6f}, Loss: {:.6f}'.format(torch.max(outputs).item(), loss.item()))

            loss.backward()
            # 'clip_grad_norm' helps prevent the exploding gradient problem in grus or LSTMs.
            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_max_norm)
            optimizer.step()

            # print training loss per 40 batches.
            if (idx+1) % 40 == 0:
                # print out the training results.
                print('{}|  Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'.format(args.model, epoch+1, args.max_epochs, idx+1, total_batches, running_loss))
                # print the trainging results to the log file.
                f_log.writelines('{}|  Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}\n'.format(args.model, epoch+1, args.max_epochs, idx+1, total_batches, running_loss))
                running_loss = 0.
        
        # save the training results.
        result_df.iloc[epoch,0] = train_loss
        print('{}|  Epoch [{}/{}], Train Loss: {:8.3f}'.format(args.model, epoch+1, args.max_epochs, train_loss))

        # Save the test loss per epoch
        test_loss = test(model, testloader=testloader, args=args)
        # print out the testing results.
        print('{}|  Epoch [{}/{}], Test Loss: {:8.3f}'.format(args.model, epoch+1, args.max_epochs, test_loss))
        # save the testing results.
        result_df.iloc[epoch,1] = test_loss.item()

        # output results per 1 epoch.
        result_df.to_csv(result_file)

        time2 = time.time()
        print('The computing time of this epoch = {:.3f} sec'.format(time2-time1))
        print(('Max allocated memory:{:.3f}GB'.format(int(torch.cuda.max_memory_allocated(device=args.gpu)/1024/1024/1024))))
        f_log.writelines('The computing time of this epoch = {:.3f} sec\n'.format(time2-time1))
        f_log.writelines('Max allocated memory:{:.3f}GB\n'.format(int(torch.cuda.max_memory_allocated(device=args.gpu)/1024/1024/1024)))
        f_log.close()

        if (epoch+1) % 10 == 0 or (epoch+1) == args.max_epochs:
            params_pt = os.path.join(args.params_folder, 'params_{}.pt'.format(epoch+1))
            remove_file(params_pt)
            # save the params per 10 epochs.
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss':loss},
                        params_pt
                        )

        if (epoch+1) == args.max_epochs:
            # counts the number of model weightings.
            total_params = sum(p.numel() for p in model.parameters())
            print('\{}|  Total_params: {:.2e}'.format(args.model, total_params))
            # save the number of model weightings.
            f_params = open(params_file, 'a')
            f_params.writelines('Total_params: {:.2e}\n'.format(total_params))
            f_params.close()

    print('Training process has finished!')
    
def test(model, testloader, args):
    '''
    Arguments: this function is about to test the given model on test data.
    model(nn.Module): trained model
    testloader(Dataloader): the dataloader for test process.
    loss_function: loss function
    device: the device where the training process takes.
    '''
    # set evaluating process
    model.eval()
    loss = 0
    n_batch = len(testloader)

    with torch.no_grad():
        for _, data in enumerate(testloader, 0):
            inputs, labels = data['inputs'].to(args.device, dtype=args.value_dtype), data['targets'].to(args.device, dtype=args.value_dtype)
            outputs = model(inputs)
            if args.normalize_target:
                outputs = (outputs - args.min_values['QPE']) / (args.max_values['QPE'] - args.min_values['QPE'])
            loss += args.loss_function(outputs, labels)/n_batch

    return loss

