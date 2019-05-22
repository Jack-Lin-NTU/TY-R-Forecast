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
from src.dataseters.mymodel import TyDataset, ToTensor, Normalize

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

def get_model(from_scratch=True, args=None):
    from src.operators.mymodel import my_single_GRU, my_multi_GRU
    from src.utils.mymodel_hyperparams import MYSINGLEMODEL_HYPERPARAMs, MYMULTIMODEL_HYPERPARAMs
    
    if args.model.upper() == 'MYSINGLEMODEL':
        MYMODEL = MYSINGLEMODEL_HYPERPARAMs(args)
        model = my_single_GRU(MYMODEL.input_frames, MYMODEL.target_frames, MYMODEL.TyCatcher_channel_input, MYMODEL.TyCatcher_channel_hidden, 
                            MYMODEL.TyCatcher_channel_n_layers, MYMODEL.gru_channel_input, MYMODEL.gru_channel_hidden, MYMODEL.gru_kernel, MYMODEL.gru_stride, MYMODEL.gru_padding, batch_norm=args.batch_norm, device=args.device, value_dtype=args.value_dtype).to(device=args.device, dtype=args.value_dtype)
    elif args.model.upper() == 'MYMULTIMODEL':
        MYMODEL = MYMULTIMODEL_HYPERPARAMs(args)
        model = my_multi_GRU(MYMODEL.input_frames, MYMODEL.target_frames, MYMODEL.TyCatcher_input, MYMODEL.TyCatcher_hidden, MYMODEL.TyCatcher_n_layers, 
                            MYMODEL.encoder_input, MYMODEL.encoder_downsample, MYMODEL.encoder_gru, MYMODEL.encoder_downsample_k, MYMODEL.encoder_downsample_s, 
                            MYMODEL.encoder_downsample_p, MYMODEL.encoder_gru_k, MYMODEL.encoder_gru_s, MYMODEL.encoder_gru_p, MYMODEL.encoder_n_cells, 
                            MYMODEL.forecaster_upsample_cin, MYMODEL.forecaster_upsample_cout, MYMODEL.forecaster_upsample_k, MYMODEL.forecaster_upsample_p, 
                            MYMODEL.forecaster_upsample_s, MYMODEL.forecaster_n_layers, MYMODEL.forecaster_output_cout, MYMODEL.forecaster_output_k, 
                            MYMODEL.forecaster_output_s, MYMODEL.forecaster_output_p, MYMODEL.forecaster_n_output_layers, 
                            batch_norm=args.batch_norm, device=args.device, value_dtype=args.value_dtype).to(device=args.device, dtype=args.value_dtype)
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
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[x for x in range(1, args.max_epochs) if x % 10 == 0], gamma=0.7)
    
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
            labels = data['targets'].to(device=args.device, dtype=args.value_dtype)  # labels.shape = [batch_size, target_frames, H, W]
            inputs = data['inputs'].to(device=args.device, dtype=args.value_dtype)  # inputs.shape = [batch_size, input_frames, input_channel, H, W]
            ty_infos = data['ty_infos'].to(device=args.device, dtype=args.value_dtype)
            radar_map = data['radar_map'].to(device=args.device, dtype=args.value_dtype)
            
            optimizer.zero_grad()
            outputs = model(encoder_inputs=inputs, ty_infos=ty_infos, radar_map=radar_map)    # outputs.shape = [batch_size, target_frames, H, W]

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
            # 'clip_grad_norm' helps prevent the exploding gradient problem in RNNs or LSTMs.
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
            labels = data['targets'].to(device=args.device, dtype=args.value_dtype)  # labels.shape = [batch_size, target_frames, H, W]
            inputs = data['inputs'].to(device=args.device, dtype=args.value_dtype)  # inputs.shape = [batch_size, input_frames, input_channel, H, W]
            ty_infos = data['ty_infos'].to(device=args.device, dtype=args.value_dtype)
            radar_map = data['radar_map'].to(device=args.device, dtype=args.value_dtype)
            
            outputs = model(encoder_inputs=inputs, ty_infos=ty_infos, radar_map=radar_map)    # outputs.shape = [batch_size, target_frames, H, W]
            if args.normalize_target:
                outputs = (outputs - args.min_values['QPE']) / (args.max_values['QPE'] - args.min_values['QPE'])
            loss += args.loss_function(outputs, labels)/n_batch

    return loss

