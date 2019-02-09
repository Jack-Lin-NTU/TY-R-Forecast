## import some useful tools
import os
import sys
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import datetime as dt

## import torch module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

# import our model and dataloader
from tools.datasetCNN2D import ToTensor, Normalize, TyDataset
from tools.loss_function import BMAE, BMSE
from tools.cnn2D_model import model
from tools.args_tools import args, createfolder

def train(net, trainloader, testloader, result_name, max_epochs=50, loss_function=BMSE,
            optimizer=optim.Adam, lr_schedule=None, device=args.device):
    # create new files to record the results

    train_file = result_name
    test_file = result_name[:-4]+"_test.txt"

    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)

    # set the optimizer
    if lr_schedule:
        scheduler = lr_schedule

    optimizer = optimizer(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_step = len(trainloader)

    for epoch in range(max_epochs):
        # set training process
        net.train()
        # Training
        # open a new file to save result.
        f_train = open(train_file,"a")
        f_test = open(test_file,"a")

        for i, data in enumerate(trainloader,0):
            # inputs.shape = [batch_size, input_frames, 60, 60]
            # labels.shape = [batch_size, forecast_frames, 60, 60]
            inputs, labels = data["RAD"].to(device, dtype=torch.float), data["QPE"].to(device, dtype=torch.float)

            # outputs.shape = [batch_size, forecast_frames, 60, 60]
            outputs = net(inputs)

            # outputs.shape = [batch_size, forecast_frames*60*60]
            outputs = outputs.view(outputs.size(0), -1)

            # labels.shape = [batch_size, forecast_frames*60*60]
            labels = labels.view(labels.size(0), -1)

            # calculate loss function (divide by batch size and size of output frames)
            loss = loss_function(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 40 == 0:
                print('|CNN2D|  Epoch [{}/{}], Step [{:03}/{}], Loss: {:8.3f}'
                    .format(epoch+1, max_epochs, i+1, total_step, loss.item()))
                f_train.writelines('Epoch [{}/{}], Step [{:03}/{}], Loss: {:8.3f}\n'
                    .format(epoch+1, max_epochs, i+1, total_step, loss.item()))

        # Save the test loss per epoch
        test_loss = test(net, testloader=testloader, loss_function=loss_function, device=device)

        print("|CNN2D|  Epoch [{}/{}], Test Loss: {:8.3f}".format(epoch+1, max_epochs, test_loss))
        f_test.writelines("Epoch [{}/{}], Test Loss: {:8.3f}\n".format(epoch+1, max_epochs, test_loss))
        if (epoch+1) % 10 == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss':loss},
                        result_name[:-4]+'.pt')

        if (epoch+1) == max_epochs:
            total_params = sum(p.numel() for p in net.parameters())
            print("\CNN2D|  Total_params: {:.2e}".format(total_params))
            f_train.writelines("\nTotal_params: {:.2e}".format(total_params))

        f_train.close()
        f_test.close()
    print("Training has finished!")

def test(net, testloader, loss_function=BMSE,device=args.device):
    # set evaluating process
    net.eval()
    loss = 0
    n_batch = len(testloader)
    with torch.no_grad():
        for i, data in enumerate(testloader,0):
            inputs, labels = data["RAD"].to(device, dtype=torch.float), data["QPE"].to(device, dtype=torch.float)
            outputs = net(inputs)
            outputs = outputs.view(outputs.shape[0], -1)
            labels = labels.view(labels.shape[0], -1)
            loss += loss_function(outputs, labels)

        loss = loss/n_batch
    return loss

def get_dataloader(input_frames, output_frames):
    # Normalize data
    mean = [7.044] * input_frames
    std = [12.180] * input_frames
    transfrom = transforms.Compose([ToTensor(),Normalize(mean=mean, std=std)])

    # set train and test dataset
    traindataset = TyDataset(ty_list_file=args.ty_list_file,
                              root_dir=args.root_dir,
                              input_frames=input_frames,
                              output_frames=output_frames,
                              train=True,
                              transform = transforms.Compose([ToTensor(),Normalize(mean,std)]))

    testdataset = TyDataset(ty_list_file=args.ty_list_file,
                              root_dir=args.root_dir,
                              input_frames=input_frames,
                              output_frames=output_frames,
                              train=False,
                              transform = transforms.Compose([ToTensor(),Normalize(mean,std)]))


    # set train and test dataloader
    params = {"batch_size":args.batch_size, "shuffle":True, "num_workers":1}
    trainloader = DataLoader(traindataset, **params)
    testloader = DataLoader(testdataset, batch_size=args.batch_size*10, shuffle=False)

    return trainloader, testloader

# Run exp1
def run(result_name, channel_factor=70, input_frames=5, output_frames=18, loss_function=BMSE, max_epochs=50, batch_norm=args.batch_norm device=args.device):

    # get dataloader
    trainloader, testloader = get_dataloader(input_frames, output_frames)

    # set the factor of cnn channels
    c = channel_factor

    # Make CNN2D Net
    encoder_input = input_frames
    encoder_hidden = [c,8*c,12*c,16*c]
    encoder_kernel = [4,4,3,2]
    encoder_n_layer = 4
    encoder_stride = [2,2,2,2]
    encoder_padding = [1,1,1,1]

    decoder_input = 16*c
    decoder_hidden = [16*c,16*c,8*c,4*c,24,output_frames]
    decoder_kernel = [1,2,3,4,4,3]
    decoder_n_layer = 6
    decoder_stride = [1,2,2,2,2,1]
    decoder_padding = [0,1,1,1,1,1]


    Net = model(encoder_input, encoder_hidden, encoder_kernel, encoder_n_layer, encoder_stride, encoder_padding,
                decoder_input, decoder_hidden, decoder_kernel, decoder_n_layer, decoder_stride, decoder_padding,
                batch_norm=batch_norm).to(device)

    # print(Net)
    # Train process
    info = "|CNN2D| Forecast frames: {:02d}, Input frames: {:02d} |".format(output_frames, input_frames)
    print("="*len(info))
    print(info)
    print("="*len(info))
    train(net=Net, trainloader=trainloader, testloader=testloader, result_name=result_name, max_epochs=max_epochs,  loss_function=loss_function, device=device)

if __name__ == "__main__":
    output_frames = args.output_frames
    channel_factor = args.channel_factor
    input_frames = args.input_frames
    result_dir=os.path.join(args.result_dir,
                            'cnn2D_i{:d}_o{:d}_c{:d}'.format(input_frames,output_frames,channel_factor),
                            'I{:d}_F{:d}'.format(args.I_shape[0],args.F_shape[0]))

    print(" [The path of the result folder]:", result_dir)
    createfolder(result_dir)
    result_name = os.path.join(result_dir,"BMSE_f.{:02d}_x.{:02d}.txt".format(output_frames, input_frames))
    print(result_name)
    run(result_name=result_name, channel_factor=channel_factor, input_frames=input_frames, output_frames=output_frames,
        loss_function=BMSE, max_epochs=50, batch_norm=args.batch_norm, device=args.device)
