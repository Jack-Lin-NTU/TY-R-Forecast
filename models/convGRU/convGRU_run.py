## import useful tools
import os
import sys

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
from tools.convGRU_model import model
from tools.args_tools import args, createfolder

def train(net, trainloader, testloader, result_name, max_epochs=50, loss_function=BMSE,
          optimizer=optim.Adam, lr_schedule=None, device=args.device):

    train_file = result_name
    test_file = result_name[:-4]+"_test.txt"
    params_file = result_name[:-4]+"_params.txt"

    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)

    # Set optimizer
    if lr_schedule:
        scheduler = lr_schedule
    optimizer = optimizer(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_step = len(trainloader)

    for epoch in range(max_epochs):
        # set training process
        net.train()
        # Training
        # open a new file to save result.
        f_test = open(test_file,"a")

        for i, data in enumerate(trainloader,0):
            # inputs.shape = [4, 10, 1, 180, 180]
            # labels.shape = [4, 18, 60, 60]
            inputs = data["RAD"].to(device, dtype=torch.float)
            labels = data["QPE"].to(device, dtype=torch.float)

            # outputs.shape = [4, 18, 60, 60]
            outputs = net(inputs)

            # outputs.shape = [4, 64800]
            outputs = outputs.view(outputs.shape[0], -1)

            # outputs.shape = [4, 64800]
            labels = labels.view(labels.shape[0], -1)

            # calculate loss function
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
        #     # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #     nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_max_norm)

            optimizer.step()
            if (i+1) % 40 == 0:
                f_train = open(train_file,"a")
                print('ConvGRUv2|  Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'
                    .format(epoch+1, max_epochs, i+1, total_step, loss.item()))
                f_train.writelines('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}\n'
                    .format(epoch+1, max_epochs, i+1, total_step, loss.item()))
                f_train.close()

        # Save the test loss per epoch
        test_loss = test(net, testloader=testloader, loss_function=loss_function, device=device)

        print("ConvGRUv2|  Epoch [{}/{}], Test Loss: {:8.3f}".format(epoch+1, max_epochs, test_loss))
        f_test.writelines("Epoch [{}/{}], Test Loss: {:8.3f}\n".format(epoch+1, max_epochs, test_loss))
        f_test.close()

        if (epoch+1) % 10 == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss':loss},
                        result_name[:-4]+'.pt')

        if (epoch+1) == max_epochs:
            total_params = sum(p.numel() for p in net.parameters())
            print("\nConvGRUv2|  Total_params: {:.2e}".format(total_params))
            f_params = open(params_file, "a")
            f_params.writelines("ConvGRUv2|  Total_params: {:.2e}".format(total_params))
            f_params.close()

    print("Training has finished!")

def test(net, testloader, loss_function=BMSE, device=args.device):
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

def get_dataloader(input_frames, output_frames, with_grid=False, normalize_tartget=False):
    # Normalize data
    mean = [7.044] * input_frames
    std = [12.180] * input_frames
    if normalize_tartget:
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
    params = {"batch_size":args.batch_size, "shuffle":True, "num_workers":1}
    trainloader = DataLoader(traindataset, **params)
    testloader = DataLoader(testdataset, batch_size=args.batch_size*10, shuffle=False)

    return inputs_channels, trainloader, testloader


def run(result_name, channel_factor, input_frames, output_frames, with_grid=args.input_with_grid,
        loss_function=BMSE, max_epochs=100, batch_norm=args.batch_norm, device=args.device):

    # get dataloader
    inputs_channels, trainloader, testloader = get_dataloader(input_frames, output_frames, with_grid=with_grid,
                                                                normalize_tartget=args.normalize_tartget)

    # set the factor of cnn channels
    c = channel_factor
    # construct convGRU net
    # initialize the parameters of the encoders and decoders
    encoder_input = inputs_channels
    encoder_downsample_layer = [2*c,32*c,96*c]
    encoder_crnn_layer = [32*c,96*c,96*c]

    if int(args.I_shape[0]/3) == args.F_shape[0]:
        encoder_downsample_k = [5,4,4]
        encoder_downsample_s = [3,2,2]
        encoder_downsample_p = [1,1,1]
    elif args.I_shape[0] == args.F_shape[0]:
        encoder_downsample_k = [3,4,4]
        encoder_downsample_s = [1,2,2]
        encoder_downsample_p = [1,1,1]


    encoder_crnn_k = [3,3,3]
    encoder_crnn_s = [1,1,1]
    encoder_crnn_p = [1,1,1]
    encoder_n_layers = 6

    decoder_input=0
    decoder_upsample_layer = [96*c,96*c,4*c]
    decoder_crnn_layer = [96*c,96*c,32*c]

    decoder_upsample_k = [4,4,3]
    decoder_upsample_s = [2,2,1]
    decoder_upsample_p = [1,1,1]

    decoder_crnn_k = [3,3,3]
    decoder_crnn_s = [1,1,1]
    decoder_crnn_p = [1,1,1]
    decoder_n_layers = 6

    decoder_output = 1
    decoder_output_k = 3
    decoder_output_s = 1
    decoder_output_p = 1
    decoder_output_layers = 1

    Net = model(n_encoders=input_frames, n_decoders=output_frames,
                encoder_input=encoder_input, encoder_downsample_layer=encoder_downsample_layer, encoder_crnn_layer=encoder_crnn_layer,
                encoder_downsample_k=encoder_downsample_k, encoder_downsample_s=encoder_downsample_s, encoder_downsample_p=encoder_downsample_p,
                encoder_crnn_k=encoder_crnn_k,encoder_crnn_s=encoder_crnn_s, encoder_crnn_p=encoder_crnn_p, encoder_n_layers=encoder_n_layers,
                decoder_input=decoder_input, decoder_upsample_layer=decoder_upsample_layer, decoder_crnn_layer=decoder_crnn_layer,
                decoder_upsample_k=decoder_upsample_k, decoder_upsample_s=decoder_upsample_s, decoder_upsample_p=decoder_upsample_p,
                decoder_crnn_k=decoder_crnn_k, decoder_crnn_s=decoder_crnn_s, decoder_crnn_p=decoder_crnn_p, decoder_n_layers=decoder_n_layers,
                decoder_output=1, decoder_output_k=decoder_output_k, decoder_output_s=decoder_output_s, decoder_output_p=deco100der_output_p,
                decoder_output_layers=decoder_output_layers, batch_norm=args.batch_norm).to(device, dtype=torch.float)
    info = "| Channel factor c: {:02d}, Forecast frames: {:02d}, Input frames: {:02d} |".format(channel_factor, output_frames,input_frames)

    print("="*len(info))
    print(info)
    print("="*len(info))
    train(net=Net, trainloader=trainloader, testloader=testloader, result_name=result_name,
            max_epochs=max_epochs, loss_function=loss_function, device=device)

if __name__ == "__main__":
    # set the parameters of the experiment
    output_frames = args.output_frames
    channel_factor = args.channel_factor
    input_frames = args.input_frames

    mother_dir = 'convGRU_i{:d}_o{:d}_c{:d}'.format(input_frames,output_frames,channel_factor)
    sub_dir = 'I{:d}_F{:d}'.format(args.I_shape[0],args.F_shape[0])

    if args.input_with_grid:
        sub_dir += '_with_grid'
    else:
        sub_dir += '_with_no_grid'

    if args.batch_norm:
        sub_dir += '_batch_norm'

    result_dir=os.path.join(args.result_dir,mother_dir,sub_dir)

    print(" [The path of the result folder]:", result_dir)
    createfolder(result_dir)
    result_name = os.path.join(result_dir,"BMSE_f{:02d}_x{:02d}_w{:.2f}.txt".format(output_frames,input_frames,args.weight_decay))
    print(" [The name of the result file]:", result_name)
    run(result_name=result_name, channel_factor=channel_factor, input_frames=input_frames, output_frames=output_frames,
        loss_function=BMSE, max_epochs=args.max_epochs, batch_norm=args.batch_norm, device=args.device)
