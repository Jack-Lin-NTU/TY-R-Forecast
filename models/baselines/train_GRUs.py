## import useful tools
import os
import time
import numpy as np
import pandas as pd
pd.set_option('precision', 4)

## import torch modules
import torch
import torch.optim as optim

# import our model and dataloader
from src.utils.parser import get_args
from src.utils.utils import createfolder
from src.runs.GRUs import get_dataloader, get_model, train, test

def main():
    args = get_args()

    pd.set_option('precision', 4)
    # set seed 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # get trainloader and testloader
    trainloader, testloader = get_dataloader(args=args)

    # get the model
    model = get_model(args=args)
    # train process
    time1 = time.time()

    args.result_folder += '_'+args.model.upper()
    args.params_folder += '_'+args.model.upper()

    size = '{}X{}'.format(args.I_shape[0], args.I_shape[1])

    if args.weather_list == []:
        args.result_folder = os.path.join(args.result_folder, size, 'RAD_no_weather')
        args.params_folder = os.path.join(args.params_folder, size, 'RAD_no_weather')
    else:
        args.result_folder = os.path.join(args.result_folder, size, 'RAD_weather')
        args.params_folder = os.path.join(args.params_folder, size, 'RAD_weather')

    args.result_folder = os.path.join(args.result_folder, 'wd{:.5f}_lr{:f}'.format(args.weight_decay, args.lr))
    args.params_folder = os.path.join(args.params_folder, 'wd{:.5f}_lr{:f}'.format(args.weight_decay, args.lr))

    if args.lr_scheduler and args.optimizer != 'Adam':
        args.result_folder += '_scheduler'
        args.params_folder += '_scheduler'
    
    args.result_folder += '_'+args.optimizer
    args.params_folder += '_'+args.optimizer

    # train model
    train(model=model, trainloader=trainloader, testloader=testloader, args=args)

    time2 = time.time()
    t = time2-time1
    h = int((t)//3600)
    m = int((t-h*3600)//60)
    s = int(t-h*3600-m*60)

    print('The total computing time of this training process: {:d}:{:d}:{:d}'.format(h,m,s))

if __name__ == '__main__':
    main()