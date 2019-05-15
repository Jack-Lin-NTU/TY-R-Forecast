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
from src.runs.mymodel import get_dataloader, get_model, train, test

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