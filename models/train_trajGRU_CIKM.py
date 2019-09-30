import os
import time
import datetime as dt

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.tools.parser import get_args
from src.tools.loss import Loss
from src.tools.utils import save_model, get_logger
from src.dataseters.CIKM import CIKMDataset, ToTensor
from src.operators.model import get_model

def train_epoch(model, dataloader, optimizer, args, logger):
    time_s = time.time()
    model.train()

    tmp_loss = 0
    total_loss = 0
    device = args.device
    dtype = args.value_dtype
    loss_function = args.loss_function

    total_idx = len(dataloader)

    for idx, data in enumerate(dataloader,0):
        src = data['inputs'].to(device=device,dtype=dtype)
        tgt = data['targets'].to(device=device,dtype=dtype)

        pred = model(src)   ## B, S, C, H, W

        optimizer.zero_grad()

        # breakpoint()
        loss = loss_function(pred, tgt)
        loss.backward()

        optimizer.step()

        tmp_loss += loss.item()/(total_idx//5)
        total_loss += loss.item()/total_idx

        if (idx+1) % (total_idx//5) == 0:
            logger.debug('[{:s}] Training Process: {:d}/{:d}, Loss = {:.2f}'.format(args.model, idx+1, total_idx, tmp_loss))
            tmp_loss = 0

    time_e =time.time()
    time_step = (time_e-time_s)/60

    logger.debug('[{:s}] Training Process: Ave_Loss = {:.2f}'.format(args.model, total_loss))
    logger.debug('[{:s}] Time spend: {:.1f} min'.format(args.model, time_step))

    return total_loss

def eval_epoch(model, dataloader, args, logger):
    time_s = time.time()
    model.eval()

    tmp_loss = 0
    total_loss = 0
    device = args.device
    dtype = args.value_dtype
    loss_function = args.loss_function

    total_idx = len(dataloader)

    with torch.no_grad():
        for idx, data in enumerate(dataloader,0):
            src = data['inputs'].to(device=device,dtype=dtype)
            tgt = data['targets'].to(device=device,dtype=dtype)
            pred = model(src)

            loss = loss_function(pred.squeeze(2), tgt)
            total_loss += loss.item()/total_idx

    logger.debug('[{:s}] Validating Process: {:d}, Loss = {:.2f}'.format(args.model, total_idx*args.batch_size, total_loss))

    time_e =time.time()
    time_step = (time_e-time_s)/60
    logger.debug('[{:s}] Time spend: {:.1f} min'.format(args.model, time_step))

    return total_loss

if __name__ == '__main__':
    args = get_args()

    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ## Get model
    model = get_model(args).to(device=args.device, dtype=args.value_dtype)

    ## Dataloaders
    # set transform tool for datasets
    transform = transforms.Compose([ToTensor()])

    # training and validating data
    trainset = CIKMDataset(train=True, test=False, args=args, transform=transform)
    valiset = CIKMDataset(train=False, test=False, args=args, transform=transform)

    # dataloader
    train_kws = {'num_workers': 4, 'pin_memory': True} if args.able_cuda else {}
    test_kws = {'num_workers': 4, 'pin_memory': True} if args.able_cuda else {}

    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, **train_kws)
    valiloader = DataLoader(dataset=valiset, batch_size=args.batch_size, shuffle=False, **test_kws)

    # optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5,0.99), weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    loss_df = pd.DataFrame([],index=pd.Index(range(args.max_epochs), name='Epoch'), columns=['Train_loss', 'Vali_loss'])

    log_file = os.path.join(args.result_folder, 'log.txt')
    loss_file = os.path.join(args.result_folder, 'loss.csv')
    logger = get_logger(log_file)

    for epoch in range(args.max_epochs):
        lr = optimizer.param_groups[0]['lr']
        logger.debug('[{:s}] Epoch {:03d}, Learning rate: {}'.format(args.model, epoch+1, lr))

        loss_df.iloc[epoch,0] = train_epoch(model, trainloader, optimizer, args, logger)
        loss_df.iloc[epoch,1] = eval_epoch(model, valiloader, args, logger)

        lr_scheduler.step()

        if (epoch+1) % 10 == 0:
            save_model(epoch, optimizer, model, args)

    loss_df.to_csv(loss_file)