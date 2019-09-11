import os
import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils.easyparser import *
from src.utils.loss import Loss
from src.utils.utils import save_model, get_logger
from src.dataseters.GRUs import TyDataset, ToTensor, Normalize
from src.operators.transformer import *

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
		tgt = data['targets'].to(device=device,dtype=dtype).unsqueeze(2)
		src_mask = torch.ones(1, src.shape[1]).to(device=device,dtype=dtype)
		tgt_mask = subsequent_mask(tgt.shape[1]).to(device=device,dtype=dtype)
		pred = model(src, tgt, src_mask, tgt_mask)
		
		optimizer.zero_grad()
		
		loss = loss_function(pred, tgt.squeeze(2))
		loss.backward()
		
		optimizer.step()
		
		tmp_loss += loss.item()/(total_idx//3)
		total_loss += loss.item()/total_idx
		
		if (idx+1) % (total_idx//3) == 0:
			logger.debug('Training Process: {:d}/{:d}, Loss = {:.2f}'.format(idx+1, total_idx, tmp_loss))
			tmp_loss = 0
			
	time_e =time.time()
	time_step = (time_e-time_s)/60

	logger.debug('Training Process: Ave_Loss = {:.2f}'.format(total_loss))
	logger.debug('Time spend: {:.1f} min'.format(time_step))
	
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
			tgt = data['targets'].to(device=device,dtype=dtype).unsqueeze(2)
			src_mask = torch.ones(1, src.shape[1]).to(device=device,dtype=dtype)
			tgt_mask = subsequent_mask(tgt.shape[1]).to(device=device,dtype=dtype)
			pred = model(src, tgt, src_mask, tgt_mask)
			
			loss = loss_function(pred, tgt.squeeze(2))
			total_loss += loss.item()/total_idx
		
	print('Validating Process: {:d} samples, Loss = {:.2f}'.format(total_idx*args.batch_size, total_loss))
			
	time_e =time.time()
	time_step = (time_e-time_s)/60
	print('Time spend: {:.1f} min'.format(time_step))

	return total_loss

if __name__ == '__main__':
	settings = parser()
	# print(settings.initial_args)
	settings.initial_args.gpu = 0
	settings.initial_args.I_size = 120
	settings.initial_args.F_size = 120
	settings.initial_args.batch_size = 5
	settings.initial_args.max_epochs = 30
	args = settings.get_args()

	torch.cuda.set_device(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)

	## dataloader
	# set transform tool for datasets
	if args.normalize_input:
		transform = transforms.Compose([ToTensor(), Normalize(args)])
	else:
		transform = transforms.Compose([ToTensor()])

	# training and validating data
	trainset = TyDataset(args, train=True, transform=transform)
	valiset = TyDataset(args, train=False, transform=transform)

	# dataloader
	train_kws = {'num_workers': 4, 'pin_memory': True} if args.able_cuda else {}
	test_kws = {'num_workers': 4, 'pin_memory': True} if args.able_cuda else {}

	trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, **train_kws)
	valiloader = DataLoader(dataset=valiset, batch_size=args.batch_size, shuffle=False, **test_kws)

	model = make_model(H=args.I_size, W=args.I_size, input_channel=1, d_channel=5, d_channel_ff=10) \
						.to(device=args.device, dtype=args.value_dtype)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.25)

	loss_df = pd.DataFrame([],index=pd.Index(range(args.max_epochs), name='Epoch'), columns=['Train_loss', 'Vali_loss'])
	
	log_file = os.path.join(args.result_folder, 'log.txt')
	loss_file = os.path.join(args.result_folder, 'loss.csv')
	logger = get_logger(log_file)

	for epoch in range(args.max_epochs):
		lr = optimizer.param_groups[0]['lr']
		logger.debug('Epoch {:03d}, Learning rate: {}'.format(epoch+1, lr))
 
		loss_df.iloc[epoch,0] = train_epoch(model, trainloader, optimizer, args, logger)
		loss_df.iloc[epoch,1] = eval_epoch(model, valiloader, args, logger)

		if (epoch+1) > 10:
			lr_scheduler.gamma = 0.95
		lr_scheduler.step()

		if (epoch+1) % 10 == 0:
			save_model(epoch, optimizer, model, args)

	loss_df.to_csv(loss_df_path)