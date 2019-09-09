import os
import tqdm
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
from src.utils.visulize import plot_input
from src.utils.loss import Loss
from src.dataseters.GRUs import TyDataset, ToTensor, Normalize
from src.operators.transformer import *

def train_epoch(model, dataloader, optimizer, args):
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
		
		tmp_loss += loss.item()/200
		total_loss += loss.item()/total_idx
		
		if (idx+1) % 200 == 0:
			print('{:d}/{:d}: Loss: {:.2f}'.format(idx+1, total_idx, tmp_loss))
			tmp_loss = 0
			
	return total_loss

if __name__ == '__main__':
	settings = parser()
	# print(settings.initial_args)
	settings.initial_args.gpu = 0
	settings.initial_args.I_size = 120
	settings.initial_args.F_size = 120
	settings.initial_args.batch_size = 5
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

	trainloss_df = pd.DataFrame([],index=pd.Index(range(args.max_epochs), name='Epoch'), columns=['Train_loss'])
	for epoch in range(args.max_epochs):
		if epoch >= 10:
			lr_scheduler.gamma = 0.85
		lr_scheduler.step()
		lr = optimizer.param_groups[0]['lr']
		print('Learning rate: {}'.format(lr))

		trainloss_df.iloc[epoch] = train_epoch(model, trainloader, optimizer, args)
		print('Epoch {:03d}: Loss= {:.2f}'.format(epoch, trainloss_df.iat[epoch,0]))