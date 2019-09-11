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
from src.utils.utils import save_model
from src.dataseters.GRUs import TyDataset, ToTensor, Normalize
from src.operators.convGRU import Model
from src.utils.GRUs_hparams import CONVGRU_HYPERPARAMs

def train_epoch(model, dataloader, optimizer, args):
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
		pred = model(src)

		optimizer.zero_grad()

		loss = loss_function(pred, tgt.squeeze(2))
		loss.backward()

		optimizer.step()

		tmp_loss += loss.item()/(total_idx//3)
		total_loss += loss.item()/total_idx

		if (idx+1) % (total_idx//3) == 0:
			print('[{:s}] Training Process: {:d}/{:d}, Loss = {:.2f}'.format(args.model, idx+1, total_idx, tmp_loss))
			tmp_loss = 0

	time_e =time.time()
	time_step = (time_e-time_s)/60

	print('Training Process: Ave_Loss = {:.2f}'.format(total_loss))
	print('Time spend: {:.1f} min'.format(time_step))

	return total_loss

def eval_epoch(model, dataloader, args):
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
			pred = model(src)

			loss = loss_function(pred, tgt.squeeze(2))
			total_loss += loss.item()/total_idx

	print('[{:s}] Validating Process: {:d}, Loss = {:.2f}'.format(args.model,total_idx, total_loss))

	time_e =time.time()
	time_step = (time_e-time_s)/60
	print('Time spend: {:.1f} min'.format(time_step))

	return total_loss

if __name__ == '__main__':
	settings = parser()
	# print(settings.initial_args)
	settings.initial_args.gpu = 1
	settings.initial_args.I_size = 120
	settings.initial_args.F_size = 120
	settings.initial_args.batch_size = 32
	settings.initial_args.max_epochs = 100
	settings.initial_args.model = 'convGRU'
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

	P = CONVGRU_HYPERPARAMs(args)
	model = Model(P.n_encoders, P.n_forecasters,
                P.encoder_input_channel, P.encoder_downsample_channels, P.encoder_gru_channels,
                P.encoder_downsample_k, P.encoder_downsample_s, P.encoder_downsample_p,
                P.encoder_gru_k, P.encoder_gru_s, P.encoder_gru_p, P.encoder_n_cells,
                P.forecaster_input_channel, P.forecaster_upsample_channels, P.forecaster_gru_channels,
                P.forecaster_upsample_k, P.forecaster_upsample_s, P.forecaster_upsample_p,
                P.forecaster_gru_k, P.forecaster_gru_s, P.forecaster_gru_p, P.forecaster_n_cells,
                P.forecaster_output_channels, P.forecaster_output_k, P.forecaster_output_s, P.forecaster_output_p, P.forecaster_output_layers, batch_norm=True, target_RAD=False).to(device=args.device, dtype=args.value_dtype)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.25)

	loss_df = pd.DataFrame([],index=pd.Index(range(args.max_epochs), name='Epoch'), columns=['Train_loss', 'Vali_loss'])

	for epoch in range(args.max_epochs):
		lr = optimizer.param_groups[0]['lr']
		print('[{:s}] Epoch {:03d}, Learning rate: {}'.format(args.model, epoch+1, lr))

		loss_df.iloc[epoch,0] = train_epoch(model, trainloader, optimizer, args)
		loss_df.iloc[epoch,1] = eval_epoch(model, valiloader, args)

		if (epoch+1) > 10:
			lr_scheduler.gamma = 0.96
		lr_scheduler.step()

		if (epoch+1) % 10 == 0:
			save_model(epoch, optimizer, model, args)

	loss_df_path = os.path.join(args.result_folder, 'loss.csv')
	loss_df.csv(loss_df_path)
