from CNN2D import *
import torch
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms, utils
sys.path.append("..")
from tools.dataset_CNN import ToTensor, Normalize, TyDataset


a = nn.Conv2d(1, 2, 5, stride=2, padding=0)
data = torch.randn(1,1,72,72)
print(a(data).size())
b = nn.Conv2d(2, 4, 4, stride=2, padding=0)
print(b(a(data)).size())
c = nn.Conv2d(4, 8, 4, stride=2, padding=1)
print(c(b(a(data))).size())
d = nn.Conv2d(8, 16, 4, stride=2, padding=1)
print(d(c(b(a(data)))).size())


e = nn.ConvTranspose2d(16, 16, 1, stride=1, padding=0)
print(e(d(c(b(a(data))))).size())
f = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)
print(f(e(d(c(b(a(data)))))).size())
g = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
print(g(f(e(d(c(b(a(data))))))).size())
h = nn.ConvTranspose2d(8, 4, 4, stride=2, padding=0)
print(h(g(f(e(d(c(b(a(data)))))))).size())
i = nn.ConvTranspose2d(4, 4, 6, stride=2, padding=0)
print(i(h(g(f(e(d(c(b(a(data))))))))).size())
j = nn.ConvTranspose2d(4, 2, 3, stride=1, padding=1)
print(j(i(h(g(f(e(d(c(b(a(data)))))))))).size())

data = torch.randn(4,5,72,72).to('cuda')
c = 70
encoder_input = 5
encoder_hidden = [c,8*c,12*c,16*c]
encoder_kernel = [5,4,4,4]
encoder_n_layer = 4
encoder_stride = [2,2,2,2]
encoder_padding = [0,0,1,1]

decoder_input = 16*c
decoder_hidden = [16*c,16*c,8*c,4*c,24,20]
decoder_kernel = [1,4,4,4,6,3]
decoder_n_layer = 6
decoder_stride = [1,2,2,2,2,1]
decoder_padding = [0,1,1,0,0,1]

Net = model(encoder_input, encoder_hidden, encoder_kernel, encoder_n_layer, encoder_stride, encoder_padding,
            decoder_input, decoder_hidden, decoder_kernel, decoder_n_layer, decoder_stride, decoder_padding,
            batch_norm=True).to('cuda')
print(Net(data).size())
