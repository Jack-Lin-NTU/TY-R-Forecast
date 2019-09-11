import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .src.operators.convGRU import *

class model(nn.Module):
    def __init__(encoder, attention, decoder, generator, input_frames, pred_frames):
        super().__init__()
        self.input_frames = input_frames
        self.pred_frames = pred_frames
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder
        self.generator = generator

    def forward(self, x):
        hiddens = []
        for i in range(self.input_frames):
            if i == 0:
                hidden = None
            hidden = self.encoder(x, hidden)
            hiddens.append(hidden)
        
        hiddens = torch.cat(hiddens, dim=1)

        for i in range(self.pred_frames):
            if i == 0:
                hidden = hiddens[-1]
            hidden = self.attention(hiddens, hidden)
            hidden, output = self.decoder(hidden)

