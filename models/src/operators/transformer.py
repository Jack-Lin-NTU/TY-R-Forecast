import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The function for model clone
def clones(module, N):
    '''
    A function to produce N identical layers.
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

## Model architecture
class EncoderDecoder(nn.Module):
    '''
    A standard Encoder-Decoder archtecture. Base for this model and other models.
    '''
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        '''
        Take in and process masked source and target sequences.
        '''
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)