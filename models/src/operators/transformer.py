import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The function for model clone
def clones(module, N):
    ''' A function to produce N identical layers. '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

## Model architecture
class EncoderDecoder(nn.Module):
    ''' A standard Encoder-Decoder archtecture. Base for this model and other models. '''
    def __init__(self, encoder, decoder, src_net, tgt_net, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_net = src_net
        self.tgt_net = tgt_net
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        '''
        Take in and process masked source and target sequences.
        '''
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_net(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_net(tgt), memory, src_mask, tgt_mask)

## Generator
class Generator(nn.Module):
    


## Encoder and Decoder
class SublayerConnection(nn.Module):
    ''' A residual connection followed by a layer norm. Note for code implicitiy the norm is first as opposed to last. '''
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        ''' Apply residual connection to any sublayer with the same size. '''
        return x + self.dropout(sublayer(self.norm(x)))

class Encoder(nn.Module):
    ''' Core encoder: a stack of N layers. '''
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, eps=1e-6)
        
    def forward(self, x, mask):
        ''' Pass the input (and mask) through each layer in turn. '''
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    ''' EncoderLayer is made up of self-attention and feed forward. '''
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        
    def forward(self, x, mask):
        ''' Follow the flows of connections. '''
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    ''' Core decoder: A stack of N layers with masking. '''
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    ''' DecoderLayer is made of self-attn, src-attn, and feed forward. '''
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        ''' Follow Figure 1 (right) for connections. '''
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


## Attention function
def attention(query, key, value, mask=None, dropout=None):
    query = query.flatten()
    key = key.flatten()
    value = value.flatten()
    d_k = query.shape[-1]
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask==0, 1e-9)
    p_attn = F.softmax(score, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn