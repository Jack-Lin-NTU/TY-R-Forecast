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
    def __init__(self, encoder, decoder, src_net, trg_net, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_net = src_net
        self.trg_net = trg_net
        self.generator = generator
    
    def forward(self, src, trg, src_mask, trg_mask):
        '''
        Take in and process masked source and target sequences.
        '''
        return self.generator(self.decode(self.encode(src, src_mask), src_mask, trg, trg_mask))
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_net(src), src_mask)
    
    def decode(self, memory, src_mask, trg, trg_mask):
        return self.decoder(self.trg_net(trg), memory, src_mask, trg_mask)

## Generator
class Generator(nn.Module):
    def __init__(self, d_channel, out_channel)
        super(Generator, self).__init__()
        self.proj = nn.Conv2d(d_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(sefl, x):
        return self.proj(x)

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
        
    def forward(self, x, memory, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, trg_mask)
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
    
    def forward(self, x, memory, src_mask, trg_mask):
        ''' Follow Figure 1 (right) for connections. '''
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, trg_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


## Attention function
def attention(query, key, value, mask=None, dropout=None):
    # (H x W)
    d_k = query.shape[-1]
    # n_batch x T x T
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask==0, 1e-9)
    p_attn = F.softmax(score, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_channel, dropout=0.1):
        ''' Take in model size and number of heads. '''
        super(MultiHeadedAttention, self).__init__()
        # We assume d_v always equals d_k
        self.h = h
        self.convs = clones(nn.Conv2d(d_channel, 1, kernel_size=3, stride=1, padding=1), N=3)
        self.output_conv = nn.Conv2d(1, d_channel, kernel_size=3, stride=1, padding=1)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        n_batch, T, C, H, W = query.shape
        
        assert (H*W) % self.h == 0
        self.d_k = (H*W) // self.h

        # Shape of query, key, and value: n_batch x T x C x H x W -> (n_batch*T) x C x H x W
        query, key, value = query.view(n_batch*T, C, H, W), key.view(n_batch*T, C, H, W), value.view(n_batch*T, C, H, W)
        # 1) Do all the CNN projections in batch from (n_batch*T) x C x H x W  => n_batch x T x h x d_k
        query, key, value = [cnn(x).squeeze(1).reshape(n_batch, T, self.h, self.d_k).transpose(1, 2) 
                            for cnn, x in zip(self.convs, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).reshape(n_batch*T, H, W).unsqueeze(1)
        return self.output_conv(x).reshape(n_batch, T, -1, H, W)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_channel, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv2d(d_channel, d_ff, kernel_size=1, stride=1, padding=0)
        self.w_2 = nn.Conv2d(d_ff, d_channel, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionEncodeing(nn.Module):
    def __init__(self, H, W, dropout, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        # shape: max_len x (H*W)
        pe = torch.zeros(max_len, H*W)
        # shape: max_len x 1
        position = torch.arange(0, max_len).unsqueeze(1)
        # shape: (H*W)/2
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape: 1 x max_len x (H*W)
        pe = pe.reshape(-1,H,W).unsqueeze(0).unsqueeze(2)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)

def make_model(src_vocab, trg_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    ''' Helper: Construct a model from hyperparameters. '''
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, trg_vocab), c(position)),
            Generator(d_model, trg_vocab)
            )
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model