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

## Flownet
class FlowNet(nn.Module):
    ''' A cnn layer to predict flow map of input image'''
    def __init__(self, in_channel, out_channel, hid_channel=20, k=3, s=1, p=1):
        super(FlowNet,self).__init__()
        self.flownet = nn.Sequential(
            nn.Conv2d(in_channel, hid_channel, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(hid_channel),
            nn.Conv2d(hid_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        output = []
        for i in range(x.shape[1]):
            output.append(self.flownet(x[:,i,:,:,:]))
        return torch.stack(output, dim=1)

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
        ''' Take in and process masked source and target images. '''
        return self.generator(self.decode(self.encode(src, src_mask), src_mask, trg, trg_mask))
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_net(src), src_mask)
    
    def decode(self, memory, src_mask, trg, trg_mask):
        return self.decoder(self.trg_net(trg), memory, src_mask, trg_mask)

## Generator
class Generator(nn.Module):
    def __init__(self, d_channel, out_channel):
        super(Generator, self).__init__()
        self.proj = nn.Conv2d(d_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = []
        for i in range(x.shape[1]):
            output.append(self.proj(x[:,i])) 
        return torch.cat(output, dim=1)

## Encoder and Decoder
class SublayerConnection(nn.Module):
    ''' A residual connection followed by a layer norm. Note for code implicitiy the norm is first as opposed to last. '''
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

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

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    
## Attention function
def attention(query, key, value, mask=None, dropout=None):
    # query and key shape: n_batch x h x T x (H1*W1)
    # value shape: n_batch x T x (C*H*W)
    n_batch, h, T1, d_k = query.shape
    n_batch, h, T2, d_k = key.shape
    # n_batch x T x T
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        score = score.masked_fill(mask==0, 1e-9)
    p_attn = F.softmax(score, dim=-1)
    # p_attn shape: n_batch x h x T x T
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.transpose(1,0), value).transpose(1,0), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_channel, dropout=0.1):
        ''' Take in model size and number of heads. '''
        super(MultiHeadedAttention, self).__init__()
        # We assume d_v always equals d_k
        self.h = h
        self.qcnn = nn.Conv2d(d_channel, h, kernel_size=7, stride=3, padding=1)
        self.kcnn = nn.Conv2d(d_channel, h, kernel_size=7, stride=3, padding=1)
        self.vcnn = nn.Conv2d(d_channel, d_channel, kernel_size=1, stride=1, padding=0)
        self.outcnn = nn.Conv2d(h*d_channel, d_channel, kernel_size=1, stride=1, padding=0)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        # original image size
        n_batch, T1, C, H, W = query.shape
        n_batch, T2, C, H, W = key.shape
        # Shape of query, key, and value: n_batch x T x C x H x W -> (n_batch*T) x C x H x W
        query, key, value = query.view(n_batch*T1, C, H, W), key.view(n_batch*T2, C, H, W), value.view(n_batch*T2, C, H, W)
        # 1) Do all the CNN projections in batch 
        # query and key: (n_batch*T) x C x H x W  => (n_batch*T) x h x H' x W'
        # value: (n_batch*T) x C x H x W  => (n_batch*T) x C x H x W 
        query, key, value = self.qcnn(query), self.kcnn(key), self.vcnn(value)
        _, h, H1, W1 = query.shape
        # query and key shape: n_batch x h x T x (H1*W1)
        # value shape: n_batch x T x (C*H*W)
        query, key, value = query.reshape(n_batch, T1, h, H1*W1).transpose(1,2), key.reshape(n_batch, T2, h, H1*W1).transpose(1,2), value.reshape(n_batch, T2, C*H*W)

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x shape: n_batch x h x T x (C*H*W)

        # 3) Apply a final cnn. 
        x = x.transpose(1, 2).reshape(n_batch*T1, h*C, H, W)
        return self.outcnn(x).reshape(n_batch, T1, C, H, W)

class PositionwiseCNN(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_channel, d_ff, dropout=0.1, groups=6):
        super(PositionwiseCNN, self).__init__()
        self.w_1 = nn.Conv2d(d_channel*groups, d_ff*groups, kernel_size=3, stride=1, padding=1, groups=groups)
        self.w_2 = nn.Conv2d(d_ff, d_channel*groups, kernel_size=3, stride=1, padding=1, groups=groups)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
#        output = []
#        for i in range(x.shape[1]):
#            output.append(self.w_2(self.dropout(F.relu(self.w_1(x[:,i,:,:,:])))).unsqueeze(1))
#        return torch.cat(output, dim=1)
        B, T, C, H, W = x.shape
        return self.w_2(self.dropout(F.relu(self.w_1(x.reshape(B,T*C,H,W))))).reshape(B,T,C,H,W)
        
class PositionEncodeing(nn.Module):
    def __init__(self, H, W, dropout=0.1, max_len=30):
        super(PositionEncodeing, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        # shape: max_len x (H*W)
        pe = torch.zeros(max_len, H*W)
        # shape: max_len x 1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # shape: (H*W)/2
        div_term = torch.exp(torch.arange(0, H*W, 2, dtype=torch.float) * -(math.log(10000.0) / (H*W)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape: 1 x max_len x (H*W)
        pe = pe.reshape(-1,H,W).unsqueeze(0).unsqueeze(2)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)

def make_model(H, W, input_channel=1, d_channel=1, d_channel_ff=3, N=6, h=8, dropout=0.1):
    ''' Helper: Construct a model from hyperparameters. '''
    c = copy.deepcopy
    # attention layer
    attn = MultiHeadedAttention(h, d_channel)
    # CNN feedforward layer
    encnnff = PositionwiseCNN(d_channel, d_channel_ff, dropout, groups=6)
    decnnff = PositionwiseCNN(d_channel, d_channel_ff, dropout, groups=18)
    # position encoding layer
    position = PositionEncodeing(H, W, dropout)
    model = EncoderDecoder(
            Encoder(layer=EncoderLayer([H,W], c(attn), c(encnnff), dropout), N=N),
            Decoder(layer=DecoderLayer([H,W], c(attn), c(attn), c(decnnff), dropout), N=N),
            src_net=nn.Sequential(FlowNet(input_channel, d_channel, k=3, s=1, p=1), c(position)),
            trg_net=nn.Sequential(FlowNet(input_channel, d_channel, k=3, s=1, p=1), c(position)),
            generator=Generator(d_channel, 1)
            )
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    return model
