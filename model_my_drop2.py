import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import torch.nn as nn
import copy
import math


import torch
import torch.nn as nn
import torch.nn.functional as F


import copy
import math
import numpy as np




class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, src_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed


    def forward(self, x41,x71,x101, src_mask):
        "Take in and process masked src and target sequences."

        return self.encoder(self.src_embed(x41),self.src_embed(x71),self.src_embed(x101), src_mask)








def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x,y,z, mask):   
        "Pass the input (and mask) through each layer in turn."

        for layer in self.layers:
            x= layer(x,y,z, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2





class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))





class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn

        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x,y,z, mask):   
        "Follow Figure 1 (left) for connections."

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)+self.self_attn(x, y, y, mask)+self.self_attn(x, z, z, mask))   

        return self.sublayer[1](x, self.feed_forward)










def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)  
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:

        scores = scores.masked_fill(mask == 0, float('-inf'))


    p_attn = F.softmax(scores, dim=-1)


    if dropout is not None:
        p_attn = dropout(p_attn)



    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):    
        "Take in model size and number of heads."
        
        super(MultiHeadedAttention, self).__init__()
        
        assert d_model % h == 0
        
        self.d_k = d_model // h
        
        self.h = h
        
        self.linears = clones(nn.Linear(d_model, d_model), 4)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        "Implements Figure 2"
        if mask is not None:

            mask = mask.unsqueeze(1)

        nbatches = query.size(0)


        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]


        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)


        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)  
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))





class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)


        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def make_model(self,  N_enc=6, N_dec=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)   
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),


            nn.Sequential(c(position)) )



        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self):
        super(TransformerModel, self).__init__()


        self.model = self.make_model(
                                     N_enc=3,
                                     N_dec=3,
                                     d_model=64,
                                     d_ff=256,
                                     h=8,
                                     dropout=0.2)
        self.linear_41 = nn.Linear(100,64)
        self.linear_101 = nn.Linear(100, 64)
        self.linear_71 = nn.Linear(100, 64)
        self.linear1 = nn.Linear(64,1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 100))
    def forward(self, x_101,x_71,x_41, masks=None):




        masks =None
        m = self.model(self.linear_41(x_41),self.linear_71(x_71),self.linear_101(x_101),masks) 

        x = self.linear1(m.mean(1))
        return x


