import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.ops import rotation
from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, LocalMultiHeadAttention
from pymovis.learning.embedding import SinusoidalPositionalEmbedding

class ScorePredictionNet(nn.Module):
    """
    Score prediction network given motion.
    """
    def __init__(self, d_motion, config):
        super(ScorePredictionNet, self).__init__()
        self.d_motion = d_motion
        self.config = config

        self.d_model        = config.d_model
        self.n_layers       = config.n_layers
        self.n_heads        = config.n_heads
        self.d_head         = self.d_model // self.n_heads
        self.d_ff           = config.d_ff
        self.pre_layernorm  = config.pre_layernorm
        self.dropout        = config.dropout

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.d_motion, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        
        # positional encodings
        self.embedding = SinusoidalPositionalEmbedding(self.d_model, max_len=100) # arbitrary max_len

        # Transformer layers
        self.layer_norm  = nn.LayerNorm(self.d_model)
        self.attn_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
        
        # output
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, motion):
        B, T, D = motion.shape

        x = self.encoder(motion)
        
        # add positional encodings
        pos = torch.arange(T, device=x.device)
        x = x + self.embedding(pos)

        # Transformer layers
        for attn_layer, pffn_layer in zip(self.attn_layers, self.pffn_layers):
            x = attn_layer(x, x)
            x = pffn_layer(x)
        
        # output
        if self.pre_layernorm:
            x = self.layer_norm(x)

        score = self.decoder(x)
        return score