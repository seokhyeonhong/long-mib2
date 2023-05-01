import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.ops import rotation
from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, LocalMultiHeadAttention
from pymovis.learning.embedding import SinusoidalPositionalEmbedding

class MotionRefineNet(nn.Module):
    """
    Score prediction network given motion.
    """
    def __init__(self, d_motion, config):
        super(MotionRefineNet, self).__init__()
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

        # Encoder
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.d_motion + 1, self.d_model), # (motion, mask(=1))
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        self.traj_encoder = nn.Sequential(
            nn.Linear(5, self.d_model), # (traj(=5))
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        
        # Positional encodings
        self.embedding = SinusoidalPositionalEmbedding(self.d_model, max_len=100) # arbitrary max_len

        # Transformer layers
        self.layer_norm   = nn.LayerNorm(self.d_model)
        self.attn_layers  = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.cross_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_motion + 4), # (motion, contact(=1))
        )

    def forward(self, motion, mask, traj):
        B, T, D = motion.shape

        # original motion
        original_motion = motion.clone()

        # encoder
        x = self.motion_encoder(torch.cat([motion, mask], dim=-1))
        context = self.traj_encoder(traj)

        # add positional encodings
        emb = self.embedding(torch.arange(T, device=x.device))
        x = x + emb
        context = context + emb

        # Transformer layers
        for attn_layer, cross_layer, pffn_layer in zip(self.attn_layers, self.cross_layers, self.pffn_layers):
            x = attn_layer(x, x)
            x = cross_layer(x, context)
            x = pffn_layer(x)
        
        # output
        if self.pre_layernorm:
            x = self.layer_norm(x)

        x = self.decoder(x)

        pred_motion, pred_contact = torch.split(x, [self.d_motion, 4], dim=-1)
        pred_motion = original_motion * mask + pred_motion * (1 - mask)
        pred_contact = torch.sigmoid(pred_contact)
        return pred_motion, pred_contact