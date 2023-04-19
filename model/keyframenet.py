import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import TransformerBlock
from model.embedding import PeriodicPositionalEmbedding

class KeyframeNet(nn.Module):
    def __init__(self, d_motion, config):
        super(KeyframeNet, self).__init__()
        self.d_motion       = d_motion
        self.config         = config
    
        self.d_model        = config.d_model
        self.n_layers       = config.n_layers
        self.n_heads        = config.n_heads
        self.d_head         = self.d_model // self.n_heads
        self.d_ff           = config.d_ff
        self.pre_layernorm  = config.pre_layernorm
        self.dropout        = config.dropout

        self.fps            = config.fps
        self.context_frames = config.context_frames

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")
        
        # encoders
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.d_motion * 2 + 5, self.d_model), # (motion, mask, traj)
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
        )

        # positional embedding
        self.embedding = PeriodicPositionalEmbedding(self.d_model, self.fps)

        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.transformer_layers.append(TransformerBlock(self.d_motion, self.config))
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion),
        )

    def forward(self, x, traj, mask):
        B, T, D = x.shape

        # motion encoder
        x = torch.cat([x, traj, mask], dim=-1)
        x = self.motion_encoder(x)

        # additive positional embedding
        pos = torch.arange(0, T, step=1, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(x.device)
        x = x + self.embedding.forward(pos)

        # Transformer layers
        for i in range(self.n_layers):
            x = self.transformer_layers[i].forward(x, x)
        
        # decoder
        x = self.decoder(x)

        return x