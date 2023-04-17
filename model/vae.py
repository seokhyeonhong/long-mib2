import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.ops import rotation, motionops
from pymovis.learning.mlp import MultiLinear
from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, LocalMultiHeadAttention

def get_keyframe_relative_position(window_length, context_frames):
    position = torch.arange(window_length, dtype=torch.float32)
    dist_ctx = position - (context_frames - 1) # distance to the last context frame
    dist_tgt = position - (window_length - 1)  # distance to the target frame

    p_kf = torch.stack([dist_ctx, dist_tgt], dim=-1) # (T, 2)

    return p_kf

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std
    
class Encoder(nn.Module):
    def __init__(self, d_motion, config, is_context=True):
        super(Encoder, self).__init__()
        self.d_motion       = d_motion
        self.config         = config
        self.is_context     = is_context
    
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
        
        # VAE token
        self.mu_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.logvar_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # encoders
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.d_motion * 2 + 5, self.d_model), # (motion, mask) + traj(=5)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )

        # relative positional encoder
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_head),
            nn.Dropout(self.dropout),
        )

        # Transformer layers
        self.atten_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, 15, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

    def forward(self, x, mask, traj):
        B, T, D = x.shape

        # motion encoder
        x = self.motion_encoder(torch.cat([x, mask, traj], dim=-1))
        mu = self.mu_token.repeat(B, 1, 1)
        logvar = self.logvar_token.repeat(B, 1, 1)
        x = torch.cat([mu, logvar, x], dim=1)
        T += 2

        # relative positional encoding
        pad_len = T-self.fps//2-1
        rel_pos = torch.arange(-self.config.fps//2, self.config.fps//2+1, device=x.device).unsqueeze(-1).float()
        lookup_table = self.relative_pos_encoder(rel_pos)
        lookup_table = F.pad(lookup_table, (0, 0, pad_len, pad_len), value=0)

        # Transformer layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, lookup_table=lookup_table)
            x = self.pffn_layers[i](x)

        # split mu and logvar
        mu, logvar = x[:, 0], x[:, 1]

        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, d_motion, config, is_context=True):
        super(Decoder, self).__init__()
        self.d_motion       = d_motion
        self.config         = config
        self.is_context     = is_context
    
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
            nn.Linear(self.d_motion * 2 + 5, self.d_model), # (motion, mask) + traj(=5)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )

        # relative positional encoder
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_head),
            nn.Dropout(self.dropout),
        )

        # Transformer layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.atten_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, 15, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.cross_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, 15, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion if self.is_context else self.d_motion + 4),
        )
    
    def forward(self, motion, traj, z, mask):
        B, T, D = motion.shape

        # original motion
        original_motion = motion.clone()

        # mask out
        if self.is_context:
            motion = motion * mask

        # encoders
        x = self.motion_encoder(torch.cat([motion, mask, traj], dim=-1))

        # relative positional encoding
        pad_len = T - (self.fps//2) - 1
        rel_pos = torch.arange(-self.config.fps//2, self.config.fps//2+1, device=x.device).unsqueeze(-1).float()
        lookup_table = self.relative_pos_encoder(rel_pos)
        lookup_table = F.pad(lookup_table, (0, 0, pad_len, pad_len), value=0)

        # Transformer layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, lookup_table=lookup_table)
            x = self.cross_layers[i](x, z, lookup_table=lookup_table)
            x = self.pffn_layers[i](x)

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # decoder
        x = self.decoder(x)

        # unmask original motion
        x[..., :D] = x[..., :D] * (1 - mask) + original_motion * mask

        # contact
        if not self.is_context:
            x[..., -4:] = torch.sigmoid(x[..., -4:])

        return x

class VAE(nn.Module):
    def __init__(self, d_motion, config, is_context):
        super(VAE, self).__init__()

        self.d_motion   = d_motion
        self.config     = config
        self.is_context = is_context

        self.encoder = Encoder(d_motion, config, is_context=is_context)
        self.decoder = Decoder(d_motion, config, is_context=is_context)
    
    def forward(self, motion, traj, mask):
        B, T, D = motion.shape

        mu, logvar = self.encoder.forward(motion, mask, traj)
        z = reparameterize(mu.unsqueeze(1).repeat(1, T, 1), logvar.unsqueeze(1).repeat(1, T, 1))

        recon = self.decoder.forward(motion, traj, z, mask=mask)
        
        return recon, mu, logvar
    
    def sample(self, motion, traj, mask):
        B, T, D = motion.shape
        z = torch.randn(B, T, self.config.d_model, dtype=motion.dtype, device=motion.device)
        pred_motion = self.decoder.forward(motion, traj, z, mask)
        return pred_motion