import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.ops import rotation, motionops
from pymovis.learning.mlp import MultiLinear
from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, LocalMultiHeadAttention

def get_mask(batch, context_frames):
    B, T, D = batch.shape

    # 0 for unknown frames, 1 for known frames
    batch_mask = torch.ones(B, T, 1, dtype=batch.dtype, device=batch.device)
    batch_mask[:, context_frames:-1, :] = 0
    return batch_mask

def get_keyframe_relative_position(window_length, context_frames):
    position = torch.arange(window_length, dtype=torch.float32)
    dist_ctx = position - (context_frames - 1) # distance to the last context frame
    dist_tgt = position - (window_length - 1)  # distance to the target frame

    p_kf = torch.stack([dist_ctx, dist_tgt], dim=-1) # (T, 2)

    return p_kf

class ContextGenerator(nn.Module):
    def __init__(self, d_motion, config):
        super(ContextGenerator, self).__init__()
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
            nn.Linear(self.d_motion + 1, self.d_model), # (motion + mask)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        self.traj_encoder = nn.Sequential(
            nn.Linear(3, self.d_model),
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
        self.relative_pos = nn.Parameter(torch.arange(-self.config.fps//2, self.config.fps//2+1).unsqueeze(-1).float(), requires_grad=False)

        # Transformer layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.atten_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, self.fps+1, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.cross_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, self.fps+1, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion),
        )
    
    def forward(self, motion, traj):
        B, T, D = motion.shape

        # original motion
        original_motion = motion.clone()

        # fill in missing frames with z
        mask   = get_mask(motion, self.config.context_frames)
        z      = torch.randn_like(motion)
        motion = mask * motion + (1-mask) * z

        # encoder
        x = self.motion_encoder(torch.cat([motion, mask], dim=-1))
        context = self.traj_encoder(traj)

        # relative positional encoding
        pad_len = T-self.fps//2-1
        rel_pos = self.relative_pos_encoder(self.relative_pos)
        rel_pos = F.pad(rel_pos, (0, 0, pad_len, pad_len), value=0)

        # Transformer layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, lookup_table=rel_pos)
            x = self.cross_layers[i](x, context, lookup_table=rel_pos)
            x = self.pffn_layers[i](x)

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # decoder
        x = self.decoder(x)

        # unmask original motion
        x = x * (1 - mask) + original_motion * mask

        return x, mask

class DetailGenerator(nn.Module):
    def __init__(self, d_motion, config):
        super(DetailGenerator, self).__init__()
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
            nn.Linear(self.d_motion + 1, self.d_model), # (motion + mask)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        self.traj_encoder = nn.Sequential(
            nn.Linear(3, self.d_model),
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
        self.relative_pos = nn.Parameter(torch.arange(-self.config.fps//2, self.config.fps//2+1).unsqueeze(-1).float(), requires_grad=False)

        # Transformer layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.atten_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, self.fps+1, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.cross_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, self.fps+1, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion),
        )
    
    def forward(self, motion, traj, mask):
        B, T, D = motion.shape

        # original motion
        original_motion = motion.clone()

        # encoder
        x = self.motion_encoder(torch.cat([motion, mask], dim=-1))
        context = self.traj_encoder(traj)

        # relative positional encoding
        pad_len = T-self.fps//2-1
        rel_pos = self.relative_pos_encoder(self.relative_pos)
        rel_pos = F.pad(rel_pos, (0, 0, pad_len, pad_len), value=0)

        # Transformer layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, lookup_table=rel_pos)
            x = self.cross_layers[i](x, context, lookup_table=rel_pos)
            x = self.pffn_layers[i](x)

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # decoder
        x = self.decoder(x)

        # unmask original motion
        x = x * (1 - mask) + original_motion * mask

        return x

class Discriminator(nn.Module):
    def __init__(self, d_motion, config):
        super(Discriminator, self).__init__()
        
        self.d_motion = d_motion
        self.config   = config
        
        self.d_model        = config.d_model
        self.n_layers       = config.n_layers
        self.n_heads        = config.n_heads
        self.d_head         = self.d_model // self.n_heads
        self.d_ff           = config.d_ff
        self.pre_layernorm  = config.pre_layernorm
        self.dropout        = config.dropout

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")
        
        # discriminators - 1D convolution
        self.short_conv = nn.Sequential(
            nn.Conv1d(self.d_motion, self.d_model, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv1d(self.d_model, 1, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )
        self.long_conv = nn.Sequential(
            nn.Conv1d(self.d_motion, self.d_model, kernel_size=15, stride=1, padding=7),
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=15, stride=1, padding=7),
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=15, stride=1, padding=7),
            nn.PReLU(),
            nn.Conv1d(self.d_model, 1, kernel_size=15, stride=1, padding=7),
            nn.Sigmoid()
        )

    def forward(self, x):
        short_scores = self.short_conv(x.transpose(1, 2))
        long_scores  = self.long_conv(x.transpose(1, 2))

        return short_scores, long_scores

class ContextGAN(nn.Module):
    def __init__(self, d_motion, config):
        super(ContextGAN, self).__init__()
        
        self.generator     = ContextGenerator(d_motion, config)
        self.discriminator = Discriminator(d_motion, config)

    def generate(self, motion, traj):
        motion, mask = self.generator.forward(motion, traj)
        return motion, mask
    
    def discriminate(self, motion):
        short, long = self.discriminator.forward(motion)
        return short, long

class DetailGAN(nn.Module):
    def __init__(self, d_motion, config):
        super(DetailGAN, self).__init__()
        
        self.generator     = DetailGenerator(d_motion, config)
        self.discriminator = Discriminator(d_motion, config)
    
    def generate(self, motion, traj, mask):
        motion, mask = self.generator.forward(motion, traj, mask)
        return motion, mask
    
    def discriminate(self, motion):
        short, long = self.discriminator.forward(motion)
        return short, long