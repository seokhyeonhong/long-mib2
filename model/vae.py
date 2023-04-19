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

def get_mask(batch, context_frames):
    B, T, D = batch.shape

    # 0 for unknown frames, 1 for known frames
    batch_mask = torch.ones_like(batch)
    batch_mask[:, context_frames:-1, :] = 0

    # attention mask: False for unmasked frames, True for masked frames
    attn_mask = torch.zeros(1, T, T, dtype=torch.bool, device=batch.device)
    attn_mask[:, :, context_frames:-1] = True

    return batch_mask, attn_mask

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class ConvNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, pre_layernorm=False):
        super(ConvNet, self).__init__()
        self.pre_layernorm = pre_layernorm

        self.layers = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, kernel_size=7, padding=3),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        if self.pre_layernorm:
            return x + self.layers(self.layer_norm(x).transpose(1, 2)).transpose(1, 2)
        else:
            return self.layer_norm(x + self.layers(x.transpose(1, 2)).transpose(1, 2))

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
        self.mu_token = nn.Parameter(torch.zeros(1, 1, self.d_model), requires_grad=False)
        self.logvar_token = nn.Parameter(torch.zeros(1, 1, self.d_model), requires_grad=False)

        # encoders
        self.motion_encoder = nn.Sequential(
            nn.Conv1d(self.d_motion + 5, self.d_model, kernel_size=7, padding=3), # (motion, traj)
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=7, padding=3),
        )

        # Transformer layers
        self.atten_layers = nn.ModuleList()
        self.conv_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.conv_layers.append(ConvNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

    def forward(self, x, traj):
        B, T, D = x.shape

        # motion encoder
        x = torch.cat([x, traj], dim=-1)
        x = self.motion_encoder(x.transpose(1, 2)).transpose(1, 2)
        mu = self.mu_token.repeat(B, 1, 1)
        logvar = self.logvar_token.repeat(B, 1, 1)
        x = torch.cat([mu, logvar, x], dim=1)
        T += 2

        # Transformer layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x)
            x = self.conv_layers[i](x)

        # split mu and logvar
        mu, logvar = x[:, 0], x[:, 1]

        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, d_motion, config):
        super(Decoder, self).__init__()
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
            nn.Conv1d(self.d_motion * 2 + 5, self.d_model, kernel_size=7, padding=3), # (motion, mask) + traj(=5)
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=7, padding=3),
        )

        # Transformer layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.atten_layers = nn.ModuleList()
        self.conv_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.conv_layers.append(ConvNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion),
        )
    
    def forward(self, motion, traj, z):
        B, T, D = motion.shape

        # original motion
        original_motion = motion.clone()

        # mask out
        batch_mask, atten_mask = get_mask(motion, self.config.context_frames)

        # encoders
        x = torch.cat([motion * batch_mask, batch_mask, traj], dim=-1)
        x = self.motion_encoder(x.transpose(1, 2)).transpose(1, 2)

        # add latent vector
        x = x + z

        # Transformer layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, mask=atten_mask)
            x = self.conv_layers[i](x)
            atten_mask = (1 - F.max_pool1d(1 - atten_mask.float(), kernel_size=15, stride=1, padding=7)).bool()

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # decoder
        x = self.decoder(x)

        # unmask original motion
        x = x * (1 - batch_mask) + original_motion * batch_mask

        return x

class VAE(nn.Module):
    def __init__(self, d_motion, config):
        super(VAE, self).__init__()

        self.d_motion   = d_motion
        self.config     = config

        self.encoder = Encoder(d_motion, config)
        self.decoder = Decoder(d_motion, config)
    
    def forward(self, motion, traj):
        B, T, D = motion.shape

        mu, logvar = self.encoder.forward(motion, traj)
        z = reparameterize(mu.unsqueeze(1).repeat(1, T, 1), logvar.unsqueeze(1).repeat(1, T, 1))

        recon = self.decoder.forward(motion, traj, z)
        
        return recon, mu, logvar
    
    def sample(self, motion, traj):
        B, T, D = motion.shape
        z = torch.randn(B, T, self.config.d_model, dtype=motion.dtype, device=motion.device)
        pred_motion = self.decoder.forward(motion, traj, z)
        return pred_motion