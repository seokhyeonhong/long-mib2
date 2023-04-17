import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.ops import rotation, motionops
from pymovis.learning.mlp import MultiLinear
from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, LocalMultiHeadAttention

def get_mask(batch, context_frames, p_unmask=0.0):
    # 0 for unknown frames, 1 for known frames
    batch_mask = torch.ones_like(batch)
    batch_mask[:, context_frames:-1, :] = 0

    # probability of unmasking
    samples = torch.rand_like(batch_mask)
    batch_mask[samples < p_unmask] = 1

    return batch_mask

def get_keyframe_relative_position(window_length, context_frames):
    position = torch.arange(window_length, dtype=torch.float32)
    dist_ctx = position - (context_frames - 1) # distance to the last context frame
    dist_tgt = position - (window_length - 1)  # distance to the target frame

    p_kf = torch.stack([dist_ctx, dist_tgt], dim=-1) # (T, 2)

    return p_kf
    
class Generator(nn.Module):
    def __init__(self, d_motion, config, is_context=True):
        super(Generator, self).__init__()
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
            nn.Conv1d(self.d_motion * 3 + 5, self.d_model, kernel_size=5, padding=2), # (motion, noise, mask) + (trajectory)
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=5, padding=2),
        )

        # relative positional encoder
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_head),
        )
        self.relative_pos = nn.Parameter(torch.arange(-self.config.fps//2, self.config.fps//2+1).unsqueeze(-1).float(), requires_grad=False)

        # Transformer layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.atten_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, 15, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_motion if self.is_context else self.d_motion + 4, kernel_size=5, padding=2),
        )
    
    def forward(self, motion, traj, mask=None, p_unmask=0.0):
        B, T, D = motion.shape

        # original motion
        original_motion = motion.clone()

        # random noise z
        z = torch.randn_like(motion)

        # scheduled noise
        t = torch.arange(T, device=motion.device).unsqueeze(-1)
        t = torch.min(torch.abs(t - self.config.context_frames), torch.abs((T-1) - t)) / self.fps
        t = torch.clip(t, 0, 1)
        z = z * t

        # mask
        if mask is None:
            mask = get_mask(motion, self.config.context_frames, p_unmask=p_unmask)
        
        # x
        x = torch.cat([motion*mask, mask, z, traj], dim=-1)
        x = self.motion_encoder(x.transpose(1, 2)).transpose(1, 2)

        # relative positional encoding
        pad_len = T - (self.fps//2) - 1
        rel_pos = self.relative_pos_encoder(self.relative_pos)
        rel_pos = F.pad(rel_pos, (0, 0, pad_len, pad_len), value=0)

        # Transformer layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, lookup_table=rel_pos)
            x = self.pffn_layers[i](x)

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # decoder
        x = self.decoder(x.transpose(1, 2)).transpose(1, 2)

        # unmask original motion
        x[..., :D] = x[..., :D] * (1 - mask) + original_motion * mask

        # contact
        if not self.is_context:
            x[..., -4:] = torch.sigmoid(x[..., -4:])

        return x, mask

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

class TwoStageGAN(nn.Module):
    def __init__(self, d_motion, config):
        super(TwoStageGAN, self).__init__()

        self.d_motion = d_motion
        self.config   = config

        # ContextGAN
        self.G_context = Generator(self.d_motion, self.config, is_context=True)
        self.D_context = Discriminator(self.d_motion, self.config)

        # DetailGAN
        self.G_detail = Generator(self.d_motion, self.config, is_context=False)
        self.D_detail = Discriminator(self.d_motion, self.config)

    def generate(self, motion, traj, p_unmask=0.0):
        recon_context, mask = self.G_context.forward(motion, traj, mask=None, p_unmask=p_unmask)
        recon_detail,  _    = self.G_detail.forward(recon_context, traj, mask=mask, p_unmask=p_unmask)
        recon_detail, recon_contact = torch.split(recon_detail, [self.d_motion, 4], dim=-1)
        return recon_context, recon_detail, recon_contact
    
    def discriminate(self, context, detail):
        short_scores_context, long_scores_context = self.D_context.forward(context)
        short_scores_detail,  long_scores_detail  = self.D_detail.forward(detail)
        return short_scores_context, long_scores_context, short_scores_detail, long_scores_detail

class ContextGAN(nn.Module):
    def __init__(self, d_motion, config):
        super(ContextGAN, self).__init__()

        self.d_motion = d_motion
        self.config   = config

        self.G = Generator(self.d_motion, self.config, is_context=True)
        self.D = Discriminator(self.d_motion, self.config)
    
    def generate(self, motion, traj, p_unmask=0.0):
        recon, mask = self.G.forward(motion, traj, mask=None, p_unmask=p_unmask)
        return recon, mask
    
    def discriminate(self, motion):
        short_scores, long_scores = self.D.forward(motion)
        return short_scores, long_scores