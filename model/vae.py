import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.ops import rotation, motionops
from pymovis.learning.mlp import MultiLinear
from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, LocalMultiHeadAttention

def get_mask(batch, context_frames, p_unmask=0.0):
    B, T, D = batch.shape

    # 0 for unknown frames, 1 for known frames
    batch_mask = torch.ones(B, T, 1, dtype=batch.dtype, device=batch.device)
    batch_mask[:, context_frames:-1, :] = 0

    # schedule sampling
    frames  = torch.arange(T)
    samples = torch.rand(T)
    batch_mask[:, frames[samples < p_unmask], :] = 1

    return batch_mask

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
    
""" ContextVAE """
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
            nn.Linear(self.d_motion, self.d_model),
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
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, self.fps+1, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_motion if self.is_context else self.d_model),
        )

    def forward(self, x):
        B, T, D = x.shape

        # motion encoder
        x = self.motion_encoder(x)
        mu = self.mu_token.repeat(B, 1, 1)
        logvar = self.logvar_token.repeat(B, 1, 1)
        x = torch.cat([mu, logvar, x], dim=1)
        T += 2

        # relative positional encoding
        pad_len = T-self.fps//2-1
        rel_pos = self.relative_pos_encoder(self.relative_pos)
        rel_pos = F.pad(rel_pos, (0, 0, pad_len, pad_len), value=0)

        # Transformer layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, lookup_table=rel_pos)
            x = self.pffn_layers[i](x)

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # decoder
        x = self.decoder(x)

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
            nn.Linear(self.d_motion + 1, self.d_model), # (motion, mask)
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
            nn.Linear(self.d_model, self.d_motion if self.is_context else self.d_motion + 4),
        )
    
    def forward(self, motion, traj, z, mask=None, p_unmask=0.0):
        B, T, D = motion.shape

        # original motion
        original_motion = motion.clone()

        # mask out and infill with z
        if self.is_context:
            mask = get_mask(motion, self.config.context_frames, p_unmask=p_unmask)
            motion = motion * mask + z * (1 - mask)

        # encoders
        x = self.motion_encoder(torch.cat([motion, mask], dim=-1))
        context = self.traj_encoder(traj)

        # relative positional encoding
        pad_len = T - (self.fps//2) - 1
        rel_pos = self.relative_pos_encoder(self.relative_pos)
        rel_pos = F.pad(rel_pos, (0, 0, pad_len, pad_len), value=0)

        # Transformer layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x if self.is_context else z, lookup_table=rel_pos)
            x = self.cross_layers[i](x, context, lookup_table=rel_pos)
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

        return x, mask

class VAE(nn.Module):
    def __init__(self, d_motion, config, is_context):
        super(VAE, self).__init__()

        self.d_motion   = d_motion
        self.config     = config
        self.is_context = is_context

        self.encoder = Encoder(d_motion, config, is_context=is_context)
        self.decoder = Decoder(d_motion, config, is_context=is_context)
    
    def forward(self, motion, traj, mask=None, p_unmask=0.0):
        B, T, D = motion.shape

        mu, logvar = self.encoder.forward(motion)
        z = reparameterize(mu.unsqueeze(1).repeat(1, T, 1), logvar.unsqueeze(1).repeat(1, T, 1))

        recon, mask = self.decoder.forward(motion, traj, z, mask=mask, p_unmask=p_unmask)
        
        return recon, mask, mu, logvar
    
    def sample(self, motion, traj, mask=None):
        B, T, D = motion.shape
        z = torch.randn(B, T, self.d_motion if self.is_context else self.config.d_model, dtype=motion.dtype, device=motion.device)
        pred_motion, mask = self.decoder.forward(motion, traj, z, mask=mask)
        return pred_motion, mask