import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.ops import rotation, motionops
from pymovis.learning.mlp import MultiLinear
from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, LocalMultiHeadAttention, LinearMultiHeadAttention

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

def _get_interpolated_motion(local_R, root_p, keyframes):
    R, p = local_R.clone(), root_p.clone()
    for i in range(len(keyframes) - 1):
        kf1, kf2 = keyframes[i], keyframes[i+1]
        t = torch.arange(0, 1, 1/(kf2-kf1), dtype=R.dtype, device=R.device).unsqueeze(-1)
        
        # interpolate joint orientations
        R1 = R[:, kf1].unsqueeze(1)
        R2 = R[:, kf2].unsqueeze(1)
        R_diff = torch.matmul(R1.transpose(-1, -2), R2)

        angle_diff, axis_diff = rotation.R_to_A(R_diff)
        angle_diff = t * angle_diff
        axis_diff = axis_diff.repeat(1, len(t), 1, 1)
        R_diff = rotation.A_to_R(angle_diff, axis_diff)

        R[:, kf1:kf2] = torch.matmul(R1, R_diff)

        # interpolate root positions
        p1 = p[:, kf1].unsqueeze(1)
        p2 = p[:, kf2].unsqueeze(1)
        p[:, kf1:kf2] = p1 + t * (p2 - p1)
    
    R6 = rotation.R_to_R6(R).reshape(R.shape[0], R.shape[1], -1)
    return torch.cat([R6, p], dim=-1)

def _get_random_keyframes(t_ctx, t_max, t_total):
    keyframes = [t_ctx-1]

    transition_start = t_ctx
    while transition_start + t_max < t_total - 1:
        transition_end = min(transition_start + t_max, t_total - 1)
        kf = random.randint(transition_start + 5, transition_end)
        keyframes.append(kf)
        transition_start = kf

    if keyframes[-1] != t_total - 1:
        keyframes.append(t_total - 1)
    
    return keyframes

def _get_mask_by_keyframe(x, t_ctx, keyframes=None):
    B, T, D = x.shape
    mask = torch.zeros(B, T, 1, dtype=x.dtype, device=x.device)
    mask[:, :t_ctx] = 1
    mask[:, -1] = 1
    if keyframes is not None:
        mask[:, keyframes] = 1
    return mask

""" ContextVAE """
class ContextEncoder(nn.Module):
    def __init__(self, d_motion, config):
        super(ContextEncoder, self).__init__()
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
            nn.Linear(self.d_model, self.d_motion),
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
        
        mu, logvar = x[:, 0], x[:, 1]

        return mu, logvar

class ContextDecoder(nn.Module):
    def __init__(self, d_motion, config):
        super(ContextDecoder, self).__init__()
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
            nn.Linear(self.d_motion, self.d_model),
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
    
    def forward(self, motion, traj, z):
        B, T, D = motion.shape

        # fill in missing frames with z
        mask = get_mask(motion, self.config.context_frames)
        motion = mask * motion + (1-mask) * z

        # encoder
        x = self.motion_encoder(motion)
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

        return x
    
class ContextVAE(nn.Module):
    def __init__(self, d_motion, config):
        super(ContextVAE, self).__init__()

        self.d_motion = d_motion
        self.config = config

        self.encoder = ContextEncoder(d_motion, config)
        self.decoder = ContextDecoder(d_motion, config)
    
    def forward(self, motion, traj):
        B, T, D = motion.shape

        mu, logvar = self.encoder(motion)
        mu = mu.unsqueeze(1).repeat(1, T, 1)
        logvar = logvar.unsqueeze(1).repeat(1, T, 1)
        z = self.reparameterize(mu, logvar)

        recon = self.decoder(motion, traj, z)
        
        return recon, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def sample(self, motion, traj):
        z = torch.randn_like(motion)
        return self.decoder(motion, traj, z)

""" ContextGAN """
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
            nn.Linear(self.d_motion, self.d_model),
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

        # fill in missing frames with z
        mask = get_mask(motion, self.config.context_frames)
        z    = torch.randn_like(motion)
        motion = mask * motion + (1-mask) * z

        # encoder
        x = self.motion_encoder(motion)
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

        return x

class ContextDiscriminator(nn.Module):
    def __init__(self, d_motion, config):
        super(ContextDiscriminator, self).__init__()
        
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
        B, T, D = x.shape
        
        short_scores = self.short_conv(x.transpose(1, 2)).squeeze(-1)
        long_scores  = self.long_conv(x.transpose(1, 2)).squeeze(-1)

        return short_scores, long_scores

class ContextGAN(nn.Module):
    def __init__(self, d_motion, config):
        super(ContextGAN, self).__init__()
        
        self.generator     = ContextGenerator(d_motion, config)
        self.discriminator = ContextDiscriminator(d_motion, config)

    def generate(self, motion, traj):
        return self.generator.forward(motion, traj)
    
    def discriminate(self, motion):
        return self.discriminator.forward(motion)