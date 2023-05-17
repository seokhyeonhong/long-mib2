import numpy as np
import torch
import torch.nn as nn

from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet

def get_mask(batch, context_frames, ratio_constrained=0.1):
    B, T, D = batch.shape

    # 0 for unknown frames, 1 for known frames
    batch_mask = torch.ones(B, T, 1, dtype=batch.dtype, device=batch.device)
    batch_mask[:, context_frames:-1, :] = 0
    
    # False for known frames, True for unknown frames
    atten_mask = torch.zeros(T, T, dtype=torch.bool, device=batch.device)
    atten_mask[:, context_frames:-1] = True

    # randomly mask out some frames
    probs = torch.rand(T) < ratio_constrained
    batch_mask[:, probs, :] = 1
    atten_mask[:, probs] = False
            
    return batch_mask, atten_mask

def get_keyframe_relative_position(window_length, context_frames):
    position = torch.arange(window_length, dtype=torch.float32)
    dist_ctx = position - (context_frames - 1) # distance to the last context frame
    dist_tgt = position - (window_length - 1)  # distance to the target frame

    p_kf = torch.stack([dist_ctx, dist_tgt], dim=-1) # (T, 2)

    return p_kf

class ContextTransformer(nn.Module):
    def __init__(self, d_motion, config, d_traj=0):
        super(ContextTransformer, self).__init__()
        self.d_motion = d_motion
        self.d_traj = d_traj
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
        
        # encoders
        self.encoder = nn.Sequential(
            nn.Linear(self.d_motion + self.d_traj + 1 , self.d_model), # (motion, traj, mask(=1))
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        self.keyframe_pos_encoder = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(self.dropout),
        )
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
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion),
        )
    
    def forward(self, x, traj=None, ratio_constrained=0.1):
        B, T, D = x.shape

        original_x = x.clone()
        
        # mask
        batch_mask, atten_mask = get_mask(x, self.config.context_frames, ratio_constrained=ratio_constrained)
        masked_x = x * batch_mask

        # encoder
        if traj is None:
            x = self.encoder(torch.cat([masked_x, batch_mask], dim=-1))
        else:
            x = self.encoder(torch.cat([masked_x, traj, batch_mask], dim=-1))

        # add keyframe positional embedding
        keyframe_pos = get_keyframe_relative_position(T, self.config.context_frames).to(x.device)
        x = x + self.keyframe_pos_encoder(keyframe_pos)

        # relative distance range: [-T+1, ..., T-1], 2T-1 values in total
        rel_dist = torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(-1).to(x.device) # (2T-1, 1)
        lookup_table = self.relative_pos_encoder(rel_dist) # (2T-1, d_model)

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, mask=atten_mask, lookup_table=lookup_table) # self-attention
            x = self.pffn_layers[i](x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)

        # recover the constrained frames from the original input
        x[:, :self.config.context_frames] = original_x[:, :self.config.context_frames]
        x[:, -1] = original_x[:, -1]

        return x, batch_mask

class DetailTransformer(nn.Module):
    def __init__(self, d_motion, d_traj, config):
        super(DetailTransformer, self).__init__()
        self.d_motion = d_motion
        self.d_traj = d_traj
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
        
        # encoders
        self.encoder = nn.Sequential(
            nn.Linear(self.d_motion + self.d_traj + 1, self.d_model), # (motion, traj, mask(=1))
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
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
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion + 4),
        )
    
    def forward(self, x, traj, batch_mask):
        B, T, D = x.shape

        original_x = x.clone()
        
        # mask
        x = self.encoder(torch.cat([x, traj, batch_mask], dim=-1))

        # relative distance range: [-T+1, ..., T-1], 2T-1 values in total
        rel_dist = torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(-1).to(x.device) # (2T-1, 1)
        lookup_table = self.relative_pos_encoder(rel_dist) # (2T-1, d_model)

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, mask=None, lookup_table=lookup_table)
            x = self.pffn_layers[i](x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)

        # recover the constrained frames from the original input
        x[:, :self.config.context_frames, :self.d_motion] = original_x[:, :self.config.context_frames]
        x[:, -1, :self.d_motion] = original_x[:, -1]

        # split motion and contact
        motion, contact = torch.split(x, [self.d_motion, 4], dim=-1)
        contact = torch.sigmoid(contact)

        return motion, contact