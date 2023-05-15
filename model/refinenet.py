import torch
import torch.nn as nn
import random

from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, LocalMultiHeadAttention
from pymovis.ops import rotation

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

class RefineNet(nn.Module):
    def __init__(self, d_motion, d_traj, d_contact, config, local_attn=False, use_pe=True):
        super(RefineNet, self).__init__()
        self.d_motion = d_motion
        self.d_traj = d_traj
        self.d_contact = d_contact
        self.config = config
        self.local_attn = local_attn
        self.use_pe = use_pe

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
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.d_motion + self.d_traj + 1, self.d_model), # (motion, traj, mask)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        if self.use_pe:
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
        self.layer_norm   = nn.LayerNorm(self.d_model)
        self.attn_layers  = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            if self.local_attn:
                self.attn_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, self.config.receptive_size, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            else:
                self.attn_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion + self.d_contact),
        )
    
    def get_random_keyframes(self, t_total):
        keyframes = [self.config.context_frames-1]

        transition_start = self.config.context_frames
        while transition_start + self.config.max_transition < t_total - 1:
            transition_end = min(transition_start + self.config.max_transition, t_total - 1)
            kf = random.randint(transition_start + 5, transition_end)
            keyframes.append(kf)
            transition_start = kf

        if keyframes[-1] != t_total - 1:
            keyframes.append(t_total - 1)
        
        return keyframes
    
    def get_mask_by_keyframe(self, x, keyframes=None):
        B, T, D = x.shape
        mask = torch.zeros(B, T, 1, dtype=x.dtype, device=x.device)
        mask[:, :self.config.context_frames] = 1
        mask[:, -1] = 1
        if keyframes is not None:
            mask[:, keyframes] = 1
        return mask

    def get_interpolated_motion(self, motion, keyframes):
        B, T, D = motion.shape
        local_R6, root_p = torch.split(motion, [D-3, 3], dim=-1)
        local_R = rotation.R6_to_R(local_R6.reshape(B, T, -1, 6))

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
    
    def forward(self, interp_motion, traj, keyframes):
        B, T, D = interp_motion.shape
        
        # original motion
        original_x = interp_motion.clone()

        # mask
        batch_mask = self.get_mask_by_keyframe(interp_motion, keyframes)
        x = self.motion_encoder(torch.cat([interp_motion, traj, batch_mask], dim=-1))

        # add keyframe positional embedding
        if self.use_pe:
            keyframe_pos = get_keyframe_relative_position(T, self.config.context_frames).to(x.device)
            keyframe_pos = self.keyframe_pos_encoder(keyframe_pos)
            x = x + keyframe_pos

        # relative distance
        lookup_table = torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(-1).to(x.device) # (2T-1, 1)
        lookup_table = self.relative_pos_encoder(lookup_table) # (2T-1, d_head)

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.attn_layers[i](x, x, lookup_table=lookup_table)
            x = self.pffn_layers[i](x)

        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)

        # recover original motion
        x[:, :self.config.context_frames, :self.d_motion] = original_x[:, :self.config.context_frames]
        x[:, -1, :self.d_motion] = original_x[:, -1]

        # output
        motion, contact = torch.split(x, [self.d_motion, self.d_contact], dim=-1)
        contact = torch.sigmoid(contact)
        
        return motion, contact
