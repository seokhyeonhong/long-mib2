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

class ContextualAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, dropout=0.1):
        super(ContextualAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head

        self.W_q = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_out = nn.Linear(n_head * d_head, d_model)

        self.atten_scale = 1 / (d_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context, mask=None):
        B, T1, D = x.shape
        _, T2, _ = context.shape

        # linear projection
        q = self.W_q(x) # (B, T1, n_head*d_head)
        k = self.W_k(context) # (B, T2, n_head*d_head)
        v = self.W_v(context) # (B, T2, n_head*d_head)

        # split heads
        q = q.view(B, T1, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T1, d_head)
        k = k.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)
        v = v.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)

        # attention score
        atten_score = torch.matmul(q, k.transpose(-2, -1)) # (B, n_head, T1, T2)
        atten_score *= self.atten_scale # (B, n_head, T1, T2)

        # mask
        if mask is not None:
            atten_score.masked_fill_(mask, -1e9)
        
        # attention
        attention = F.softmax(atten_score, dim=-1) # (B, n_head, T1, T2)
        attention = torch.matmul(attention, v).transpose(1, 2).contiguous().view(B, -1, self.n_head * self.d_head) # (B, T1, n_head*d_head)

        # output
        output = self.W_out(attention) # (B, T1, d_model)
        output = self.dropout(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_head, n_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.attn = ContextualAttention(d_model, d_head, n_heads, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(d_model*2, d_model), nn.LeakyReLU())
        self.mlp = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        attn = self.attn.forward(x, x, mask=mask)
        attn = self.dropout(attn)

        x = torch.cat([x, attn], dim=-1)
        x = self.fc.forward(x)
        x = self.mlp.forward(x)
        return x

class TransformerStage(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_head, d_ff, dropout):
        super(TransformerStage, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([TransformerBlock(d_model, d_head, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )

    def forward(self, x, mask=None):
        original = x.clone()
        for layer in self.layers:
            x = layer.forward(x, mask=mask)
            if mask is not None:
                mask = (1 - F.max_pool1d(1 - mask.float(), kernel_size=3, stride=1, padding=1)).bool()
        x = self.conv.forward(x.transpose(1, 2)).transpose(1, 2)
        return x + original, mask

class ProgressiveTransformer(nn.Module):
    def __init__(self, d_motion, config):
        super(ProgressiveTransformer, self).__init__()
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
            nn.Conv1d(self.d_motion * 2 + 5, self.d_model, kernel_size=3, padding=1), # (motion, mask) + traj(=5)
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
        )
        # if self.is_context:
        #     self.position_encoder = nn.Sequential(
        #         nn.Linear(2, self.d_model),
        #         nn.LeakyReLU(),
        #         nn.Dropout(self.dropout),
        #         nn.Linear(self.d_model, self.d_model),
        #         nn.LeakyReLU(),
        #         nn.Dropout(self.dropout),
        #     )

        # # relative positional encoder
        # self.relative_pos_encoder = nn.Sequential(
        #     nn.Linear(1, self.d_model),
        #     nn.LeakyReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.d_model, self.d_head),
        #     nn.Dropout(self.dropout),
        # )

        # Transformer layers
        self.transformer_stages = nn.ModuleList()
        for _ in range(self.n_layers):
            self.transformer_stages.append(TransformerStage(self.d_model, 2, self.n_heads, self.d_head, self.d_ff, self.dropout))
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.d_motion),
        )
    
    def forward(self, motion, traj):
        B, T, D = motion.shape

        # original motion
        original_motion = motion.clone()

        # mask out
        batch_mask, attn_mask = get_mask(motion, self.context_frames)
        motion = motion * batch_mask

        # encoder
        x = torch.cat([motion, batch_mask, traj], dim=-1)
        x = self.motion_encoder(x.transpose(1, 2)).transpose(1, 2)

        # Transformer layers
        for i in range(self.n_layers):
            x, attn_mask = self.transformer_stages[i](x, mask=attn_mask)

        # decoder
        x = self.decoder(x)

        # unmask original motion
        x[..., :D] = x[..., :D] * (1 - batch_mask) + original_motion * batch_mask


        return x