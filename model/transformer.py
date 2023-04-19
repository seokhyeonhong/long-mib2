import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi Head Contextual Periodic Biased Attention
class MultiHeadContextualBiasedAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, period, dropout=0.1):
        super(MultiHeadContextualBiasedAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.period = period

        self.W_q = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_out = nn.Linear(n_head * d_head, d_model)

        self.atten_scale = 1 / (d_head ** 0.5)
        self.dropout = nn.Dropout(dropout)

        self.head_specific_sclae = nn.Parameter(torch.pow(2, -torch.arange(1, n_head + 1, dtype=torch.float32)), requires_grad=False)
    
    def forward(self, x, context):
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
        atten_score = torch.matmul(q, k.transpose(-2, -1)) * self.atten_scale # (B, n_head, T1, T2)

        # temporal bias
        frame_from = torch.arange(T1, device=x.device).view(T1, 1)
        frame_to   = torch.arange(T2, device=x.device).view(1, T2)
        bias       = torch.abs(frame_from - frame_to) # (T1, T2)
        bias       = -torch.div(bias, self.period, rounding_mode='floor') # (T1, T2)
        bias       = bias.view(1, T1, T2) * self.head_specific_sclae.view(self.n_head, 1, 1) # (n_head, T1, T2)
        
        # attention
        attention = F.softmax(atten_score + bias, dim=-1) # (B, n_head, T1, T2)
        attention = torch.matmul(attention, v).transpose(1, 2).contiguous().view(B, -1, self.n_head * self.d_head) # (B, T1, n_head*d_head)

        # output
        output = self.W_out(attention) # (B, T1, d_model)
        output = self.dropout(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_motion, config):
        super(TransformerBlock, self).__init__()
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

        self.attention = MultiHeadContextualBiasedAttention(self.d_model, self.d_head, self.n_heads, self.fps, dropout=self.dropout)
        self.pffn = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_ff),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model),
            nn.Dropout(self.dropout)
        )

    def forward(self, x, context):
        x = x + self.pffn(torch.cat([self.attention(x, context), x], dim=-1))
        return x