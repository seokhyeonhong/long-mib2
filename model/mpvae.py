import torch
import torch.nn as nn

from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet
from pymovis.learning.embedding import SinusoidalPositionalEmbedding

class MotionPredictionVAE(nn.Module):
    """
    Conditional VAE for motion prediction, conditioned on trajectory.
    """
    def __init__(self, d_motion, d_traj, config):
        super(MotionPredictionVAE, self).__init__()
        self.d_motion = d_motion
        self.d_traj = d_traj
        self.config = config

        self.encoder = MotionEncoder(d_motion, d_traj, config)
        self.decoder = MotionDecoder(d_motion, d_traj, config)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_mask(self, motion):
        B, T, D = motion.shape
        mask = torch.ones(B, T, 1).to(motion.device)
        mask[:, self.config.context_frames:] = 0
        return mask
    
    def forward(self, motion, traj):
        # mask
        mask = self.get_mask(motion)

        # encoder
        mean, logvar = self.encoder.forward(motion, mask, traj)
        z = self.reparameterize(mean, logvar)

        # decoder
        recon = self.decoder.forward(motion, mask, traj, z)
        return recon, mean, logvar
    
    def sample(self, motion, traj):
        B, T, D = traj.shape
        mask = self.get_mask(motion)
        z = torch.randn(B, 1, self.config.d_model).to(traj.device)
        recon = self.decoder.forward(motion, mask, traj, z)
        return recon
    
class MotionEncoder(nn.Module):
    def __init__(self, d_motion, d_traj, config):
        super(MotionEncoder, self).__init__()
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

        # tokens
        self.mu_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.logvar_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # linear projection
        self.fc = nn.Sequential(
            nn.Linear(self.d_motion + self.d_traj + 1, self.d_model), # (motion, traj, mask=1)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        
        # positional encodings
        self.embedding = SinusoidalPositionalEmbedding(self.d_model, max_len=100) # arbitrary max_len

        # Transformer layers
        self.attn_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

    def forward(self, motion, mask, traj):
        B, T, D = motion.shape

        # concat
        x = torch.cat([motion, mask, traj], dim=-1)

        # linear projection
        x = self.fc(x)

        # tokens
        mean = self.mu_token.repeat(B, 1, 1)
        logvar = self.logvar_token.repeat(B, 1, 1)
        x = torch.cat([mean, logvar, x], dim=1)
        
        # add positional encodings
        pos = torch.arange(T+2, device=x.device) # +2 for tokens
        x = x + self.embedding(pos)

        # Transformer layers
        for attn_layer, pffn_layer in zip(self.attn_layers, self.pffn_layers):
            x = attn_layer(x, x)
            x = pffn_layer(x)
        
        # output
        mean = x[:, 0:1, :]
        logvar = x[:, 1:2, :]

        return mean, logvar

class MotionDecoder(nn.Module):
    def __init__(self, d_motion, d_traj, config):
        super(MotionDecoder, self).__init__()
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

        # linear projection
        self.fc = nn.Sequential(
            nn.Linear(self.d_motion + self.d_traj + 1, self.d_model), # (motion, traj, mask=1)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        
        # positional encodings
        self.embedding = SinusoidalPositionalEmbedding(self.d_model, max_len=100) # arbitrary max_len

        # Transformer layers
        self.layer_norm  = nn.LayerNorm(self.d_model)
        self.attn_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.cross_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_motion),
        )

    def forward(self, motion, mask, traj, z):
        B, T, D = motion.shape

        # linear projection
        original_motion = motion.clone()
        masked_motion = motion * mask
        x = self.fc(torch.cat([masked_motion, mask, traj], dim=-1))
        
        # add positional encodings
        pos = torch.arange(T, device=x.device)
        x = x + self.embedding(pos)

        # Transformer layers
        for attn_layer, cross_layer, pffn_layer in zip(self.attn_layers, self.cross_layers, self.pffn_layers):
            x = attn_layer(x, x)
            x = cross_layer(x, z)
            x = pffn_layer(x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)
        x = original_motion * mask + x * (1 - mask)
        return x