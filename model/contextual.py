import torch
import torch.nn as nn

from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, MultiHeadBiasedAttention
from pymovis.learning.embedding import SinusoidalPositionalEmbedding

class ContextualVAE(nn.Module):
    """
    Conditional VAE for motion prediction, conditioned on trajectory.
    """
    def __init__(self, d_motion, d_traj, config):
        super(ContextualVAE, self).__init__()
        self.d_motion = d_motion
        self.d_traj = d_traj
        self.config = config

        self.encoder = ContextualEncoder(d_motion, d_traj, config)
        self.decoder = ContextualDecoder(d_motion, d_traj, config)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_mask(self, motion):
        mask = torch.ones(self.config.context_frames + 2, 1).to(motion.device) # +2 for transition and target
        mask[-2, :] = 0
        return mask
    
    def forward(self, motion, traj, target_pos):
        # mask
        mask = self.get_mask(motion)

        # encoder
        mean, logvar = self.encoder.forward(motion, traj, target_pos)
        z = self.reparameterize(mean, logvar)

        # decoder
        masked_motion = motion * mask
        recon = self.decoder.forward(masked_motion, traj, z, target_pos)
        recon = masked_motion + recon * (1 - mask)
        return recon, mean, logvar
    
    def sample(self, motion, traj, target_pos):
        B, T, D = traj.shape
        # mask
        mask = self.get_mask(motion)

        # decoder
        masked_motion = motion * mask
        z = torch.randn(B, 1, self.config.d_model).to(traj.device)
        recon = self.decoder.forward(masked_motion, traj, z, target_pos)
        return recon
    
class ContextualEncoder(nn.Module):
    def __init__(self, d_motion, d_traj, config):
        super(ContextualEncoder, self).__init__()
        self.d_motion = d_motion
        self.d_traj = d_traj
        self.config = config

        self.d_model        = config.d_model
        self.n_layers       = config.n_layers
        self.n_heads        = config.n_heads
        self.d_head         = self.d_model // self.n_heads
        self.d_ff           = config.d_ff
        self.max_len        = config.context_frames + config.max_transition + 1 + 2 # +1 for target frame, +2 for tokens
        self.pre_layernorm  = config.pre_layernorm
        self.dropout        = config.dropout

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")

        # tokens
        self.mu_token     = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.logvar_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # linear projection
        self.linaer = nn.Sequential(
            nn.Linear(self.d_motion + self.d_traj, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        
        # positional encodings
        self.embedding = SinusoidalPositionalEmbedding(self.d_model, max_len=self.max_len) # arbitrary max_len

        # Transformer layers
        self.attn_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadBiasedAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

    def forward(self, motion, traj, target_pos):
        B, T, D = motion.shape

        # concat and linear projection
        x = torch.cat([motion, traj], dim=-1)
        x = self.linaer(x)

        # concat tokens
        mean   = self.mu_token.repeat(B, 1, 1)
        logvar = self.logvar_token.repeat(B, 1, 1)
        x      = torch.cat([mean, logvar, x], dim=1)
        
        # add positional encodings
        pos = torch.arange(T+2, device=x.device) # +2 for tokens
        pos[-1] = target_pos
        x = x + self.embedding(pos)

        # Transformer layers
        time_to_arrival = (target_pos - self.config.context_frames + 1)
        atten_bias = torch.zeros(T+2, T+2, device=x.device)
        atten_bias[:, -1] = -time_to_arrival

        for attn_layer, pffn_layer in zip(self.attn_layers, self.pffn_layers):
            x = attn_layer(x, x, atten_bias)
            x = pffn_layer(x)
        
        # output
        mean = x[:, 0:1, :]
        logvar = x[:, 1:2, :]

        return mean, logvar

class ContextualDecoder(nn.Module):
    def __init__(self, d_motion, d_traj, config):
        super(ContextualDecoder, self).__init__()
        self.d_motion = d_motion
        self.d_traj = d_traj
        self.config = config

        self.d_model        = config.d_model
        self.n_layers       = config.n_layers
        self.n_heads        = config.n_heads
        self.d_head         = self.d_model // self.n_heads
        self.d_ff           = config.d_ff
        self.max_len        = config.context_frames + config.max_transition + 1 # +1 for target frame
        self.pre_layernorm  = config.pre_layernorm
        self.dropout        = config.dropout

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")

        # linear projection
        self.linear = nn.Sequential(
            nn.Linear(self.d_motion + self.d_traj, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        
        # positional encodings
        self.embedding = SinusoidalPositionalEmbedding(self.d_model, max_len=self.max_len) # arbitrary max_len

        # Transformer layers
        self.layer_norm  = nn.LayerNorm(self.d_model)
        self.attn_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadBiasedAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.cross_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_motion),
        )

    def forward(self, motion, traj, z, target_pos):
        B, T, D = motion.shape

        # linear projection
        x = torch.cat([motion, traj], dim=-1)
        x = self.linear(x)
        
        # add positional encodings
        pos = torch.arange(T, device=x.device)
        pos[-1] = target_pos
        x = x + self.embedding(pos)

        # Transformer layers
        time_to_arrival = (target_pos - self.config.context_frames + 1)
        atten_bias = torch.zeros(T, T, device=x.device)
        atten_bias[:, -1] = -time_to_arrival

        for attn_layer, cross_layer, pffn_layer in zip(self.attn_layers, self.cross_layers, self.pffn_layers):
            x = attn_layer(x, x, atten_bias)
            x = cross_layer(x, z)
            x = pffn_layer(x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)
        return x