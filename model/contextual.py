import torch
import torch.nn as nn

from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, MultiHeadBiasedAttention
from pymovis.learning.embedding import SinusoidalPositionalEmbedding

class ContextualTransformer(nn.Module):
    def __init__(self, d_motion, d_traj, config):
        super(ContextualTransformer, self).__init__()
        self.d_motion = d_motion
        self.d_traj = d_traj
        self.config = config

        self.d_model        = config.d_model
        self.n_layers       = config.n_layers
        self.n_heads        = config.n_heads
        self.d_head         = self.d_model // self.n_heads
        self.d_ff           = config.d_ff
        self.max_len        = config.context_frames + config.max_future + 1 + 1 # +1 for target frame, +1 for infinite
        self.pre_layernorm  = config.pre_layernorm
        self.dropout        = config.dropout

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")

        # linear projection
        self.linaer = nn.Sequential(
            nn.Linear(self.d_motion + self.d_traj + 1, self.d_model), # (motion, traj, mask(=1))
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

        # decoder
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_motion),
        )

    def get_mask(self, batch, constrained_target=True):
        B, T, D = batch.shape
        mask = torch.zeros(B, T, 1, dtype=batch.dtype, device=batch.device)
        mask[:, :self.config.context_frames, :] = 1
        if constrained_target:
            mask[:, -1, :] = 1
        
        return mask
    
    def forward(self, motion, traj, mask, target_pos):
        B, T, D = motion.shape

        # original motion
        original_motion = motion.clone()

        # clip
        target_pos = min(target_pos, self.max_len - 1)

        # concat and linear projection
        # motion += torch.randn_like(motion) * 0.001
        masked_motion = motion * mask
        x = torch.cat([masked_motion, traj, mask], dim=-1)
        x = self.linaer(x)

        # add positional encodings
        pos = torch.arange(T)
        pos[-1] = target_pos
        x = x + self.embedding(pos)

        # Transformer layers
        time_to_arrival = (target_pos - self.config.context_frames + 1)
        atten_bias = torch.zeros(T, T, dtype=x.dtype, device=x.device)
        atten_bias[:, -1] = min(30 - time_to_arrival, 0) if target_pos != self.max_len - 1 else -1e9

        for attn_layer, pffn_layer in zip(self.attn_layers, self.pffn_layers):
            x = attn_layer(x, x, atten_bias)
            x = pffn_layer(x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)

        recon = self.decoder(x)
        recon = original_motion * mask + recon * (1 - mask)

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
        self.max_len        = config.context_frames + config.max_future + 1 + 1 # +1 for target frame, +1 for infinite
        self.pre_layernorm  = config.pre_layernorm
        self.dropout        = config.dropout

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")

        # linear projection
        self.linaer = nn.Sequential(
            nn.Linear(self.d_motion + self.d_traj + 1, self.d_model), # (motion, traj, mask(=1))
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )

        # tokens
        self.mean_token   = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.logvar_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        # positional encodings
        self.embedding = SinusoidalPositionalEmbedding(self.d_model, max_len=self.max_len + 2) # arbitrary max_len, 2 for tokens

        # Transformer layers
        self.attn_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadBiasedAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
    
    def forward(self, motion, traj, mask, target_pos):
        B, T, D = motion.shape

        # clip
        target_pos = min(target_pos + 2, self.max_len - 1)

        # concat and linear projection
        # motion += torch.randn_like(motion) * 0.001
        x = torch.cat([motion, traj, mask], dim=-1)
        x = self.linaer(x)

        # concat tokens
        mean_token   = self.mean_token.repeat(B, 1, 1)
        logvar_token = self.logvar_token.repeat(B, 1, 1)
        x = torch.cat([mean_token, logvar_token, x], dim=1)

        # add positional encodings
        pos = torch.arange(T+2)
        pos[-1] = target_pos
        x = x + self.embedding(pos)

        # Transformer layers
        time_to_arrival = (target_pos - self.config.context_frames + 1)
        atten_bias = torch.zeros(T+2, T+2, dtype=x.dtype, device=x.device)
        atten_bias[:, -1] = min(30 - time_to_arrival, 0) if target_pos != self.max_len - 1 else -1e9

        for attn_layer, pffn_layer in zip(self.attn_layers, self.pffn_layers):
            x = attn_layer(x, x, atten_bias)
            x = pffn_layer(x)
        
        # decoder
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
        self.max_len        = config.context_frames + config.max_future + 1 + 1 # +1 for target frame, +1 for infinite
        self.pre_layernorm  = config.pre_layernorm
        self.dropout        = config.dropout

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")

        # linear projection
        self.linaer = nn.Sequential(
            nn.Linear(self.d_motion + self.d_traj + 1, self.d_model), # (motion, traj, mask(=1))
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        
        # positional encodings
        self.embedding = SinusoidalPositionalEmbedding(self.d_model, max_len=self.max_len) # arbitrary max_len

        # Transformer layers
        self.attn_layers  = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadBiasedAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.cross_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_motion),
        )

    def forward(self, motion, traj, mask, z, target_pos):
        B, T, D = motion.shape

        # original motion
        original_motion = motion.clone()

        # clip
        target_pos = min(target_pos, self.max_len - 1)

        # concat and linear projection
        # motion += torch.randn_like(motion) * 0.001
        masked_motion = motion * mask
        x = torch.cat([masked_motion, traj, mask], dim=-1)
        x = self.linaer(x)

        # add positional encodings
        pos = torch.arange(T)
        pos[-1] = target_pos
        x = x + self.embedding(pos)

        # Transformer layers
        time_to_arrival = (target_pos - self.config.context_frames + 1)
        atten_bias = torch.zeros(T, T, dtype=x.dtype, device=x.device)
        atten_bias[:, -1] = min(30 - time_to_arrival, 0) if target_pos != self.max_len - 1 else -1e9

        for attn_layer, cross_layer, pffn_layer in zip(self.attn_layers, self.cross_layers, self.pffn_layers):
            x = attn_layer(x, x, atten_bias)
            x = cross_layer(x, z)
            x = pffn_layer(x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)

        recon = self.decoder(x)
        recon = original_motion * mask + recon * (1 - mask)

        return recon

class ContextualVAE(nn.Module):
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
    
    def get_mask(self, batch, constrained_target=True):
        B, T, D = batch.shape
        mask = torch.zeros(B, T, 1, dtype=batch.dtype, device=batch.device)
        mask[:, :self.config.context_frames, :] = 1
        if constrained_target:
            mask[:, -1, :] = 1
        
        return mask
    
    def forward(self, motion, traj, target_pos, constrained=True):
        # mask
        mask = self.get_mask(motion, constrained_target=constrained)

        # encoder
        mean, logvar = self.encoder.forward(motion, mask, traj, target_pos)
        z = self.reparameterize(mean, logvar)

        # decoder
        recon = self.decoder.forward(motion, traj, mask, z, target_pos)
        return recon, mean, logvar
    
    def sample(self, motion, traj, target_pos, constrained=True):
        B, T, D = traj.shape
        mask = self.get_mask(motion, constrained_target=constrained)
        z = torch.randn(B, 1, self.config.d_model).to(traj.device)
        recon = self.decoder.forward(motion, traj, mask, z, target_pos)
        return recon