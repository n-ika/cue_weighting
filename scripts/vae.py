import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class MelVAE(nn.Module):
    def __init__(self, max_len, dec=False, N=1):
        super(MelVAE, self).__init__()
        self.dec = dec
        self.max_len = max_len

        self.encoder = nn.Sequential(
            # Full resolution first — learn local patterns before compressing
            nn.Conv2d(1, 16, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.2),

            # Compress time only, freq stays at 80
            nn.Conv2d(16, 32, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),

            # Mild freq compression, more time compression
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),
        )

        # # Infer actual output shape — no manual arithmetic, no bugs
        # with torch.no_grad():
        #     dummy   = torch.zeros(1, 1, 80, max_len)
        #     enc_out = self.encoder(dummy)
        #     self.enc_shape  = enc_out.shape[1:]   # (C, freq, time)
        #     self.flatten_dim = enc_out.numel()
        # print(f"Encoder output shape: {enc_out.shape}, flatten_dim: {self.flatten_dim}")

        self.fc_mu     = nn.Conv2d(64, 16, kernel_size=1)
        self.fc_logvar = nn.Conv2d(64, 16, kernel_size=1)
        with torch.no_grad():
            dummy   = torch.zeros(1, 1, 80, max_len)
            enc_out = self.encoder(dummy)
            self.flatten_dim = 16 * enc_out.shape[2] * enc_out.shape[3]
        # print(f"Encoder output shape: {enc_out.shape}, flatten_dim: {self.flatten_dim}")
        self.decision_layer = nn.Linear(self.flatten_dim, N)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(3,3), stride=(2,2), padding=1, output_padding=(1,1)),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.GroupNorm(8, 32),
            # nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=(3,3), stride=(2,2), padding=1, output_padding=(1,1)),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 1, kernel_size=(3,3), stride=(2,2), padding=1, output_padding=(1,1)),
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z):
        x = self.decoder(z)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        if self.dec:
            z_flat = torch.flatten(z, start_dim=1)
            decision = self.decision_layer(z_flat)
        else:
            decision = torch.tensor(0.0, device=x.device) 
        return x_recon, mu, logvar, decision




class Conv1dVAE(nn.Module):
    def __init__(self, max_len, n_channels, dec=False, N=1):
        super(Conv1dVAE, self).__init__()
        self.dec = dec
        self.max_len = max_len
        self.N = N
        self.n_channels = n_channels
        if self.N > 1:
            self.n_channels_latent = 64
        else:
            self.n_channels_latent = self.n_channels

        # 1. ENCODER
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.2),

            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),
        )

        # Infer actual output shape automatically
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, max_len)
            enc_out = self.encoder(dummy)
            self.enc_shape = enc_out.shape[1:]   # (Channels, Time)
            self.flatten_dim = enc_out.numel()
        # print(f"Encoder output shape: {enc_out.shape}, flatten_dim: {self.flatten_dim}")

        # 2. Mu & LOGVAR
        self.fc_mu     = nn.Conv1d(64, self.n_channels_latent, kernel_size=1)
        self.fc_logvar = nn.Conv1d(64, self.n_channels_latent, kernel_size=1)
        
        # Decision layer uses latent vector as input
        self.z_flatten_dim = self.n_channels_latent * self.enc_shape[1]
        # print(f"Decision layer input dimension: {self.z_flatten_dim}")
        self.decision_layer = nn.Linear(self.z_flatten_dim, N)

        # 3. DECODER
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.n_channels_latent, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
            
            # Smoothing layer
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(16, self.n_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        x = self.encoder(x)
        # x = self.flatten(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z):
        x = self.decoder(z)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        if self.dec:
            z_flat = torch.flatten(z, start_dim=1)
            decision = self.decision_layer(z_flat) #USES INTERNAL SIGMOID
        else:
            decision = torch.tensor(0.0, device=x.device)
        return x_recon, mu, logvar, decision


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function: reconstruction loss + KL divergence
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (beta-VAE)
        
    Returns:
        Total loss, reconstruction loss, KL divergence
    """
    # Reconstruction loss (MSE for spectrograms)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Total loss
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div, 0


def filter_cue(x, cue='formants', weight=0.99):
    """
    Upweighted duration/formant filter for reconstructed spectrograms

    Args:
        recon_x: Reconstructed input
        x: Original input
    Returns:
        Mask indicating which samples pass the duration/formant filter
    """

    # dur_threshold = 0.2
    # formant_threshold = 0.4
    if cue == 'duration':
        avg_value = torch.mean(x[:,0,:,2],dim=(1))
        view_shape = [-1] + [1] * (x.ndim - 1)
        avg_value = avg_value.view(*view_shape)
        # mask = torch.where(x < avg_value*0.5, weight, 1-weight)
        mask = torch.where(x < avg_value, weight, 1-weight)
    elif cue == 'formants':
        mask = torch.full(x.shape, 1-weight)
        mask[:, :, :, :8] = weight
        # max_threshold = min(avg_value*1.75, 0.9)
        # mask = torch.where(x > avg_value, weight, 1-weight)
        # mask = torch.where(x >= max_threshold, weight, 1-weight)
    else:
        mask = torch.ones(x.size())
    return mask


def filter_cue_mfcc(x, cue='formants', weight=0.99):
    """
    Upweighted duration/formant filter for reconstructed spectrograms

    Args:
        recon_x: Reconstructed input
        x: Original input
    Returns:
        Mask indicating which samples pass the duration/formant filter
    """
    if cue == 'duration':
        mask = torch.full(x.shape, 1-weight)
        mask[:, 5:, :8] = weight
    elif cue == 'formants':
        mask = torch.full(x.shape, 1-weight)
        mask[:, :5, :8] = weight
    else:
        mask = torch.ones(x.size())
    return mask

    # """
    # Filter out reconstructions that are too short based on energy threshold
    
    # Args:
    #     recon_x: Reconstructed input
    #     x: Original input
    #     threshold: Energy threshold to determine valid duration
        
    # Returns:
    #     Mask indicating which samples pass the duration filter
    # """
    # batch_size = x.size(0)
    # mask = torch.zeros(batch_size, dtype=torch.bool)
    
    # for i in range(batch_size):
    #     original_energy = torch.sum(x[i]**2)
    #     recon_energy = torch.sum(recon_x[i]**2)
        
    #     if recon_energy / original_energy > threshold:
    #         mask[i] = True
            
    # return mask


# def masked_recon_loss(recon_x, x, lengths):
#     """
#     recon_x, x: [B, 1, n_mels, T]  (T = max_len, padded)
#     lengths:    [B]  -- number of REAL (non-padded) time frames per sample
#     """
#     B, C, F, T = x.shape
#     device = x.device

#     # Build a [B, T] mask: True for real frames, False for padding
#     time_idx = torch.arange(T, device=device).unsqueeze(0)      # [1, T]
#     mask = (time_idx < lengths.unsqueeze(1))                    # [B, T], bool

#     # Expand mask to match x's shape: [B, 1, F, T]
#     mask = mask.unsqueeze(1).unsqueeze(1)          # [B, 1, 1, T]
#     mask = mask.expand(B, C, F, T).float()          # [B, 1, F, T] -> broadcasts to F

#     sq_diff = (recon_x - x) ** 2                    # [B, 1, F, T]
#     masked_sq_diff = sq_diff * mask

#     # Sum per-sample over valid elements only, normalize by valid count per-sample
#     per_sample_loss = masked_sq_diff.view(B, -1).sum(dim=1)     # [B]
#     valid_counts = mask.view(B, -1).sum(dim=1).clamp(min=1)     # [B], avoid /0

#     per_sample_loss = per_sample_loss / valid_counts   # mean over valid elements, per sample
#     recon_loss = per_sample_loss.mean()                 # mean over batch

#     return recon_loss

def omega_loss(recon_x, x, mu, logvar, decision, label, dec, 
               cue='duration', weight=0.99, beta=1.0):
    # 1. Reconstruction Loss
    recon_diff = (recon_x - x) ** 2

    # # 1.1 Apply cue-based weighting if weight != 1.0
    if weight != 1.0:
        if x.ndim == 3 and x.shape[1] == 13:
            omega_weight = filter_cue_mfcc(x, cue=cue, weight=weight).to(x.device)
            # recon_diff = recon_diff[:,:,10]
        else:
            omega_weight = filter_cue(x, cue=cue, weight=weight).to(x.device)
            # recon_diff = recon_diff[:,:,:,10]
        dims_to_sum = tuple(range(1, recon_diff.dim())) # Sum over everything except Batch (dim 0)
        recon_loss = torch.mean(torch.sum(recon_diff * omega_weight, dim=dims_to_sum))
        # recon_loss = torch.mean(torch.sum(recon_diff, dim=dims_to_sum))
    else:
        dims_to_sum = tuple(range(1, recon_diff.dim())) # Sum over everything except Batch (dim 0)
        recon_loss = torch.mean(torch.sum(recon_diff, dim=dims_to_sum))


    # 2. KL Divergence with Free Bits
    # free_bits = 0.1
    kl_per_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # [B, C, F, T]
    # kl_per_elem = torch.clamp(kl_per_elem, min=free_bits)           # per-element floor
    kl_div = kl_per_elem.view(kl_per_elem.size(0), -1).sum(dim=1)   # SUM over C*F*T -> [B]
    kl_div = kl_div.mean() 

    # 3. Decision Loss
    d_loss = dec_loss(decision, label) if dec else torch.tensor(0.0, device=x.device)
    
    # 4. Total Loss
    total = recon_loss + (beta * kl_div) + d_loss

    return total, recon_loss, kl_div, d_loss


def dec_loss(decision,label):
    if decision.ndim <= 2:
        ce_loss = nn.BCEWithLogitsLoss()
    else:
        ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(decision, label)
    return loss
