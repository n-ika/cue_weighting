import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class ConvVAE(nn.Module):
    def __init__(self, dec=True, h_dim=64):
        """
        VAE for spectrograms using 2D convolutional layers
        
        Args:
            input_shape: Shape of input spectrogram (batch_size, channels, freq_bins, time_frames)
            h_dim: Dimension of bottleneck layer
        """
        super(ConvVAE, self).__init__()
        
        self.h_dim = h_dim
        self.N = h_dim//2
        self.pooled_dim = h_dim*self.N*2
        self.dec = dec

        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 8, kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            
            # Layer 2
            nn.Conv2d(8, 16, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            # Layer 3
            nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 4 (preserving time-frequency patterns)
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
            )
        # Latent space projections
        self.fc_mu = nn.Linear(self.pooled_dim, self.h_dim)
        self.fc_logvar = nn.Linear(self.pooled_dim, self.h_dim)
        self.decision_layer = nn.Linear(self.h_dim, 1) #FIXME

        # Decoder input
        self.decoder_input = nn.Linear(self.h_dim, self.pooled_dim)
        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(self.h_dim, 32, kernel_size=(4,3), stride=(2,2), output_padding=(1,1)), #padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 2
            nn.ConvTranspose2d(32, 16, kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            # Layer 3
            nn.ConvTranspose2d(16, 8, kernel_size=(4,3), stride=(2,2), output_padding=(1,1)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            
            # Layer 4
            nn.ConvTranspose2d(8, 1, kernel_size=(4,3), stride=(1,1))
            )
        # # Encoder
        # self.encoder = nn.Sequential(
        #     # Layer 1
        #     nn.Conv2d(1, 8, kernel_size=(4,3), stride=(2,1)),
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(0.2),
            
        #     # Layer 2
        #     nn.Conv2d(8, 16, kernel_size=(4,2), stride=(3,1)),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.2),
            
        #     # Layer 3
        #     nn.Conv2d(16, 32, kernel_size=(4,2), stride=(2,1)),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2),
            
        #     # Layer 4 (preserving time-frequency patterns)
        #     nn.Conv2d(32, 64, kernel_size=(4,2), stride=(1,1)),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2)
        # )
   
        # # Latent space projections
        # self.fc_mu = nn.Linear(self.pooled_dim, self.h_dim)
        # self.fc_logvar = nn.Linear(self.pooled_dim, self.h_dim)
        
        # # Decoder input
        # self.decoder_input = nn.Linear(self.h_dim, self.pooled_dim)
        
        # # Decoder
        # self.decoder = nn.Sequential(
        #     # Layer 1
        #     nn.ConvTranspose2d(h_dim, 32, kernel_size=(4,2), stride=(1,1)), #padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2),
            
        #     # Layer 2
        #     nn.ConvTranspose2d(32, 16, kernel_size=(4,2), stride=(2,1)), #,padding=(1,1), output_padding=(1,1)),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.2),
            
        #     # Layer 3
        #     nn.ConvTranspose2d(16, 8, kernel_size=(8,2), stride=(3,1)), #,padding=(1,1), output_padding=(1,1)),
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(0.2),
            
        #     # Layer 4
        #     nn.ConvTranspose2d(8, 1, kernel_size=(9,3), stride=(2,1)),
        # )
        
    def encode(self, x):
        """Encode input to latent space parameters mu and logvar"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.AdaptiveAvgPool1d(self.pooled_dim)(x)
        # Get latent space parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, batch, time):
        """Decode latent vector to reconstruction"""
        x = self.decoder_input(z)
        x = x.view(batch,self.h_dim,self.N//2,4)
        x = self.decoder(x)
        x = nn.functional.interpolate(x[:,0,:,:], size=time, mode='linear', align_corners=False).unsqueeze(1)
        return torch.sigmoid(x)
    
    def forward(self, x):
        """Forward pass through the VAE"""
        batch,channels,frequency,time = x.shape
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z,batch,time)
        if self.dec:
            decision = self.decision_layer(z)
        else:
            decision = 0
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
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


def filter_cue(x, cue='duration', weight=0.99):
    """
    Upweighted duration/formant filter for reconstructed spectrograms

    Args:
        recon_x: Reconstructed input
        x: Original input
    Returns:
        Mask indicating which samples pass the duration/formant filter
    """
    avg_value = torch.mean(x)
    # dur_threshold = 0.2
    # formant_threshold = 0.4
    if cue == 'duration':
        mask = torch.where(x < avg_value*0.5, weight, 1-weight)
    elif cue == 'formants':
        mask = torch.where(x >= avg_value, weight, 1-weight)
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


def omega_loss(recon_x, x, mu, logvar, decision, label, dec, device, cue='duration', weight=0.99, beta=1.0):
    """
    Omega loss function: reconstruction loss + KL divergence with modified weight
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
        
    Returns:
        Total loss, reconstruction loss, KL divergence
    """
    # Ensure that weight is between 0 and 1
    weight = np.abs(np.clip(weight,0,1))
    # mse_loss = F.mse_loss(recon_x, x, reduction='mean')
    recon_diff = (recon_x - x) **2
    if weight < 1.0:
        omega_weight = filter_cue(x, cue=cue, weight=weight).to(device)
        masked_squared_diff = recon_diff * omega_weight
        recon_loss = torch.mean(masked_squared_diff) #* 10000
    else:
        recon_loss = torch.mean(recon_diff) #* 10000
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if dec == True:
        d_loss = dec_loss(decision,label)
    else:
        d_loss = 0
    total_loss = recon_loss + beta * kl_div + d_loss
    
    return total_loss, recon_loss, kl_div, d_loss


def dec_loss(decision,label):
    if decision.ndim <= 2:
        ce_loss = nn.BCEWithLogitsLoss()
    else:
        ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(decision, label)
    return loss


class VAEmfcc(nn.Module):
    def __init__(self, dec=False, h_dim=4):
        """
        VAE for spectrograms using 2D convolutional layers
        
        Args:
            input_shape: Shape of input spectrogram (batch_size, channels, freq_bins, time_frames)
            h_dim: Dimension of bottleneck layer
        """
        super(VAEmfcc, self).__init__()

        self.dec = dec
        self.h_dim = h_dim
        self.N = h_dim//2
        self.pooled_dim = h_dim*self.N*2

        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 8, kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            
            # Layer 2
            nn.Conv2d(8, 16, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            # Layer 3
            nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 4 (preserving time-frequency patterns)
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
            )
        # Latent space projections
        self.fc_mu = nn.Linear(self.pooled_dim, self.h_dim)
        self.fc_logvar = nn.Linear(self.pooled_dim, self.h_dim)
        self.decision_layer = nn.Linear(self.h_dim, 1) #FIXME

        # Decoder input
        self.decoder_input = nn.Linear(self.h_dim, self.pooled_dim)
        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(self.h_dim, 32, kernel_size=(3,3), stride=(2,2)), #padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 2
            nn.ConvTranspose2d(32, 16, kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            # Layer 3
            nn.ConvTranspose2d(16, 8, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            
            # Layer 4
            nn.ConvTranspose2d(8, 1, kernel_size=(3,3), stride=(1,1))
            )

        
    def encode(self, x):
        """Encode input to latent space parameters mu and logvar"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.AdaptiveAvgPool1d(self.pooled_dim)(x)
        # Get latent space parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, batch, time):
        """Decode latent vector to reconstruction"""
        x = self.decoder_input(z)
        x = x.view(batch,self.h_dim,self.N//2,4)
        x = self.decoder(x)
        x = nn.functional.interpolate(x[:,0,:,:], size=time, mode='linear', align_corners=False).unsqueeze(1)
        return torch.sigmoid(x)
    
    def forward(self, x):
        """Forward pass through the VAE"""
        batch,channels,frequency,time = x.shape
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z,batch,time)
        if self.dec:
            decision = self.decision_layer(z)
        else:
            decision = 0
        return x_recon, mu, logvar, decision