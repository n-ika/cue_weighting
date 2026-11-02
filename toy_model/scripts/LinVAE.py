import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class LinearVAE(nn.Module):
    def __init__(self, hdim=64, N=1):
        """
        VAE  using linear layers
        
        Args:
            h_dim: Dimension of bottleneck layer
        """
        super(LinearVAE, self).__init__()

        self.N = N
        self.hdim = hdim

        self.encoder = nn.Sequential(
            # Layer 1
            nn.Linear(2,hdim),
            nn.LeakyReLU(0.2),
            # Layer 2
            nn.Linear(hdim,hdim),
            nn.LeakyReLU(0.2),
            
            )
        # Latent space projections
        self.fc_mu = nn.Linear(hdim, 2)
        self.fc_logvar = nn.Linear(hdim, 2)
        # Decoder input
        self.decoder_input = nn.Linear(2, hdim)
        self.decision_layer = nn.Linear(2, self.N) #FIXME
        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1
            nn.Linear(hdim,hdim),
            nn.LeakyReLU(0.2),
            # Layer 2
            nn.Linear(hdim,2),
            nn.Sigmoid()
            )
        
    def encode(self, x):
        """Encode input to latent space parameters mu and logvar"""
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decide(self, z):
        """Decision function based on latent vector"""
        # decision = torch.sigmoid(decision_layer(z))
        decision = self.decision_layer(z)
        return decision

    def decode(self, z):
        """Decode latent vector to reconstruction"""
        x = self.decoder_input(z)
        # x = x.view(batch,self.h_dim,self.N//2,4)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Forward pass through the VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decision = self.decide(z)
        x_recon = self.decode(z)
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


def omega_loss(recon_x, x, mu, logvar, decision, label, device, cue='f1', weight=0.99, beta=1.0):
    """
    Omega loss function: reconstruction loss + KL divergence with modified weight
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        z: Latent vector
        label: Ground truth labels
        device: Device for tensor operations
        cue: Cue type for weighting
        weight: Weight for the upweighting filter
        beta: Weight for KL divergence term
        
    Returns:
        Total loss, reconstruction loss, KL divergence, decision loss
    """
    d_loss = dec_loss(decision,label)
    # # Ensure that weight is between 0 and 1
    # weight = np.abs(np.clip(weight,0,1))
    # squared_diff = (recon_x - x) ** 2
    # # bce_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
    # if cue == 'none':
    #     recon_loss = torch.mean(torch.sum(squared_diff))
    # else:
    #     if cue == 'formants':
    #         # print('Using formant cue weighting')
    #         weight1 = weight  # Weight for f1
    #         weight2 = 1 - weight  # Weight for duration
    #     elif cue == 'duration':
    #         weight1 = 1 - weight  # Weight for f1
    #         weight2 = weight  # Weight for duration
    #     masked_squared_diff = squared_diff * torch.tensor([weight1,weight2],dtype=torch.float32).to(device)
    #     recon_loss = torch.mean(torch.sum(masked_squared_diff))

    kl_div = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    # Total loss with omega modification
    # total_loss = recon_loss + beta * kl_div + d_loss
    total_loss =  beta * kl_div + d_loss #FIXME
     
    return total_loss, torch.tensor(0.0, device=x.device), beta * kl_div, d_loss #FIXME

def dec_loss(decision,label):
    if decision.shape[-1] == 1:
        ce_loss = nn.BCEWithLogitsLoss()
    else:
        ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(decision, label)
    return loss