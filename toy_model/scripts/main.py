import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import os
from LinVAE import *
import pandas as pd
import argparse
from tqdm import tqdm

def evaluate_vae(model, data_loader, device, cue, weight, beta):
    """
    Evaluates a VAE on speech data, returning separated loss components.
    """
    model.eval()
    metrics = {
        'total_loss': 0.0,
        'recon_loss': 0.0,
        'kl_loss': 0.0,
        'decision_loss': 0.0,
        'count': 0
    }
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Eval (VAE)")
        for batch in pbar:
            # Handle standard (data, target) or single (data,) loaders
            data = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            labels = batch[1][:,0].unsqueeze(1).to(device) #FIXME
            batch_size = data.size(0)
            # VAE forward pass usually returns: (reconstruction, mu, logvar)
            recon_batch, mu, logvar, decisions = model(data)
            loss, recon_loss, kl_loss, d_loss = omega_loss(recon_batch, 
                                                           data, 
                                                           mu, 
                                                           logvar, 
                                                           decisions, 
                                                           labels, 
                                                           device=device, 
                                                           cue=cue, 
                                                           weight=weight, 
                                                           beta=beta)
            # dec_loss = dec_loss(decisions, labels) #FIXME
            metrics['total_loss'] += loss.item()
            metrics['recon_loss'] += recon_loss.item()
            metrics['kl_loss'] += kl_loss.item()
            metrics['decision_loss'] += d_loss.item()
            metrics['count'] += batch_size
    # Normalize by total samples
    avg_metrics = {k: v / metrics['count'] for k, v in metrics.items() if k != 'count'}
    pbar.set_postfix(avg_metrics)
    return avg_metrics


def train_vae(model, dataloader, eval_loader, optimizer, device, cue, weight, epochs, beta, corpus_out_dir):
    """Train the VAE model"""
    model.train()
    loss_track = []
    for epoch in range(1,epochs+1):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        for idx, batch in enumerate(dataloader):
            data = batch[0].to(device)
            labels = batch[1][:,0].unsqueeze(1).to(device) #FIXME
            optimizer.zero_grad()
            recon_batch, mu, logvar, decisions = model(data) #FIXME
            loss, recon_loss, kl_loss, d_loss = omega_loss(recon_batch, 
                                                           data, 
                                                           mu, 
                                                           logvar, 
                                                           decisions, 
                                                           labels, 
                                                           device=device, 
                                                           cue=cue, 
                                                           weight=weight, 
                                                           beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            dec_loss_item = d_loss.item()
            loss_track.append(loss.item())
        if epoch % 20 == 0:
            evaluate_vae(model, eval_loader, device, cue, weight, beta)
            torch.save(model.state_dict(), f"{corpus_out_dir}/ckpt_{epoch}.pth")
        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon_loss = total_recon_loss / len(dataloader.dataset)
        avg_kl_loss = total_kl_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, '
              f'Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}',
              f'Decision Loss: {dec_loss_item:.4f}')

    df = pd.DataFrame(loss_track)
    df.to_csv(f'{corpus_out_dir}/loss.csv')


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    root = args.root if hasattr(args, 'root') else '/fs/clip-projects/realspeech/projects/speech_in_noise/me_final/toy_model/'
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    beta = args.beta  # KL weight
    cue = args.cue
    weight = args.weight if cue != 'none' else 1.0
    name_corpus = args.name_corpus
    sim_num = args.sim_num
    print(f"Using weight: {weight}")

    corpus_out_dir = root+f"models_dec/{name_corpus}/{cue}_{beta}_{weight}/sim{sim_num}" #FIXME
    os.makedirs(corpus_out_dir, exist_ok=True)
    os.makedirs(root+'figs', exist_ok=True)

    loaded_data = torch.load(root+f'data/{name_corpus}.pt',weights_only=False)
    train_dataset = loaded_data['train_dataset'] 
    val_dataset = loaded_data['val_dataset']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    sample_labels = next(iter(train_loader))[1][:,0]
    if sample_labels.unique().shape[0] > 2:
        N = 20 
    else:
        N = 1
    # Create model
    model = LinearVAE(N=N).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    train_vae(model, train_loader, val_loader, optimizer, device, 
                cue=cue, weight=weight, epochs=epochs, beta=beta, 
                corpus_out_dir=corpus_out_dir)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="/fs/clip-projects/realspeech/projects/speech_in_noise/me_final/toy_model/", help="Root directory for data and models")
    parser.add_argument("--name_corpus", "-n", type=str, default="timit", help="Corpus name for data loading")  
    parser.add_argument("--weight", "-w", type=float, default=0.99, help="Weight for the upweighting filter, between 0 and 1")
    parser.add_argument("--beta", "-b", type=float, default=1.0, help="Weight for the KL divergence term")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--cue", "-c", type=str, choices=["dur", "f1", "none"], default="duration")
    parser.add_argument("--sim_num", "-sn", type=int, default=1, help="Simulation number")

    args = parser.parse_args()
    print(args)
    print("Using CUDA:", torch.cuda.is_available())
    main(args)