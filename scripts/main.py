import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from vae import *
from corpus_data_extraction import *
import pandas as pd
import argparse
from tqdm import tqdm
import random


def get_beta(current_step, warmup_steps, anneal_steps, max_beta):
    """
    Calculates the current beta value for VAE KL annealing.
    
    Args:
        current_step (int): The current global batch step.
        warmup_steps (int): Steps to hold beta at 0.
        anneal_steps (int): Steps to linearly increase beta.
        max_beta (float): The final target beta value.
    """
    if current_step < warmup_steps:
        return 0.0
    elif current_step < (warmup_steps + anneal_steps):
        # Calculate how far along we are in the annealing phase (0.0 to 1.0)
        progress = (current_step - warmup_steps) / anneal_steps
        return progress * max_beta
    else:
        # Annealing is done, hold at max_beta
        return max_beta
    
def evaluate_vae(model, data_loader, mapping, device, cue, weight, beta, dec, one_hot_lookup):
    """
    Evaluates a VAE on speech data, returning separated loss components.
    """
    model.eval()
    metrics = {
        'total_loss': 0.0,
        'recon_loss': 0.0,
        'kl_loss': 0.0,
        'dec_loss': 0.0,
        'count': 0
    }
    
    num_classes = len(mapping)
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Eval (VAE)")
        for batch in pbar:
            data = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            # data = data[..., :8] #FIXME
            labels_str = [f.split('/')[-1].split('_')[1] for f in batch[1]]
            labels = torch.tensor([mapping[label] for label in labels_str], dtype=torch.long)
            if num_classes > 2:
                labels = one_hot_lookup[labels].to(device)
            else:
                labels = labels.unsqueeze(1).to(device=device, dtype=torch.float32)
            
            batch_size = data.size(0)
            recon_batch, mu, logvar, decision = model(data)

            loss, recon_loss, kl_loss, d_loss = omega_loss(recon_batch, data, mu, logvar, 
                                                           decision, labels, dec,
                                                           cue=cue, weight=weight, beta=beta,
                                                           )
            metrics['total_loss'] += loss.item()
            metrics['recon_loss'] += recon_loss.item()
            metrics['dec_loss'] += d_loss.item()
            metrics['kl_loss'] += kl_loss.item()
            metrics['count'] += batch_size
    avg_metrics = {k: v / metrics['count'] for k, v in metrics.items() if k != 'count'}
    pbar.set_postfix(avg_metrics)
    return avg_metrics


def train_vae(model, dataloader, eval_loader, optimizer, device, cue, weight, epochs, beta, corpus_dir, dec, mapping):
    """Train the VAE model"""
    model.train()
    loss_track = []
    avg_eval_metrics_history = []
    global_step = 0
    warmup_ratio = 0.1
    anneal_ratio = 0.3
    
    total_steps = len(dataloader) * epochs
    WARMUP_STEPS = int(total_steps * warmup_ratio)
    ANNEAL_STEPS = int(total_steps * anneal_ratio)
    num_classes = len(mapping)
    one_hot_lookup = torch.eye(num_classes, device=device)

    for epoch in range(1,epochs+1):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        for idx, batch in enumerate(dataloader):
            data = batch[0].to(device)
            # data = data[..., :8] #FIXME
            labels_str = [f.split('/')[-1].split('_')[1] for f in batch[1]]
            labels = torch.tensor([mapping[label] for label in labels_str], dtype=torch.long)
            if num_classes > 2:
                labels = one_hot_lookup[labels].to(device)
            else:
                labels = labels.unsqueeze(1).to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            recon_batch, mu, logvar, decision = model(data)
            # print(recon_batch.shape)
            current_beta = get_beta(global_step, WARMUP_STEPS, ANNEAL_STEPS, beta)

            loss, recon_loss, kl_loss, d_loss = omega_loss(recon_batch, 
                                                   data, 
                                                   mu, 
                                                   logvar, 
                                                   decision,
                                                   labels, 
                                                   dec,                                                  
                                                   cue=cue, 
                                                   weight=weight, 
                                                   beta=current_beta)
            print('kl', kl_loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            loss_track.append(loss.item())
            global_step += 1
        if epoch % 10 == 0:
            avg_eval_metrics_history.append(evaluate_vae(model, eval_loader, mapping, device, cue, weight, beta, dec, one_hot_lookup))
            torch.save(model.state_dict(), f"{corpus_dir}/ckpt_{epoch}.pth")
        avg_loss = total_loss / len(dataloader) 
        avg_recon_loss = total_recon_loss / len(dataloader) 
        avg_kl_loss = total_kl_loss / len(dataloader) 
        print('total kl', total_kl_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, '
              f'Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}')

    df = pd.DataFrame(loss_track)
    df.to_csv(f"{corpus_dir}/loss.csv")
    torch.save(avg_eval_metrics_history, f"{corpus_dir}/eval_metrics.pth")


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    root = args.root if hasattr(args, 'root') else '/fs/clip-projects/realspeech/projects/speech_in_noise/me_final/'
    batch_size = args.batch_size
    # h_dim = args.h_dim
    learning_rate = args.learning_rate
    epochs = args.epochs
    beta = args.beta  # KL weight
    type_input = args.type_input
    cue = args.cue
    weight = args.weight
    name_corpus = args.name_corpus
    sim_num = args.sim_num
    only_ae_eh = args.only_ae_eh
    if args.dec:
        dec = '_dec' 
    else:
        dec = ''
    
    if type_input == 'art':
        nfft = 320
        step = 160
        n_channels = 6
    else:
        nfft = args.nfft
        step = args.step
        n_channels = 13
    print(f"Using cue: {cue} with weight: {weight}")
    corpus_data_id = f'{name_corpus}_{type_input}_{nfft}_{step}'
    df_meta = pd.read_csv(root+f'data/meta/{corpus_data_id}.csv')
    # if type_input == 'art':
    #     corpus_data_id = f'{name_corpus}_{type_input}'
    dataset_dict = torch.load(root+f'data/{corpus_data_id}.pt',weights_only=False)
    train_dict = dataset_dict['train']
    val_dict = dataset_dict['val']

    if only_ae_eh:
        df_meta = df_meta[df_meta['phone'].isin(['eh', 'ae'])]
        filenames = set(df_meta['filename'])
        train_dict = {k: v for k, v in train_dict.items() if k in filenames}
        val_dict = {k: v for k, v in val_dict.items() if k in filenames}
        mapping = {'eh': 0, 'ae': 1}
        mod_dir = f'models_filt{dec}'
        N = 1
    else:
        mod_dir = f'models_all{dec}'
        mapping = dataset_dict['mapping']
        N = 20
    corpus_out_dir = root+f"{mod_dir}/{corpus_data_id}/{beta}_{cue}_{weight}/sim_{sim_num}"
    os.makedirs(corpus_out_dir, exist_ok=True)
    os.makedirs(root+'figs', exist_ok=True)
    
    train_lengths = [v.shape[-1] for v in train_dict.values()]
    test_lengths = [v.shape[-1] for v in val_dict.values()]
    max_train = max(train_lengths)
    max_test = max(test_lengths)
    max_len = max(max_train, max_test)
    max_len = ((max_len + 7) // 8) * 8

    train_loader = DataLoader(CorpusData(train_dict), batch_size=batch_size, shuffle=True, collate_fn=PadCollate(dim=1,max_len=max_len,type_input=type_input))
    val_loader = DataLoader(CorpusData(val_dict), batch_size=batch_size, shuffle=True, collate_fn=PadCollate(dim=1,max_len=max_len,type_input=type_input))

    # max_len = 8 #FIXME 
    # Create model
    if type_input == 'mel':
        model = MelVAE(dec=args.dec, max_len=max_len, N=N).to(device)
    else:
        model = Conv1dVAE(max_len=max_len, n_channels=n_channels, dec=args.dec, N=N)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    train_vae(model, train_loader, val_loader, optimizer, device, 
              cue=cue, weight=weight, epochs=epochs, beta=beta, 
              corpus_dir=corpus_out_dir, dec=args.dec, mapping=mapping)

    # # Generate samples
    # with torch.no_grad():
    #     # Sample from latent space
    #     z = torch.randn(batch_size,h_dim).to(device)
    #     samples = model.decode(z,batch=batch_size,time=10).cpu()
        
    #     # Plot samples
    #     fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    #     for i, ax in enumerate(axes.flat):
    #         if i < len(samples):
    #             ax.imshow(samples[i,0,:,:], aspect='auto', origin='lower', cmap='viridis')
    #             ax.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(root+f"figs/{name_corpus}_recon_samples_{beta}_{cue}_{weight}.png")
    #     plt.close()

    # # Generate original samples
    # sample_batch = next(iter(train_loader))[0]
    # # sample_batch = sample_batch[0]
    # # Plot samples
    # fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    # for i, ax in enumerate(axes.flat):
    #     if i < len(sample_batch):
    #         ax.imshow(sample_batch[i,0,:,:], aspect='auto', origin='lower', cmap='viridis')
    #         ax.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(root+f"figs/{name_corpus}_original_samples_{beta}_{cue}_{weight}.png")
    #     plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="/fs/clip-projects/realspeech/projects/speech_in_noise/me_final/", help="Root directory for data and models")
    parser.add_argument("--name_corpus", "-n", type=str, default="tim", help="Corpus name for data loading")  
    parser.add_argument("--cue", "-c", type=str, choices=["duration", "formants", "none"], default="duration")
    parser.add_argument("--weight", "-w", type=float, default=0.99, help="Weight for the upweighting filter, between 0 and 1")
    parser.add_argument("--beta", "-b", type=float, default=1.0, help="Weight for the KL divergence term")
    parser.add_argument("--nfft", "-nf", type=int, default=512, help="FFT size for (mel) spectrogram extraction")
    parser.add_argument("--step", "-s", type=int, default=128, help="Hop length for (mel) spectrogram extraction")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size for training")
    # parser.add_argument("--h_dim", "-hd", type=int, default=64, help="Dimensionality of the latent space")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--type_input", "-t", type=str, default='spectral', help="Type of input for spectrogram extraction [spectral, mel]")
    parser.add_argument("--sim_num", "-sim", type=int, default=1, help="Simulation number")
    parser.add_argument("--only_ae_eh", action='store_true', help="Boolean. Whether to filter dataset to only include 'ae' and 'eh' phones")
    parser.add_argument("--dec", action='store_true', help="Decision model is used")

    args = parser.parse_args()
    print(args)
    print("Using CUDA:", torch.cuda.is_available())
    main(args)