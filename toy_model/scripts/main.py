import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
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
            data = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            labels = batch[1].to(device)
            if labels.ndim == 1:
                labels=labels.unsqueeze(1)
            batch_size = data.size(0)
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
            labels = batch[1].to(device) #[:,0].unsqueeze(1).to(device) #FIXME
            if labels.ndim == 1:
                labels=labels.unsqueeze(1)
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


def get_data(df_tim, filt):
    if filt:
        df_tim = df_tim[(df_tim.phone == 'ae') | (df_tim.phone == 'eh')]
    df_tim['f1_norm'] = df_tim.groupby('spk')['f1'].transform(lambda x: (x - x.mean()) / x.std())
    df_tim['f2_norm'] = df_tim.groupby('spk')['f2'].transform(lambda x: (x - x.mean()) / x.std())
    df_tim['dur_norm'] = df_tim.groupby('spk')['dur'].transform(lambda x: (x - x.mean()) / x.std())

    df_tr = df_tim[df_tim.data_type=='train']
    df_ts = df_tim[df_tim.data_type=='test']
    vals_tr = torch.tensor(df_tr[['f1_norm', 'dur_norm']].values).to(torch.float32)
    vals_ts = torch.tensor(df_ts[['f1_norm', 'dur_norm']].values).to(torch.float32)
    unique_labels = df_tim['phone'].unique()
    mapping = {label: index for index, label in enumerate(unique_labels)}
    labels_tr = torch.tensor([mapping[val] for val in df_tr['phone'].values]).to(torch.float32)
    labels_ts = torch.tensor([mapping[val] for val in df_ts['phone'].values]).to(torch.float32)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(vals_tr)
    X_val_scaled = scaler.transform(vals_ts) # Crucial: only use .transform() here!
    X_train_scaled = torch.from_numpy(X_train_scaled).float()
    X_val_scaled = torch.from_numpy(X_val_scaled).float()

    train_dataset = TensorDataset(X_train_scaled, labels_tr)
    val_dataset = TensorDataset(X_val_scaled, labels_ts)
    checkpoint = {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'scaler': scaler,
        'mapping': mapping
    }
    return(checkpoint)

def data2onehot(dataset,N):
    # make dataset into data + labels which are onehot instead of ints
    X = dataset.tensors[0] 
    y = dataset.tensors[1] 
    if N > 1:
        y_onehot = F.one_hot(y.long(), num_classes=N)
        dataset_newlabels = TensorDataset(X, y_onehot.to(torch.float32))
    else:
        dataset_newlabels = TensorDataset(X, y.to(torch.float32))
    return(dataset_newlabels)

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
    filt = args.filt
    print(f"Using weight: {weight}")

    if filt == False:
        name_corpus = name_corpus + '_all'
        N = 20
        hdim = 64
    else:
        N = 1
        hdim = 32
    corpus_out_dir = root+f"models_test/{name_corpus}/{cue}_{beta}_{weight}/sim{sim_num}" #FIXME !!!
    os.makedirs(corpus_out_dir, exist_ok=True)
    os.makedirs(root+'figs', exist_ok=True)
    df_tim = pd.read_csv('/fs/clip-realspeech/projects/speech_in_noise/me_final/data/meta/tim_mel_512_128.csv')

    # loaded_data = torch.load(root+f'data/{name_corpus}.pt',weights_only=False)
    loaded_data = get_data(df_tim,filt)
    # num_labels = torch.unique(loaded_data['train_dataset'].tensors[1]).shape[0]
    # if num_labels > 2:
    #     N = 20 
    # else:
    #     N = 1
    train_dataset = data2onehot(loaded_data['train_dataset'],N)
    val_dataset = data2onehot(loaded_data['val_dataset'],N)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = LinearVAE(hdim=hdim,N=N).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    train_vae(model, train_loader, val_loader, optimizer, device, 
                cue=cue, weight=weight, epochs=epochs, beta=beta, 
                corpus_out_dir=corpus_out_dir)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="/fs/clip-projects/realspeech/projects/speech_in_noise/me_final/toy_model/", help="Root directory for data and models")
    parser.add_argument("--name_corpus", "-n", type=str, default="timit", help="Corpus name for data loading")  
    parser.add_argument("--weight", "-w", type=float, default=0.999, help="Weight for the upweighting filter, between 0 and 1")
    parser.add_argument("--beta", "-b", type=float, default=1.0, help="Weight for the KL divergence term")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--cue", "-c", type=str, choices=["duration", "formants", "none"], default="none")
    parser.add_argument("--sim_num", "-sn", type=int, default=1, help="Simulation number")
    parser.add_argument("--filt", action='store_true', help="Only ae and eh used in training")


    args = parser.parse_args()
    print(args)
    print("Using CUDA:", torch.cuda.is_available())
    main(args)