import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import os
from parselmouth.praat import call
import seaborn as sns
import torch
from LinVAE import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import statsmodels.formula.api as smf
import argparse



def get_val_results(val_loader, mapping, model, device):
    res = {'z':[], 'recon':[], 'mu':[], 'logvar':[], 'vowel':[], 'orig':[], 'f1':[], 'dur':[], 'dec':[]}
    for data, labels in val_loader:
        data = data.to(device)
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
        dec = model.decide(z)
        decision = torch.sigmoid(dec)
        if decision.ndim == 2 and decision.shape[1] == 20:
            # decision = torch.argmax(decision, dim=1)
            if decision[0,mapping['eh']] > decision[0,mapping['ae']]:
                res['dec'].append(0)
            elif decision[0,mapping['eh']] < decision[0,mapping['ae']]:
                res['dec'].append(1)
            else:
                res['dec'].append(0.5)
        else:
            res['dec'].extend(decision.cpu().detach())
        x_recon = model.decode(z)
        res['z'].append(z)
        res['recon'].append(x_recon)
        res['mu'].append(mu)
        res['logvar'].append(logvar)
        res['vowel'].append(labels.cpu().detach())
        res['f1'].append(data[0,0].cpu().detach())
        res['dur'].append(data[0,1].cpu().detach())
        res['orig'].append(data.cpu().detach())
    res['z'] = torch.cat(res['z'], dim=0).cpu().detach().numpy()
    res['recon'] = torch.cat(res['recon'], dim=0).cpu().detach().numpy()
    res['mu'] = torch.cat(res['mu'], dim=0).cpu().detach().numpy()
    res['logvar'] = torch.cat(res['logvar'], dim=0).cpu().detach().numpy()
    res['vowel'] = torch.cat(res['vowel'], dim=0).cpu().detach().numpy()
    res['f1'] = np.array(res['f1'])
    res['dur'] = np.array(res['dur'])
    res['orig'] = torch.cat(res['orig'], dim=0).cpu().detach().numpy()
    res['dec'] = np.array(res['dec']).flatten()
    if dec.shape[1] != 20:
        res['dec'] = np.where(res['dec'] > 0.5, 1, 0)
    return res


def test_model(model, wudf, mapping, scaler, device):
    print("Testing model on WU stimuli...")
    vals = wudf[['f1', 'dur']].to_numpy()
    vals = (vals - vals.mean(axis=0)) / vals.std(axis=0)
    vals = torch.tensor(scaler.transform(vals), dtype=torch.float32)
    labels = torch.tensor(wudf[['spec_step', 'dur_step']].to_numpy(), dtype=torch.float32)

    wu_d = {'z':[], 'recon':[], 'mu':[], 'logvar':[], 'orig':[], 'f1':[], 'dur':[],'dec':[],'acc':[]}
    for data, label in zip(vals, labels):
        data = data.to(device)
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
        decision = torch.sigmoid(model.decide(z))
        if decision.shape[0] == 20:
            # decision = torch.argmax(decision, dim=0)
            if decision[mapping['eh']] > decision[mapping['ae']]:
                wu_d['dec'].append(0)
            elif decision[mapping['eh']] < decision[mapping['ae']]:
                wu_d['dec'].append(1)
            else:
                wu_d['dec'].append(0.5)
        else:
            wu_d['dec'].append(decision.cpu().detach())
        x_recon = model.decode(z)
        wu_d['z'].append(z.unsqueeze(0))
        wu_d['recon'].append(x_recon)
        wu_d['mu'].append(mu)
        wu_d['logvar'].append(logvar)
        wu_d['f1'].append(label[0])
        wu_d['dur'].append(label[1])
        wu_d['orig'].append(data)
    wu_d['z'] = torch.cat(wu_d['z'], dim=0).cpu().detach().numpy()
    wu_d['recon'] = torch.cat(wu_d['recon'], dim=0).cpu().detach().numpy()
    wu_d['mu'] = torch.cat(wu_d['mu'], dim=0).cpu().detach().numpy()
    wu_d['logvar'] = torch.cat(wu_d['logvar'], dim=0).cpu().detach().numpy()
    wu_d['f1'] = torch.stack(wu_d['f1'], dim=0).cpu().detach().numpy()
    wu_d['dur'] = torch.stack(wu_d['dur'], dim=0).cpu().detach().numpy()
    wu_d['orig'] = torch.cat(wu_d['orig'], dim=0).cpu().detach().numpy()
    wu_d['dec'] = np.array(wu_d['dec']).flatten()
    if decision.shape[0] != 20:
        wu_d['dec'] = np.where(wu_d['dec'] > 0.5, 1, 0)
    dec_df = pd.DataFrame({
    'z1': wu_d['z'][:,0],
    'z2': wu_d['z'][:,1],
    'dec': wu_d['dec'],
    'f1': wu_d['f1'],
    'dur': wu_d['dur']
        })
    return(dec_df)



def get_weights(dec_df):
    lmm = smf.ols(formula = 'dec' +' ~ dur + f1', data=dec_df)
    results = lmm.fit()
    dur_w = np.abs(float(results.params.dur)) 
    spec_w = np.abs(float(results.params.f1))
    summed = dur_w + spec_w
    if summed > 0:
        cw_dur = dur_w/summed
        cw_spec = spec_w/summed
    else:
        print(summed,dur_w,spec_w)
    return cw_dur, cw_spec



def main(args):
    root = args.root

    if args.filt == True:
        name_corpus = 'timit'
        N = 1
        hdim = 32
    else:
        name_corpus = 'timit_all'
        N = 20
        hdim = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaded_data = torch.load(root+f'data/timit.pt') # only ever test on AE and EH
    val_dataset = loaded_data['val_dataset']
    scaler = loaded_data['scaler']
    mapping = loaded_data['mapping']
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    wudf = pd.read_csv('/fs/clip-realspeech/projects/speech_in_noise/me_final/data/meta/wustim_metadata.csv')
    wudf['dur'] = wudf.groupby('dur_step')['dur'].transform('mean')
    wudf['f1'] = wudf.groupby('spec_step')['f1'].transform('mean')

    fin = {
        'cue': [],
        'beta': [],
        'weight': [],
        'sim_num': [],
        'exp_cw_dur': [],
        'exp_cw_spec': [],
        'real_cw_dur': [],
        'real_cw_spec': []
    }
    for cue in ['none', 'formants', 'duration']:
        for beta in [0.01]: #, 0.0078, 0.0156]: #, 0.0321, 0.0625, 0.125, 0.25, 0.5, 1.0]:
            for sim_num in range(1,100+1):
                try:
                    weight = 1.0 if cue == 'none' else 0.999
                    model = LinearVAE(hdim=hdim,N=N).to(device)
                    model.load_state_dict(torch.load(root+f'models_dec/{name_corpus}/{cue}_{float(beta)}_{weight}/sim{sim_num}/ckpt_20.pth',
                                        map_location=device,
                                        weights_only=False))
                    print("Loaded model for cue: {}, beta: {}, sim_num: {}".format(cue, beta, sim_num))
                    res = get_val_results(val_loader, mapping, model, device)
                    res_df = pd.DataFrame({
                            'z1': res['z'][:,0],
                            'z2': res['z'][:,1],
                            'dec': res['dec'],
                            'f1': res['f1'],
                            'dur': res['dur']
                        })
                    if res_df.dec.nunique() > 1:
                        cw_dur, cw_spec = get_weights(res_df)
                    else:
                        cw_dur, cw_spec = 0,0
                    fin['real_cw_dur'].append(cw_dur)
                    fin['real_cw_spec'].append(cw_spec)
                    dec_df = test_model(model, wudf, mapping, scaler, device)
                    print("Tested model")
                    if dec_df.dec.nunique() > 1:
                        cw_dur, cw_spec = get_weights(dec_df)
                    else:
                        cw_dur, cw_spec = 0,0
                    print("Extracted weights: cw_dur: {:.4f}, cw_spec: {:.4f}".format(cw_dur, cw_spec))
                    fin['cue'].append(cue)
                    fin['beta'].append(beta)
                    fin['weight'].append(weight)
                    fin['sim_num'].append(sim_num)
                    fin['exp_cw_dur'].append(cw_dur)
                    fin['exp_cw_spec'].append(cw_spec)
                except Exception as e:
                    print(f'ERROR processing sim_{sim_num}: {type(e).__name__}: {e}')
                    import traceback; traceback.print_exc()
    final_df = pd.DataFrame(fin)
    final_df.to_csv(root+f'dec_cw_results_{name_corpus}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--root", "-r", type=str, default="/fs/clip-projects/realspeech/projects/speech_in_noise/me_final/toy_model/", help="Root directory for data and models")
    # parser.add_argument("--beta", "-b", type=float, default=1.0, help="Weight for the KL divergence term")
    # parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training epochs")
    # parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size for training")
    # parser.add_argument("--cue", "-c", type=str, choices=["duration", "formants", "none"], default="none")
    parser.add_argument("--filt", action='store_true', help="Only ae and eh used in training")


    args = parser.parse_args()
    print(args)
    print("Using CUDA:", torch.cuda.is_available())
    main(args)