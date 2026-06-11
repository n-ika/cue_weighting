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




def get_val_results(val_loader, model, device):
    res = {'z':[], 'recon':[], 'mu':[], 'logvar':[], 'vowel':[], 'orig':[], 'f1':[], 'dur':[], 'dec':[]}
    for data, labels in val_loader:
        data = data.to(device)
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
        decisions = model.decide(z) #FIXME
        x_recon = model.decode(z)
        res['z'].append(z)
        res['recon'].append(x_recon)
        res['mu'].append(mu)
        res['logvar'].append(logvar)
        res['vowel'].append(labels[:,0].cpu().detach())
        res['f1'].append(labels[:,1].cpu().detach())
        res['dur'].append(labels[:,2].cpu().detach())
        res['dec'].append(decisions.cpu().detach()) #FIXME
        res['orig'].append(data.cpu().detach())
    res['z'] = torch.cat(res['z'], dim=0).cpu().detach().numpy()
    res['recon'] = torch.cat(res['recon'], dim=0).cpu().detach().numpy()
    res['mu'] = torch.cat(res['mu'], dim=0).cpu().detach().numpy()
    res['logvar'] = torch.cat(res['logvar'], dim=0).cpu().detach().numpy()
    res['vowel'] = torch.cat(res['vowel'], dim=0).cpu().detach().numpy()
    res['f1'] = torch.cat(res['f1'], dim=0).cpu().detach().numpy()
    res['dur'] = torch.cat(res['dur'], dim=0).cpu().detach().numpy()
    res['orig'] = torch.cat(res['orig'], dim=0).cpu().detach().numpy()
    res['dec'] = torch.stack(res['dec'], dim=0).cpu().detach().numpy()
    return res


def test_model(model, wudf, scaler, device):
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
        dec = torch.sigmoid(model.decide(z))
        x_recon = model.decode(z)
        wu_d['z'].append(z.unsqueeze(0))
        wu_d['recon'].append(x_recon)
        wu_d['mu'].append(mu)
        wu_d['logvar'].append(logvar)
        wu_d['f1'].append(label[0])
        wu_d['dur'].append(label[1])
        wu_d['dec'].extend(dec.cpu().detach().numpy())
        wu_d['orig'].append(data)
    wu_d['z'] = torch.cat(wu_d['z'], dim=0).cpu().detach().numpy()
    wu_d['recon'] = torch.cat(wu_d['recon'], dim=0).cpu().detach().numpy()
    wu_d['mu'] = torch.cat(wu_d['mu'], dim=0).cpu().detach().numpy()
    wu_d['logvar'] = torch.cat(wu_d['logvar'], dim=0).cpu().detach().numpy()
    wu_d['f1'] = torch.stack(wu_d['f1'], dim=0).cpu().detach().numpy()
    wu_d['dur'] = torch.stack(wu_d['dur'], dim=0).cpu().detach().numpy()
    wu_d['orig'] = torch.cat(wu_d['orig'], dim=0).cpu().detach().numpy()
    wu_d['dec'] = torch.tensor(wu_d['dec'], dtype=torch.float32).cpu().detach().numpy()
    # vals = wudf[['f1', 'dur']].to_numpy()
    # vals = (vals - vals.mean(axis=0)) / vals.std(axis=0)
    # vals = torch.tensor(scaler.transform(vals), dtype=torch.float32)
    # labels = torch.tensor(wudf[['spec_step', 'dur_step']].to_numpy(), dtype=torch.float32)
    # wu_d = {'z':[], 
    #         # 'recon':[], 
    #         # 'mu':[], 
    #         # 'logvar':[], 
    #         # 'orig':[], 
    #         'f1':[], 
    #         'dur':[],
    #         'dec':[]}
    # for data, label in zip(vals, labels):
    #     data = data.to(device)
    #     mu, logvar = model.encode(data)
    #     z = model.reparameterize(mu, logvar)
    #     print(z.shape)
    #     dec = torch.sigmoid(model.decide(z)) #FIXME
    #     # x_recon = model.decode(z)
    #     wu_d['z'].append(z.unsqueeze(0))
    #     # wu_d['recon'].append(x_recon)
    #     # wu_d['mu'].append(mu)
    #     # wu_d['logvar'].append(logvar)
    #     wu_d['f1'].append(label[0])
    #     wu_d['dur'].append(label[1])
    #     # wu_d['orig'].append(data)
    #     wu_d['dec'].extend(dec.cpu().detach().numpy()) #FIXME
    # wu_d['z'] = torch.cat(wu_d['z'], dim=0).cpu().detach().numpy()
    # # wu_d['recon'] = torch.cat(wu_d['recon'], dim=0).cpu().detach().numpy()
    # # wu_d['mu'] = torch.cat(wu_d['mu'], dim=0).cpu().detach().numpy()
    # # wu_d['logvar'] = torch.cat(wu_d['logvar'], dim=0).cpu().detach().numpy()
    # wu_d['f1'] = torch.stack(wu_d['f1'], dim=0).cpu().detach().numpy()
    # wu_d['dur'] = torch.stack(wu_d['dur'], dim=0).cpu().detach().numpy()
    # # wu_d['orig'] = torch.cat(wu_d['orig'], dim=0).cpu().detach().numpy()
    # wu_d['dec'] = torch.tensor(wu_d['dec'], dtype=torch.float32).cpu().detach().numpy()
    df = pd.DataFrame({
                        'z1': wu_d['z'][:,0],
                        'z2': wu_d['z'][:,1],
                        'f1': wu_d['f1'],
                        'dur': wu_d['dur'],
                        'dec': wu_d['dec'], #FIXME
                        })
    print(df.head())
    df['acc'] = (df['dec'] > 0.5).astype(int) #FIXME
    print(df.head())
    return(df)



def get_weights(dec_df):
    lmm = smf.ols(formula = 'dec' +' ~ dur + f1', data=dec_df)
    results = lmm.fit()
    dur_w = np.abs(float(results.params.dur)) 
    spec_w = np.abs(float(results.params.f1))
    summed = dur_w + spec_w
    cw_dur = dur_w/summed
    cw_spec = spec_w/summed
    return cw_dur, cw_spec


root = '/fs/clip-realspeech/projects/speech_in_noise/me_final/toy_model/'
name_corpus = 'timit'
batch_size = 16
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loaded_data = torch.load(root+f'data/{name_corpus}.pt')
loaded_data = torch.load(root+f'data/timit.pt')
val_dataset = loaded_data['val_dataset']
scaler = loaded_data['scaler']
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

wudf = pd.read_csv('/fs/clip-realspeech/projects/speech_in_noise/me_final/wustim_metadata.csv')
wudf['dur'] = wudf.groupby('dur_step')['dur'].transform('mean')
wudf['f1'] = wudf.groupby('spec_step')['f1'].transform('mean')


fin = {
    'cue': [],
    'beta': [],
    'weight': [],
    'sim_num': [],
    'cw_dur': [],
    'cw_spec': []
}
for cue in ['none', 'f1', 'dur']:
    for beta in [0.0039]: #, 0.0078, 0.0156]: #, 0.0321, 0.0625, 0.125, 0.25, 0.5, 1.0]:
        for sim_num in range(1,100+1):
            try:
                weight = 1.0 if cue == 'none' else 0.999
                model = LinearVAE(N=1).to(device)
                model.load_state_dict(torch.load(root+f'models_dec/{name_corpus}/{cue}_{float(beta)}_{weight}/sim{sim_num}/ckpt_20.pth',
                                    map_location=device,
                                    weights_only=False))
                print("Loaded model for cue: {}, beta: {}, sim_num: {}".format(cue, beta, sim_num))
                # res = get_val_results(val_loader, model, device)
                dec_df = test_model(model, wudf, scaler, device)
                print("Tested model")
                cw_dur, cw_spec = get_weights(dec_df)
                print("Extracted weights: cw_dur: {:.4f}, cw_spec: {:.4f}".format(cw_dur, cw_spec))
                fin['cue'].append(cue)
                fin['beta'].append(beta)
                fin['weight'].append(weight)
                fin['sim_num'].append(sim_num)
                fin['cw_dur'].append(cw_dur)
                fin['cw_spec'].append(cw_spec)
            except Exception as e:
                print(f"Error processing {cue}, {beta}, sim {sim_num}: {e}")
final_df = pd.DataFrame(fin)
final_df.to_csv(root+f'dec_cw_results_{name_corpus}.csv', index=False)