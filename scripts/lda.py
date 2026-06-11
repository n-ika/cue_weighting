import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import os
from parselmouth.praat import call
import seaborn as sns
import torch
from vae import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import statsmodels.formula.api as smf
from corpus_data_extraction import *
import torchaudio.functional as F

def get_val_results(val_loader, model, device, mapping, meta_df):
    res = {'z':[], 
        #    'recon':[], 
        #    'mu':[], 
        #    'logvar':[], 
           'vowel':[], 
        #    'orig':[], 
           'f1':[], 'dur':[]
           }
    model.eval()
    with torch.no_grad():
        for data, labels in val_loader:
            label_vowel = [label.split('/')[-1].split('_')[1] for label in labels]
            num_label = [mapping[vowel] for vowel in label_vowel]
            data = data.to(device)
            batch,channels,frequency,time = data.shape
            mu, logvar = model.encode(data)
            z = model.reparameterize(mu, logvar)
            res['z'].append(z)
            # res['mu'].extend(mu)
            # res['logvar'].extend(logvar)
            res['vowel'].extend(num_label)
            f1_vals = meta_df[meta_df['filename'].isin(labels)].f1.values
            dur_vals = meta_df[meta_df['filename'].isin(labels)].dur.values
            res['f1'].extend(f1_vals)   
            res['dur'].extend(dur_vals)

        res['z'] = torch.cat(res['z'], dim=0).cpu().detach().numpy()
        # res['mu'] = torch.cat(res['mu'], dim=0).cpu().detach().numpy()
        # res['logvar'] = torch.cat(res['logvar'], dim=0).cpu().detach().numpy()
        res['vowel'] = torch.tensor(res['vowel'])
        res['f1'] = torch.tensor(res['f1'])
        res['dur'] = torch.tensor(res['dur'])
    return res

def train_lda(res):
    lda_model = LDA()
    lda_model.fit(res['z'], res['vowel'])
    return lda_model


def get_exp_stim(root, wudf, min_scale, max_scale):
    type_in = 'mel'
    n_fft = 512
    n_mels = 80
    step = 128
    sr = 16000
    stims = {}
    min_lens = {1:[], 2:[], 3:[],4:[], 5:[], 6:[], 7:[]}
    for file in os.listdir(root):
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            waveform, sr_orig = torchaudio.load(file_path)
            waveform = F.resample(waveform, orig_freq=sr_orig, new_freq=sr)
            spec = audio_to_spectrogram(waveform, sr, type_in, n_fft, n_mels, hop_length=step)
            spec_step = int(file.split('.')[0].split('_')[1][-1])
            dur_step = int(file.split('.')[0].split('_')[0][-1])
            start = wudf[(wudf.dur_step==dur_step)&(wudf.spec_step==spec_step)].start.values[0]
            end = wudf[(wudf.dur_step==dur_step)&(wudf.spec_step==spec_step)].end.values[0]
            frame_start = int(np.floor(start*sr/step)) 
            frame_end = int(np.ceil(end*sr/step))
            sliced_spec = spec[:, :, frame_start:frame_end]
            stims[file] = sliced_spec
            min_lens[dur_step].append(sliced_spec.shape[2])
    for dur_step in min_lens:
        min_lens[dur_step] = min(min_lens[dur_step])
    stims = {key: vals[:,:, :min_lens[int(key.split('_')[0][-1])]] for key, vals in stims.items()}
    stims = {key: (vals - min_scale) / (max_scale - min_scale + 1e-6) for key, vals in stims.items()}
    return(stims)

def test_lda(model, lda_model, min_scale, max_scale, wudf, device):
    root = "/fs/clip-realspeech/projects/speech_in_noise/me_final/data/orig_test_stimuli/"
    stims = get_exp_stim(root, wudf, min_scale, max_scale)
    vals = stims.values()
    keys = stims.keys()
    labels = torch.tensor([[int(key.split('.')[0].split('_')[0][-1]), int(key.split('.')[0].split('_')[1][-1])] for key in keys], dtype=torch.float32)

    wu_d = {'z':[], 'recon':[], 'mu':[], 'logvar':[], 'orig':[], 'f1':[], 'dur':[],'pred':[],}

    model.eval()
    with torch.no_grad():
        for data, label in zip(vals, labels):
            data = data.unsqueeze(0).to(device)
            batch,channels,frequency,time = data.shape
            mu, logvar = model.encode(data)
            z = model.reparameterize(mu, logvar)
            # x_recon = model.decode(z,batch,time)
            pred = lda_model.predict(z.cpu().detach().numpy())[0]
            wu_d['z'].append(z)
            # wu_d['recon'].append(x_recon)
            wu_d['mu'].append(mu)
            wu_d['logvar'].append(logvar)
            wu_d['f1'].append(label[1])
            wu_d['dur'].append(label[0])
            wu_d['pred'].append(pred)
            # wu_d['orig'].append(data)
        wu_d['z'] = torch.stack(wu_d['z'], dim=0).cpu().detach().numpy()
        # wu_d['recon'] = torch.cat(wu_d['recon'], dim=0).cpu().detach().numpy()
        wu_d['mu'] = torch.cat(wu_d['mu'], dim=0).cpu().detach().numpy()
        wu_d['logvar'] = torch.cat(wu_d['logvar'], dim=0).cpu().detach().numpy()
        wu_d['f1'] = torch.stack(wu_d['f1'], dim=0).cpu().detach().numpy()
        wu_d['dur'] = torch.stack(wu_d['dur'], dim=0).cpu().detach().numpy()
        # wu_d['orig'] = torch.cat(wu_d['orig'], dim=0).cpu().detach().numpy()

        lda_df = pd.DataFrame({
                            # 'z': wu_d['z'],
                            # 'z2': wu_d['z'][:,1],
                            'pred': wu_d['pred'],
                            'f1': wu_d['f1'],
                            'dur': wu_d['dur']
                            })

    return(lda_df)

def get_weights(lda_df):
    lmm = smf.ols(formula = 'pred' +' ~ dur + f1', data=lda_df)
    results = lmm.fit()
    dur_w = np.abs(float(results.params.dur)) 
    spec_w = np.abs(float(results.params.f1))
    summed = dur_w + spec_w
    cw_dur = dur_w/summed
    cw_spec = spec_w/summed
    return cw_dur, cw_spec

# def get_umap(z):
#     data = np.array(z) #[:,0,:] # 1st dim is 0
#     reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
#     embedding = reducer.fit_transform(data)
#     return embedding


name_corpus = 'tim'
type_data = 'filt'
type_input = 'mfcc'
nfft = 512
step = 128
dec = False
if dec:
    dec_id = '_dec'
    print("using decision model")
else:
    dec_id = ''
    print("using bare model")
root = "/fs/clip-projects/realspeech/projects/speech_in_noise/me_final/"
corpus_data_id = f'{name_corpus}_{type_input}_{nfft}_{step}'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_meta = pd.read_csv(root+f'data/meta/{corpus_data_id}.csv')
meta_df = df_meta[df_meta['phone'].isin(['eh', 'ae'])]
wudf = pd.read_csv('/fs/clip-realspeech/projects/speech_in_noise/me_final/wustim_metadata.csv')
wudf['dur'] = wudf.groupby('dur_step')['dur'].transform('mean')
wudf['f1'] = wudf.groupby('spec_step')['f1'].transform('mean')

dataset_dict = torch.load(root+f'data/{corpus_data_id}.pt',weights_only=False)
train_dict = dataset_dict['train']
val_dict = dataset_dict['val']
filenames = set(meta_df['filename'])
train_dict = {k: v for k, v in train_dict.items() if k in filenames}
val_dict = {k: v for k, v in val_dict.items() if k in filenames}
min_scale = dataset_dict['min']
max_scale = dataset_dict['max']
mapping = {'eh':0, 'ae':1}
val_loader = DataLoader(CorpusData(val_dict), batch_size=16, shuffle=False, collate_fn=PadCollate(dim=1))

fin = {
    'cue': [],
    'beta': [],
    'weight': [],
    # 'sim_num': [],
    'cw_dur': [],
    'cw_spec': []
}
for cue in ['none', 'formants', 'duration']:
    for beta in [1e-05]: #, 0.0001, 0.001]:
        for sim_num in range(1,100+1):
            try:
            # beta = 0.001
                if cue == 'none':
                    weight = 1.0
                else:
                    weight = 0.99
                if type_input == 'mfcc':
                    model = VAEmfcc(dec=dec,h_dim=4).to(device)
                else:
                    model = ConvVAE(dec=dec,h_dim=64).to(device)
                model.load_state_dict(torch.load(root+f'models_{type_data}/{name_corpus}_{type_input}_512_128/{beta}_{cue}_{weight}/sim_{sim_num}/ckpt_20.pth',
                                    map_location=device,
                                    weights_only=False))

                res = get_val_results(val_loader, model, device, mapping, meta_df)
                lda_model = train_lda(res)
                lda_df = test_lda(model, lda_model, min_scale, max_scale, wudf, device)
                if lda_df['pred'].unique().ndim == 1:
                    cw_dur, cw_spec = 0.0, 0.0
                else:
                    cw_dur, cw_spec = get_weights(lda_df)
                fin['cue'].append(cue)
                fin['beta'].append(beta)
                fin['weight'].append(weight)
                # fin['sim_num'].append(sim_num)
                fin['cw_dur'].append(cw_dur)
                fin['cw_spec'].append(cw_spec)
                            # except Exception as e:
                            #     print(f"Error processing {cue}, {beta}, sim {sim_num}: {e}")
                print(f"Processed {cue}, {beta}")
            except Exception as e:
                print(f"Error processing {cue}, {beta}: {e}")

final_df = pd.DataFrame(fin)
final_df.to_csv(root+f'cw_results_{type_input}_{type_data}{dec_id}.csv', index=False)

