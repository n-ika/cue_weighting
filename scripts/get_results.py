import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import torch
from vae import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import statsmodels.formula.api as smf
from corpus_data_extraction import *
import torchaudio.functional as F
import argparse



def get_val_results(type_model, val_loader, model, device, mapping, df_meta):
    df = df_meta.set_index('filename')
    res = {'mu':[], 
           'vowel':[], 
           'f1':[],
           'f2':[], 
           'duration':[]
           }
    if type_model == 'dec':
        res['pred']=[]
    model.eval()
    with torch.no_grad():
        for data, labels in val_loader:
            label_vowel = [label.split('/')[-1].split('_')[1] for label in labels]
            num_label = [mapping[vowel] for vowel in label_vowel]
            data = data.to(device)
            recon, mu, logvar, dec = model(data)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            z = torch.flatten(z, start_dim=1)
            res['mu'].append(mu.cpu().detach())
            if type_model == 'dec':
                decision = torch.sigmoid(dec)
                if decision.ndim == 2 and decision.shape[1] == 20:
                    decision = torch.argmax(decision, dim=1)
                res['pred'].extend(decision.cpu().detach())
            res['vowel'].extend(num_label)
            f1_vals = df.loc[list(labels), 'f1_norm'].values
            f2_vals = df.loc[list(labels), 'f2_norm'].values
            dur_vals = df.loc[list(labels), 'dur_norm'].values
            res['f1'].extend(f1_vals)   
            res['f2'].extend(f2_vals)   
            res['duration'].extend(dur_vals)

        # res['mu'] = torch.cat(res['mu'], dim=0).cpu().detach().numpy()
        print(res['mu'][0].shape)
        if mu.ndim == 3:
            res['mu'] = np.concatenate(res['mu'], axis=0).reshape(-1,mu.shape[1]*mu.shape[2])
        elif mu.ndim == 4:
            res['mu'] = np.concatenate(res['mu'], axis=0).reshape(-1,mu.shape[1]*mu.shape[2]*mu.shape[3])
        res['vowel'] = np.array(res['vowel'])
        res['f1'] = np.array(res['f1'])
        res['f2'] = np.array(res['f2'])
        res['duration'] = np.array(res['duration'])
        if type_model == 'dec':
            res['pred'] = np.array(res['pred']).flatten()
            if dec.shape[1] != 20:
                res['pred'] = np.where(res['pred'] > 0.5, 1, 0)
    return res


def train_lda(res):
    lda_model = LDA()
    lda_model.fit(res['mu'], res['vowel'])
    return lda_model



def get_art_stim(min_scale, max_scale):
    origfile_path='/fs/clip-realspeech/projects/speech_in_noise/me_final/data/orig_test_stimuli/'
    folder_path = '/fs/clip-realspeech/projects/speech_in_noise/SSL-SI-tool/stimfeats/'
    step = 0.01
    min_lens = {1:[], 2:[], 3:[],4:[], 5:[], 6:[], 7:[]}

    res = {'start': [], 
            'end':[], 
            'frame_start':[], 
            'frame_end':[],
            'dur_step':[],
            'spec_step':[],
            'filename':[],
            }
    stims = {}
    for dir_file in os.listdir(folder_path):
        if dir_file.endswith('.npy'):
            npf = np.load(folder_path+dir_file)
            filename = '_'.join(dir_file.split('_')[-3:-1])
            wav = filename + '.wav'
            tg_path = origfile_path + filename + '.TextGrid'
            dur_step = int(dir_file.split('_')[-3][-1])
            spec_step = int(dir_file.split('_')[-2][-1])
            if os.path.exists(tg_path):
                sound = parselmouth.Sound(os.path.join(origfile_path, wav))
                tg = parselmouth.read(tg_path)
                tier_number = 1 
                num_intervals = parselmouth.praat.call(tg, "Get number of intervals", tier_number)
                for i in range(1, num_intervals + 1):
                    label = parselmouth.praat.call(tg, "Get label of interval", tier_number, i)
                    start = parselmouth.praat.call(tg, "Get start time of interval", tier_number, i)
                    end = parselmouth.praat.call(tg, "Get end time of interval", tier_number, i)
                    if label: # Skip empty intervals
                        print(f"File: {wav} | Label: {label} | {start:.3f} - {end:.3f}")
                        res['dur_step'].append(dur_step)
                        res['spec_step'].append(spec_step)
                        res['start'].append(start)
                        res['end'].append(end)
                        frame_start = int(np.floor(start/step))
                        frame_end = int(np.ceil(end/step))
                        res['frame_start'].append(frame_start)
                        res['frame_end'].append(frame_end)
                        res['filename'].append(filename)
                        sliced_spec = torch.tensor(npf[:, frame_start:frame_end], dtype=torch.float32)
                        stims[filename] = sliced_spec
                        min_lens[dur_step].append(sliced_spec.shape[-1])
    for dur_step in min_lens:
        min_lens[dur_step] = min(min_lens[dur_step])
    stims = {key: vals[:, :(min_lens[int(key.split('_')[0][-1])])] for key, vals in stims.items()}
    stims = {key: (vals - min_scale) / (max_scale - min_scale + 1e-6) for key, vals in stims.items()}
    return(stims)

######
def get_exp_stim(root, wudf, min_scale, max_scale, type_in):
    sr = 16000
    if type_in == 'mfcc' or type_in == 'mel':
        n_fft = 512
        n_mels = 80
        step = 128
    else:
        n_fft = 320
        step = 160        
    stims = {}
    min_lens = {1:[], 2:[], 3:[],4:[], 5:[], 6:[], 7:[]}
    for file in os.listdir(root):
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            wav, sr_ = torchaudio.load(file_path)
            waveform = F.resample(wav, orig_freq=sr_, new_freq=sr)
            spec = audio_to_spectrogram(waveform, sr, type_in, n_fft, n_mels, hop_length=step)
            spec_step = int(file.split('.')[0].split('_')[1][-1])
            dur_step = int(file.split('.')[0].split('_')[0][-1])
            start = wudf[(wudf.dur_step==dur_step)&(wudf.spec_step==spec_step)].start.values[0]
            end = wudf[(wudf.dur_step==dur_step)&(wudf.spec_step==spec_step)].end.values[0]
            frame_start = int(np.floor(start*sr/step)) 
            frame_end = int(np.ceil(end*sr/step))
            sliced_spec = spec[:, :, frame_start:frame_end]
            stims[file] = sliced_spec
            min_lens[dur_step].append(sliced_spec.shape[-1])
    for dur_step in min_lens:
        min_lens[dur_step] = min(min_lens[dur_step])
    stims = {key: vals[:, :(min_lens[int(key.split('_')[0][-1])])] for key, vals in stims.items()}
    stims = {key: (vals - min_scale) / (max_scale - min_scale + 1e-6) for key, vals in stims.items()}
    return(stims)

def test_model(model, min_scale, max_scale, max_len, wudf, device, type_in, lda_model=None):
    root = "/fs/clip-realspeech/projects/speech_in_noise/me_final/data/orig_test_stimuli/"
    if type_in == 'art':
        stims = get_art_stim(min_scale, max_scale)
    else:
        stims = get_exp_stim(root, wudf, min_scale, max_scale, type_in)
    stims_loader = DataLoader(
                        CorpusData(stims),
                        batch_size=4,
                        shuffle=True,
                        collate_fn=PadCollate(dim=-1,type_input=type_in,max_len=max_len)
                    )

    wu_d = {'mu':[], 'pred':[], 'z':[], 'logvar':[], 'formant':[], 'duration':[],
            }

    model.eval()
    with torch.no_grad():
        for data, labels in stims_loader:
            f_step = torch.tensor([int(label.split('.')[0].split('_')[1][-1]) for label in labels])
            dur_step = torch.tensor([int(label.split('.')[0].split('_')[0][-1]) for label in labels])
            # batch,channels,frequency,time = data.shape
            data = data.to(device)
            recon, mu, logvar, dec = model(data)
            if lda_model == None:
                std = torch.exp(0.5 * logvar)
                z = mu + torch.randn_like(std) * std 
                z = torch.flatten(z, start_dim=1)
                decision = torch.sigmoid(dec)
                if decision.shape[1] == 20:
                    pred = torch.argmax(decision, dim=1).cpu().detach().numpy()
                else:
                    pred = decision[:, 0].cpu().detach().numpy()
            else:
                pred = lda_model.predict(z.cpu().detach().numpy())[0]
            wu_d['pred'].extend(pred)
            # wu_d['mu'].append(mu.cpu().detach().numpy())
            # wu_d['z'].append(z.cpu().detach().numpy())
            # wu_d['logvar'].append(logvar.cpu().detach().numpy())
            wu_d['formant'].append(f_step)
            wu_d['duration'].append(dur_step)
        # wu_d['mu'] = np.array(wu_d['mu'])
        # wu_d['logvar'] = np.array(wu_d['logvar']).flatten()
        wu_d['formant'] = np.array(wu_d['formant'])
        wu_d['duration'] = np.array(wu_d['duration'])
        if dec.shape[1] == 20:
            wu_d['pred'] = np.array(wu_d['pred']).flatten()
        else:
            wu_d['pred'] = np.where(np.array(wu_d['pred']) > 0.5, 1, 0)

        dec_df = pd.DataFrame({
                            'pred': wu_d['pred'],
                            'formant': wu_d['formant'],
                            'duration': wu_d['duration']
                            })

    return(dec_df)

def get_weights(dec_df, col='pred'):
    lmm = smf.ols(formula = col +' ~ duration + formant', data=dec_df)
    results = lmm.fit()
    dur_w = np.abs(float(results.params.duration)) 
    spec_w = np.abs(float(results.params.formant))
    summed = dur_w + spec_w
    if summed > 0:
        cw_dur = dur_w/summed
        cw_spec = spec_w/summed
    else:
        print(summed,dur_w,spec_w)
    return cw_dur, cw_spec

def minmax_data(df_meta):
    f_min = df_meta['f1_norm'].min()
    f_max = df_meta['f1_norm'].max()
    df_meta['f1_norm'] = (df_meta['f1_norm'] - f_min) / (f_max - f_min)
    f_min = df_meta['f2_norm'].min()
    f_max = df_meta['f2_norm'].max()
    df_meta['f2_norm'] = (df_meta['f2_norm'] - f_min) / (f_max - f_min)
    d_min = df_meta['dur_norm'].min()
    d_max = df_meta['dur_norm'].max()
    df_meta['dur_norm'] = (df_meta['dur_norm'] - d_min) / (d_max - d_min)
    return(df_meta)

def test_lda_real(lda_model, res):
    d = {'vowel':[], 'pred':[], 'f1':[], 'f2':[], 'duration':[]}
    for i in range(len(res['mu'])):
        mu = res['mu'][i]
        pred = lda_model.predict(mu.reshape(1, -1))
        d['f1'].append(res['f1'][i])
        d['f2'].append(res['f2'][i])
        d['duration'].append(res['duration'][i])
        d['pred'].append(pred[0])
        d['vowel'].append(res['vowel'][i].item())
    d['f1'] = np.array(d['f1'])
    d['f2'] = np.array(d['f2'])
    d['duration'] = np.array(d['duration'])
    lda_df = pd.DataFrame({
                        'pred': d['pred'],
                        'f1': d['f1'],
                        'f2': d['f2'],
                        'duration': d['duration'],
                        'vowel': d['vowel']
                        })
    return(lda_df)

def main(args):
    root = args.root
    dec = args.dec
    type_input = args.type_input
    filt = args.filt
    batch_size = args.batch_size
    beta = args.beta
    name_corpus = args.name_corpus
    epoch = args.epoch
    # f_choice = args.fchoice
    cue = 'none'

    os.makedirs(root+'outputs', exist_ok=True)
    if type_input == 'art':
        nfft = 320
        step = 160
        max_len = 42
        n_channels = 6
    else:
        n_channels = 13
        nfft = 512
        step = 128

    if dec:
        type_model = 'dec'
        decid = '_dec'
    else:
        type_model = 'lda'
        decid = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    wudf = pd.read_csv('/fs/clip-realspeech/projects/speech_in_noise/me_final/data/meta/wustim_metadata.csv')
    wudf['duration'] = wudf.groupby('dur_step')['dur'].transform('mean')
    wudf['formant'] = wudf.groupby('spec_step')['f1'].transform('mean') 
    corpus_data_id = f'{name_corpus}_{type_input}_{nfft}_{step}'
    df_meta = pd.read_csv(root+f'data/meta/{corpus_data_id}.csv')
    df_meta['f1_norm'] = df_meta.groupby('spk')['f1'].transform(lambda x: (x - x.mean()) / x.std())
    df_meta['f2_norm'] = df_meta.groupby('spk')['f2'].transform(lambda x: (x - x.mean()) / x.std())
    df_meta['dur_norm'] = df_meta.groupby('spk')['dur'].transform(lambda x: (x - x.mean()) / x.std())
    df_meta = minmax_data(df_meta)
    dataset_dict = torch.load(root+f'data/{corpus_data_id}.pt',weights_only=False)
    train_dict = dataset_dict['train']
    val_dict = dataset_dict['val']

    if filt == True:
        df_meta = df_meta[df_meta['phone'].isin(['eh', 'ae'])]
        filenames = set(df_meta['filename'])
        train_dict = {k: v for k, v in train_dict.items() if k in filenames}
        val_dict = {k: v for k, v in val_dict.items() if k in filenames}
        mapping = {'eh':0, 'ae':1}
        filt_id = 'filt'
        max_len = 56
        N = 1
        if type_input == 'art':
            max_len = 42
    else:
        filt_id = 'all'
        mapping = dataset_dict['mapping']
        max_len = 64
        N = 20
        if type_input == 'art':
            max_len = 49

    val_loader = DataLoader(CorpusData(val_dict), batch_size=batch_size, shuffle=False, collate_fn=PadCollate(dim=1,max_len=max_len,type_input=type_input))
    min_scale = dataset_dict['min']
    max_scale = dataset_dict['max']

    for f_choice in [1,2]:
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
        if cue == 'none':
            weight = 1.0
        else:
            weight = 0.99
        for sim_num in range(1,100+1):
            # try:
            print(f'processing models_{filt_id}{decid}/{name_corpus}_{type_input}_{nfft}_{step}/{beta}_{cue}_{weight}/sim_{sim_num}/ckpt_{epoch}.pth')
            if type_input == 'mel':
                model = MelVAE(dec=args.dec, max_len=max_len, N=N).to(device)
            else:
                model = Conv1dVAE(max_len=max_len, n_channels=n_channels, dec=args.dec, N=N)
            model.load_state_dict(torch.load(root+f'models_{filt_id}{decid}/{name_corpus}_{type_input}_{nfft}_{step}/{beta}_{cue}_{weight}/sim_{sim_num}/ckpt_{epoch}.pth',
                                map_location=device,
                                weights_only=False))
            model.to(device)
            model.eval()
            print('model loaded')
            res = get_val_results(type_model, val_loader, model, device, mapping, df_meta)
            
            if dec==False:
                lda_model = train_lda(res)
                dec_real_df = test_lda_real(lda_model, res)
            else:
                lda_model = None
                dec_real_df = pd.DataFrame({'pred':res['pred'],
                                            'vowel':res['vowel'],
                                            'f1':res['f1'],
                                            'f2':res['f2'],
                                            'duration':res['duration'] 
                                    })
            if dec_real_df['pred'].nunique() == 1:
                real_cw_dur, real_cw_spec = 0.0, 0.0
            else:
                dec_real_df['formant'] = dec_real_df[f'f{f_choice}'].copy()
                real_cw_dur, real_cw_spec = get_weights(dec_real_df)
            dec_df = test_model(model, min_scale, max_scale, max_len, wudf, device, type_input, lda_model)
            # exp_cw_dur, exp_cw_spec = get_weights(dec_df, 'pred')
            if dec_df.dec.nunique() > 1:
                exp_cw_dur, exp_cw_spec = get_weights(dec_df, 'pred')
            else:
                exp_cw_dur, exp_cw_spec = 0,0
        
            fin['cue'].append(cue)
            fin['beta'].append(beta)
            fin['weight'].append(weight)
            fin['sim_num'].append(sim_num)
            fin['exp_cw_dur'].append(exp_cw_dur)
            fin['exp_cw_spec'].append(exp_cw_spec)
            fin['real_cw_dur'].append(real_cw_dur)
            fin['real_cw_spec'].append(real_cw_spec)
            print('done processing')
            # except FileNotFoundError:
            #     print(f'ERROR, did not find: {root}models_{filt_id}{decid}/{name_corpus}_{type_input}_{nfft}_{step}/{beta}_{cue}_{weight}/sim_{sim_num}/ckpt_{epoch}.pth')
            # except Exception as e:
            #     print(f'ERROR processing sim_{sim_num}: {type(e).__name__}: {e}')
            #     import traceback; traceback.print_exc()

        final_df = pd.DataFrame(fin)
        final_df.to_csv(root+f'outputs/cw_results_{type_input}_{filt_id}{decid}_{cue}_f{f_choice}.csv', index=False)
        print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="/fs/clip-projects/realspeech/projects/speech_in_noise/me_final/", help="Root directory for data and models")
    parser.add_argument("--name_corpus", "-n", type=str, default="tim", help="Corpus name for data loading")  
    parser.add_argument("--beta", "-b", type=float, default=0.00001, help="Weight for the KL divergence term")
    parser.add_argument("--epoch", "-e", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("--type_input", "-t", type=str, default='mel', help="Type of input for spectrogram extraction [spectral, mel, mfcc, art]")
    parser.add_argument("--filt", action='store_true', help="Boolean. Whether to filter dataset to only include 'ae' and 'eh' phones")
    parser.add_argument("--dec", action='store_true', help="Boolean. Whether the decision model is present")
    # parser.add_argument("--fchoice", "-f", type=int, default=1, help="Which formant to use as cue weight, 1 or 2")

    args = parser.parse_args()
    print(args)
    print("Using CUDA:", torch.cuda.is_available())
    main(args)
