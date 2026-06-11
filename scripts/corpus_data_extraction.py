import torch
import torchaudio
import torchaudio.transforms as T
import parselmouth
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import buckeye
# from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler
import torchaudio.functional as F

class PadCollate:
    def __init__(self, dim=1):
        self.dim = dim
        
    def __call__(self, batch):
        label, items = zip(*batch)
        max_len = max(map(lambda x: x.shape[self.dim], items))
        max_len = max(max_len, 20)

        padded_batch = []
        for data in items:
            pad_size = list(data.shape)
            pad_size[self.dim] = max_len - data.size(self.dim)
            padded = torch.cat([data, torch.zeros(*pad_size)], dim=self.dim)
            if padded.ndim < 4:
                padded = padded.unsqueeze(0)
            padded_batch.append(padded)
        return torch.stack(padded_batch), label

class CorpusData(Dataset):
    def __init__(self, data_dict):
        self.keys = list(data_dict.keys())
        self.data = data_dict

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        data_id = self.keys[idx]
        sample = self.data[data_id]
        return data_id, sample
    

def get_formants(audio_file_path):
    sound = parselmouth.Sound(audio_file_path)
    formant_object = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=6000)
    return(formant_object)

def get_phon_data(all_formants,sr,start,end,n_formants=2):
    midpoint = ((end + start)/2)/sr
    dur = (end-start)/sr
    formants = {}
    for i in range(1,n_formants+1):
        formants[i] = all_formants.get_value_at_time(i, midpoint)
    return(formants,dur)

# def resample_file(waveform,sr):
#     target_sample_rate = 16000
#     resampler = T.Resample(orig_freq=sr, new_freq=target_sample_rate)
#     resampled_waveform = resampler(waveform)
#     return(resampled_waveform,target_sample_rate)

def audio_to_spectrogram(waveform, sample_rate, type_in, n_fft, n_mels, hop_length=None):
    """
    Convert audio file to spectrogram
    
    Args:
        audio_file: Path to audio file
        n_fft: FFT size
        hop_length: Hop length between frames
        
    Returns:
        (mel-)spectrogram of shape [1, freq_bins, time_frames]
    """
    if hop_length == None:
        hop_length = n_fft // 4
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0 
    )
    
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
        n_mels=n_mels,
        center=True

    )

    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, 
                                                n_mfcc=13, 
                                                dct_type=2, 
                                                norm='ortho', 
                                                log_mels=True,
                                                melkwargs={"n_fft": n_fft, 
                                                           "hop_length": hop_length, 
                                                           "n_mels": n_mels, 
                                                           "center": True}
    )
    if type_in == 'spectral':
        spec = spectrogram_transform(waveform)
        out = torch.log2(spec + 1e-9)  # Adding small constant to avoid log(0)
    elif type_in == 'mel':
        mel = mel_spectrogram_transform(waveform)
        out = torchaudio.transforms.AmplitudeToDB(top_db=90.0)(mel)
        # out = torch.log2(mel + 1e-9)
    elif type_in == 'mfcc':
        out = mfcc_transform(waveform)
    else:
        raise ValueError(f"Unsupported type_in: {type_in}. Choose from 'spectral', 'mel', or 'mfcc'.") 
    return out



def extract_timit(out_dir='/fs/clip-realspeech/projects/speech_in_noise/me_final/data/',
                  type_in = 'mel',
                  n_fft = 512, 
                  corpus_root = '/fs/clip-realspeech/corpora/TIMIT/TIMIT_ORIG/',
                  step = 128,
                  n_mels = 80
                  ):
    maxs = []
    mins = []
    vowels = ["iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", 
              "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h"]

    
    dfs = []
    data_d = {}
    for type_dir in ['train','test']:
        data_d[type_dir] = {}
        corpus_path = corpus_root + type_dir + '/'
        for dialect_dir in os.listdir(corpus_path):
            folder_path = corpus_path+dialect_dir
            if os.path.isdir(folder_path):
                for speaker in os.listdir(folder_path):
                    speaker_path = folder_path + '/' + speaker
                    if os.path.isdir(speaker_path):
                        for file_spk in os.listdir(speaker_path):
                            if file_spk.endswith('.wav'):
                                wav = dialect_dir + '/' + speaker + '/' + file_spk.split('.')[0]
                                phnpath = speaker_path + '/' + file_spk.split('.')[0] + '.phn'
                                df_phn = pd.read_csv(phnpath, delimiter=' ', header=None, names=['start','end','phone'])
                                audio_file_path = speaker_path + '/' + file_spk
                                waveform,sr = torchaudio.load(audio_file_path)
                                all_formants = get_formants(audio_file_path)
                                if sr != 16000:
                                    # waveform, sr, = resample_file(waveform,16000)
                                    waveform, sr = F.resample(waveform, orig_freq=sr, new_freq=16000)
                                    print('wrong SR!', audio_file_path)
                                spec = audio_to_spectrogram(waveform, sr, type_in, n_fft, n_mels, hop_length=step)
                                if type_dir == 'train':
                                    maxs.append(spec.max().item())
                                    mins.append(spec.min().item())
                                # for vowel in ['eh', 'ae']:
                                for vowel in vowels:
                                    df_vowel = df_phn[df_phn.phone==vowel].reset_index()
                                    d_interim = {'start': [], 
                                            'end':[], 
                                            'frame_start':[], 
                                            'frame_end':[],
                                            'dur':[],
                                            'f1':[],
                                            'f2':[],
                                            'filename':[],
                                            'phone':[],
                                            'spk':[],
                                            'data_type':[]
                                            }
                                    for i,start in enumerate(df_vowel['start']):
                                        name_id = wav+'_'+vowel+'_'+str(start)
                                        frame_start = int(np.floor(start/step))
                                        end = df_vowel['end'][i]
                                        frame_end = int(np.ceil(end/step))
                                        d_interim['start'].append(start)
                                        d_interim['end'].append(end)
                                        d_interim['frame_start'].append(frame_start)
                                        d_interim['frame_end'].append(frame_end)
                                        formants,dur = get_phon_data(all_formants,sr,start,end,n_formants=2)
                                        d_interim['dur'].append(dur)
                                        d_interim['f1'].append(formants[1])
                                        d_interim['f2'].append(formants[2])
                                        d_interim['filename'].append(name_id)
                                        d_interim['phone'].append(vowel)
                                        d_interim['spk'].append(speaker)
                                        d_interim['data_type'].append(type_dir)
                                        
                                        data_d[type_dir][name_id] = spec[0,:,frame_start:frame_end]
                                    df=pd.DataFrame(d_interim)
                                    dfs.append(df)

      
    spec_min = min(mins)
    spec_max = max(maxs)
    data_d['train'] = {k:(v - spec_min) / (spec_max - spec_min + 1e-6) for k,v in data_d['train'].items()}
    data_d['test'] = {k:(v - spec_min) / (spec_max - spec_min + 1e-6) for k,v in data_d['test'].items()}
    
    df_fin = pd.concat(dfs).reset_index()
    del df_fin['index']
    df_fin.to_csv(f'{out_dir}meta/tim_{type_in}_{n_fft}_{step}.csv')
    print("Dataframe saved")
    mapping = {label: index for index, label in enumerate(vowels)}

    # X_train = np.array(list(data_d['train'].values()))
    # y_train = np.array(list(data_d['train'].keys()))
    # X_val = np.array(list(data_d['test'].values()))
    # y_val = np.array(list(data_d['test'].keys()))
    
    # numeric_y_train = [mapping[val] for val in y_train]
    # numeric_y_val = [mapping[val] for val in y_val]

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_val_scaled = scaler.transform(X_val)
    # X_train_scaled = torch.from_numpy(X_train_scaled).float()
    # X_val_scaled = torch.from_numpy(X_val_scaled).float()
    # train_dataset = TensorDataset(X_train_scaled, numeric_y_train)
    # val_dataset = TensorDataset(X_val_scaled, numeric_y_val)

    checkpoint = {
        'train': data_d['train'],
        'val': data_d['test'],
        'min': spec_min,
        'max': spec_max,
        'mapping': mapping
    }
    torch.save(checkpoint, f'{out_dir}tim_{type_in}_{n_fft}_{step}.pt')
    # torch.save(data_d, f'{out_dir}tim_{type_in}_{n_fft}_{step}.pt')
    print("Dataset saved")



def extract_buc(out_dir='/fs/clip-realspeech/projects/speech_in_noise/me_final/data/', 
                type_in = 'mel', 
                n_fft = 512,
                corpus_root='/fs/clip-realspeech/corpora/buckeye_speech_corpus',
                step = 128,
                n_mels = 80):
    
    meta_df_buc = pd.read_csv('/fs/clip-realspeech/projects/speech_in_noise/me_final/data/meta/buc_spk2gen.csv', delimiter=' ')

    res = {'phone':[],'start':[],'end':[],'frame_start':[],'frame_end':[],'f1':[],'f2':[],'dur':[]}
    data_d = {}
    dfs = []
    maxs = []
    mins = []
    for spk in meta_df_buc.spk.values:
        print(spk)
        spk_path = f'{corpus_root}/{spk}/'
        for file_spk in os.listdir(spk_path):
            root = spk_path + f'{file_spk}/'
            audio_file_path = root + file_spk + '.wav'
            waveform,sr = torchaudio.load(audio_file_path)
            all_formants = get_formants(audio_file_path)
            if sr != 16000:
                waveform, sr, = resample_file(waveform,sr)
                print('wrong SR!', audio_file_path)
            spec = audio_to_spectrogram(waveform, sr, type_in, n_fft, n_mels, hop_length=step)
            maxs.append(spec.max().item())
            mins.append(spec.min().item())
            track = buckeye.Track(name=file_spk,
                                    words=root+f'{file_spk}.words',
                                    phones=root+f'{file_spk}.phones',
                                    log=root+f'{file_spk}.log',
                                    txt=root+f'{file_spk}.txt',
                                    wav=root+f'{file_spk}.wav'
                                    )
            for phone in track.phones:
                if phone.seg == 'ae' or phone.seg=='eh':
                    start_sample = int(phone.beg*sr)
                    end_sample = int(phone.end*sr)
                    formants,dur = get_phon_data(all_formants,sr,start_sample,end_sample,n_formants=2)
                    res['phone'].append(phone.seg.lower())
                    res['start'].append(phone.beg)
                    res['end'].append(phone.end)
                    res['f1'].append(formants[1])
                    res['f2'].append(formants[2])
                    res['dur'].append(dur)
                    start = int(np.floor(phone.beg*sr/step))
                    end = int(np.ceil(phone.end*sr/step))
                    res['frame_start'].append(start)
                    res['frame_end'].append(end)
                    name_id = file_spk+'_'+phone.seg.lower()+'_'+str(start_sample)
                    data_d[name_id] = spec[0,:,start:end]

            df=pd.DataFrame(res)
            df['filename'] = name_id
            df['spk'] = spk
            df['gender'] = meta_df_buc[meta_df_buc.spk==spk].spk_gender.iloc[0]
            dfs.append(df.reset_index())
    spec_min = min(mins)
    spec_max = max(maxs)
    data_d = {k:(v - spec_min) / (spec_max - spec_min + 1e-6) for k,v in data_d.items()}
    df_fin = pd.concat(dfs).reset_index()
    del df_fin['index']
    df_fin.to_csv(f'{out_dir}meta/buc_{type_in}_{n_fft}_{step}.csv')
    print("Dataframe saved")
    torch.save(data_d, f'{out_dir}buc_{type_in}_{n_fft}_{step}.pt')
    print("Dataset saved")

def extract_wsj(out_dir='/fs/clip-realspeech/projects/speech_in_noise/me_final/data/', 
                type_in = 'mel', 
                n_fft = 512,
                step = 128,
                n_mels = 80):
    
    meta_wsj = pd.read_csv('/fs/clip-realspeech/corpora/spock-format/WSJ/align/alignment.txt', header=None, 
                           names=['id','start','end','null','phone','word'],
                          delimiter=' ')
    del meta_wsj['null']
    wavs_dir = '/fs/clip-realspeech/corpora/spock-format/WSJ/aligned_data/wavs/'
    utt2spk = pd.read_csv('/fs/clip-realspeech/corpora/spock-format/WSJ/data/utt2spk.txt', 
                        names = ['file', 'spk'],
                        delimiter=' ')
    spk2gender = pd.read_csv('/fs/clip-realspeech/corpora/spock-format/WSJ/data/spk2gender_WSJ.txt', 
                            names = ['spk', 'gender'],
                            delimiter=' ')

    res = {'phone':[],'start':[],'end':[],'frame_start':[],
           'frame_end':[],'f1':[],'f2':[],'dur':[],'filename':[],'spk':[],'gender':[]}
    data_d = {}
    mins = []
    maxs = []
    for i,spk_file in enumerate(meta_wsj.id.unique()):
        print(f'done {i}/{meta_wsj.id.unique().shape[0]}')
        audio_file_path = f'{wavs_dir}{spk_file}.wav'
        spk_df = meta_wsj[(meta_wsj['id']==spk_file)&((meta_wsj['phone']=='AE')|(meta_wsj['phone']=='EH'))]
        spk = utt2spk[utt2spk['file']==spk_file].spk.values[0]
        waveform,sr=torchaudio.load(audio_file_path)
        all_formants = get_formants(audio_file_path)
        if sr != 16000:
            waveform, sr, = resample_file(waveform,sr)
            print('wrong SR!', audio_file_path)
        spec = audio_to_spectrogram(waveform, sr, type_in, n_fft, n_mels, hop_length=step)
        maxs.append(spec.max().item())
        mins.append(spec.min().item())
        for row in spk_df.itertuples():
            start_sample = int(row.start*sr)
            end_sample = int(row.end*sr)
            formants,dur = get_phon_data(all_formants,sr,start_sample,end_sample,n_formants=2)
            res['phone'].append(row.phone.lower())
            res['start'].append(row.start)
            res['end'].append(row.end)
            res['f1'].append(formants[1])
            res['f2'].append(formants[2])
            res['dur'].append(dur)
            start = int(np.floor(row.start*sr/step))
            end = int(np.ceil(row.end*sr/step))
            res['frame_start'].append(start)
            res['frame_end'].append(end)
            name_id = spk_file+'_'+row.phone.lower()+'_'+str(start_sample)
            res['filename'].append(name_id)
            res['spk'].append(spk)
            res['gender'].append(spk2gender[spk2gender['spk']==spk].gender.values[0])
            data_d[name_id] = spec[0,:,start:end]

        df_fin=pd.DataFrame(res)
        df_fin.to_csv(f'{out_dir}meta/wsj_{type_in}_{n_fft}_{step}.csv')
        print("Dataframe saved")
        torch.save(data_d, f'{out_dir}wsj_{type_in}_{n_fft}_{step}.pt')
        print("Dataset saved")
        print(f"Completed {i/meta_wsj.id.unique().shape[0]}%")
    spec_min = min(mins)
    spec_max = max(maxs)
    data_d = {k:(v - spec_min) / (spec_max - spec_min + 1e-6) for k,v in data_d.items()}
    df_fin=pd.DataFrame(res)
    df_fin.to_csv(f'{out_dir}meta/wsj_{type_in}_{n_fft}_{step}.csv')
    print("FINAL Dataframe saved")
    torch.save(data_d, f'{out_dir}wsj_{type_in}_{n_fft}_{step}.pt')
    print("FINAL Dataset saved")
    print(f"Completed all data extraction for WSJ")
