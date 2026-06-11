import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

def audio_to_spectrogram(audio_file, type_in, n_fft, hop_length=None):
    """
    Convert audio file to spectrogram
    
    Args:
        audio_file: Path to audio file
        n_fft: FFT size
        hop_length: Hop length between frames
        
    Returns:
        (mel-)spectrogram of shape [1, freq_bins, time_frames]
    """
    if hop_length == None:x
        hop_length = n_fft // 4
    waveform, sample_rate = torchaudio.load(audio_file,format='wav')
    
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
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=80,
        mel_scale="htk",
    )

    if type_in == 'spectral':
        spec = spectrogram_transform(waveform)
        out = torch.log2(spec + 1e-9)  # Adding small constant to avoid log(0)
    elif type_in == 'mel':
        mel = mel_spectrogram_transform(waveform)
        out = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(mel)
    return out



# def plot_spectrogram(spec, title=None):
#     """Plot a spectrogram"""
#     plt.figure(figsize=(10, 4))
#     plt.imshow(spec[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
#     plt.colorbar(format='%+1.0f dB')
#     plt.xlabel('Time Frames')
#     plt.ylabel('Frequency Bins')
#     if title:
#         plt.title(title)
#     plt.tight_layout()
#     plt.show()



# class SpectrogramDataset(Dataset):
#     """Dataset for audio spectrograms"""
#     def __init__(self, audio_dir, type_in, n_fft, hop_length, transform):
#         self.audio_dir = audio_dir
#         self.transform = transform
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.audio_files = self.get_all_subdirs(audio_dir)
#         self.type_in = type_in

#     def get_all_subdirs(self,directory):
#         subdirs = []
#         for dirpath, dirnames, _ in os.walk(directory):
#             for dirname in dirnames:
#                 if dirname.lower().endswith('ae') or dirname.lower().endswith('eh'):
#                     for filename in os.listdir(os.path.join(dirpath, dirname)):
#                         if filename.endswith('.wav'):
#                             subdirs.append(os.path.join(dirpath, dirname, filename))
#         return subdirs
        
#     def __len__(self):
#         return len(self.audio_files)
    
#     def __getitem__(self, idx):
#         audio_path = self.audio_files[idx]
#         spec = audio_to_spectrogram(audio_path, type_in=self.type_in, n_fft=self.n_fft, hop_length=self.hop_length)
        
#         # Normalize to [0, 1] range
#         spec_min = spec.min()
#         spec_max = spec.max()
#         spec = (spec - spec_min) / (spec_max - spec_min + 1e-6)
#         if self.transform:
#             spec = self.transform(spec)
#         return audio_path, spec 


def precompute_and_save(audio_dir, save_dir, type_in, n_fft, hop_length):
    os.makedirs(save_dir, exist_ok=True)
    
    # First pass: Extract and find global stats
    all_specs = []
    file_list = []
    
    # Use your existing logic to get files
    temp_ds = SpectrogramDataset(audio_dir, type_in, n_fft, hop_length, transform=None)
    
    global_min = float('inf')
    global_max = float('-inf')

    print("Extracting spectrograms and calculating global stats...")
    for i in range(len(temp_ds)):
        audio_path, spec = temp_ds[i] # This uses your original audio_to_spectrogram
        
        # Update global stats
        global_min = min(global_min, spec.min().item())
        global_max = max(global_max, spec.max().item())
        
        # Save temporary path and spec
        file_name = os.path.basename(audio_path).replace('.wav', '.pt')
        save_path = os.path.join(save_dir, file_name)
        torch.save(spec, save_path)
        file_list.append(save_path)

    # Second pass: Normalize and overwrite
    print(f"Normalizing with Global Min: {global_min}, Max: {global_max}")
    for f_path in file_list:
        spec = torch.load(f_path)
        # Global normalization to [0, 1]
        norm_spec = (spec - global_min) / (global_max - global_min + 1e-6)
        torch.save(norm_spec, f_path)
        
    # Save the stats for later (inference/reconstruction)
    np.save(os.path.join(save_dir, 'stats.npy'), {'min': global_min, 'max': global_max})
    print("Precomputation complete.")


class PrecomputedSpectrogramDataset(Dataset):
    """Dataset for pre-extracted normalized spectrograms"""
    def __init__(self, precomputed_dir, transform=None):
        self.precomputed_dir = precomputed_dir
        self.transform = transform
        # List all .pt files
        self.files = [os.path.join(precomputed_dir, f) for f in os.listdir(precomputed_dir) if f.endswith('.pt')]
        # Load global stats for reconstruction reference
        stats = np.load(os.path.join(precomputed_dir, 'stats.npy'), allow_pickle=True).item()
        self.global_min = stats['min']
        self.global_max = stats['max']

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        spec = torch.load(self.files[idx])
        
        if self.transform:
            spec = self.transform(spec)
            
        return self.files[idx], spec