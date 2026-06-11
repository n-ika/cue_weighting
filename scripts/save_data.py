import torchaudio
import torch
import argparse
import os
from corpus_data_extraction import *

def main(args):
    root_out = args.out_root
    type_in = args.type_in
    n_mels = args.n_mels
    os.makedirs(root_out, exist_ok=True) # .pt files (arrays)
    os.makedirs(f'{root_out}/meta/', exist_ok=True) # .csv files (metadata)
    n_fft = args.n_fft
    hop_length = args.hop_length

    if 'tim' in args.corpora:
        extract_timit(out_dir=root_out,
                    type_in = type_in,
                    n_fft = n_fft,
                    step = hop_length, 
                    corpus_root = '/fs/clip-realspeech/corpora/TIMIT/TIMIT_ORIG/',
                    n_mels = n_mels
                    )
        print('done timit')
    if 'buc' in args.corpora:
        extract_buc(out_dir=root_out, 
                    type_in = type_in, 
                    n_fft = n_fft,
                    step = hop_length,
                    corpus_root='/fs/clip-realspeech/corpora/buckeye_speech_corpus',
                    n_mels = n_mels)
        print('done buckeye')
    if 'wsj' in args.corpora:
        extract_wsj(out_dir=root_out, 
                    type_in = type_in, 
                    n_fft = n_fft,
                    step = hop_length,
                    n_mels = n_mels)
        print('done wsj')
    print("done")
    return()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", 
                         "-c", 
                         nargs='+',
                         type=str,
                         default=['tim', 'buc', 'wsj'],
                         help="Space-separated list of corpora (e.g., -c tim buc wsj)")
    parser.add_argument("--out_root", 
                        "-o", 
                        type=str, 
                        default="/fs/clip-projects/realspeech/projects/speech_in_noise/me_final/data/", 
                        help="Root directory to save dataset")
    parser.add_argument("--n_fft", 
                        "-n", 
                        type=int, 
                        default=512, 
                        help="FFT size for spectrogram extraction")
    parser.add_argument("--hop_length", 
                        "-hl", 
                        type=int, 
                        default=128, 
                        help="Hop length for spectrogram extraction")
    parser.add_argument("--type_in", 
                        "-t", 
                        type=str, 
                        default='mel', 
                        help="Type of input for spectrogram extraction [spectral, mel]")
    parser.add_argument("--n_mels", 
                        "-nm", 
                        type=int, 
                        default=128, 
                        help="Number of mel coefficients for mel-spectrogram extraction")
    args = parser.parse_args()
    print(args)
    main(args)