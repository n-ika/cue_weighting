#!/bin/bash

# env setting (can be done only once)
export root=/fs/clip-projects/realspeech/projects/speech_in_noise/me_final
export PATH=/fs/clip-realspeech/projects/speech_in_noise/miniconda/bin:$PATH
source /fs/clip-realspeech/projects/speech_in_noise/miniconda/bin/activate
conda activate /fs/clip-realspeech/projects/speech_in_noise/miniconda/envs/cue_env
module load cuda
module load gcc

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# program launch
export type_input="mfcc"
export nfft=512
export step=128
export ID="extract"
export corpora='tim'
export nmels=80


export resid=ex_"$corpora"_"$type_input"_"$nfft"_"$step"
export err=$root/tmp/err/"$resid".txt
export out=$root/tmp/out/"$resid".txt
export wav_path=/fs/clip-realspeech/projects/feats_analysis/vowel_analysis/extracted_wavs/

sbatch --job-name=$resid --ntasks=1  --partition=clip --account=clip --qos=medium --gres=gpu:1 --cpus-per-task=8 \
		--time 0-00:30:00 --begin=now -e $err -o $out --wrap \
		"python $root/scripts/save_data.py --n_fft $nfft --type_in $type_input --corpora $corpora --hop_length $step --n_mels $nmels" 


conda deactivate

