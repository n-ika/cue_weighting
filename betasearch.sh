#!/bin/bash

# env setting (can be done only once)
export root=/fs/clip-projects/realspeech/projects/speech_in_noise/me_final
export PATH=/fs/clip-realspeech/projects/speech_in_noise/miniconda/bin:$PATH
source /fs/clip-realspeech/projects/speech_in_noise/miniconda/bin/activate
conda activate /fs/clip-realspeech/projects/speech_in_noise/miniconda/envs/cue_env

# program launch
# export type_input="mel"
# export beta=0.00001 #1e-05
# export nfft=512 #256
# export step=128
# export corpus="tim"

export resid="beta"
export err=$root/tmp/err/"$resid"_%A.txt
export out=$root/tmp/out/"$resid"_%A.txt
export wav_path=/fs/clip-realspeech/projects/feats_analysis/vowel_analysis/extracted_wavs/"${corpus^^}"

sbatch --job-name=$resid --ntasks=1  --partition=clip --account=clip --qos=medium --gres=gpu:1 \
--cpus-per-task=8 --time 0-03:00:00 --begin=now -e $err -o $out \
--wrap "python $root/scripts/betasearch.py" 

conda deactivate

