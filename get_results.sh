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
export type_input="mel"
export beta=1.25 #1e-05
# export nfft=512 #256
# export step=128
export corpus="tim"
export filtparam="filt"
export epoch=100

export resid=$type_input_$beta_$filtparam
export err=$root/tmp/err/"$resid"_%A.txt
export out=$root/tmp/out/"$resid"_%A.txt
export wav_path=/fs/clip-realspeech/projects/feats_analysis/vowel_analysis/extracted_wavs/"${corpus^^}"

sbatch --job-name=$resid --ntasks=1  --partition=clip --account=clip --qos=medium --gres=gpu:1 \
--cpus-per-task=8 --time 0-03:00:00 --begin=now -e $err -o $out \
--wrap "job_nums=($nums_string); job_num=\${job_nums[\$SLURM_ARRAY_TASK_ID]};\
python $root/scripts/get_results.py --name_corpus $corpus -b $beta -t \
$type_input --epoch $epoch --dec" 

#-- filt include to include only ae and eh
conda deactivate

