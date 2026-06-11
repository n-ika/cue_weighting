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

export resid="lda"
export err=$root/tmp/err/"$resid"_%A_%a.txt
export out=$root/tmp/out/"$resid"_%A_%a.txt

sbatch --job-name=$resid --ntasks=1  --partition=clip --account=clip --qos=medium --gres=gpu:1 \
--cpus-per-task=8 --mem=32G --time 0-06:00:00 --begin=now -e $err -o $out \
--wrap "python $root/scripts/lda.py" 

conda deactivate

#--begin=now+30minutes