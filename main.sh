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
# export corpora=("tim") # "wsj" "buc")
export cue="none" # "formants" or duration or none
export type_input="mfcc"
export betas=(0.00001) # 0.0001 0.001)
# export betas=(0.00001)
export weight=1.0
export ID="train"
export nfft=512 #256
export step=128
export corpus="tim"
export nums=({1..100})
nums_string="${nums[*]}"

# for corpus in ${corpora[*]}
# do
for beta in ${betas[*]}
do
	export resid="$corpus"_train_"$beta"_"$type_input"_"$weight"
	export err=$root/tmp/err/"$resid"_%A_%a.txt
	export out=$root/tmp/out/"$resid"_%A_%a.txt
	export wav_path=/fs/clip-realspeech/projects/feats_analysis/vowel_analysis/extracted_wavs/"${corpus^^}"
	sbatch --job-name=$resid --array=0-99 --ntasks=1  --partition=clip --account=clip --qos=medium --gres=gpu:1 \
	--cpus-per-task=8 --time 0-03:00:00 --begin=now -e $err -o $out \
	--wrap "job_nums=($nums_string); job_num=\${job_nums[\$SLURM_ARRAY_TASK_ID]};\
	python $root/scripts/main.py -c $cue --name_corpus $corpus -w $weight -b $beta -t \
	$type_input --nfft $nfft --step $step --sim_num \$job_num --epochs 30 --only_ae_eh" 
done
# done
# only use --only_ae_eh for when filtering dataset to only include 'ae' and 'eh' phones

conda deactivate

