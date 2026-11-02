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
# DO ALL WITH LDA, do beta=1,2,4 for dec all Vs
export cue="none" # "formants" or duration or none
export type_input="mel"
export betas=(1.25)
export weight=1.0 #0.999 #0.99 #1.0
export corpus="tim"
export nums=({1..1})
nums_string="${nums[*]}"

# for corpus in ${corpora[*]}
# do
for beta in ${betas[*]}
do
	export resid="$corpus"_"$type_input"_"$weight"_train_"$beta"
	export err=$root/tmp/mdl/err/"$resid"_%A_%a.txt
	export out=$root/tmp/mdl/out/"$resid"_%A_%a.txt
	export wav_path=/fs/clip-realspeech/projects/feats_analysis/vowel_analysis/extracted_wavs/"${corpus^^}"
	sbatch --job-name=$resid --array=0-0 --ntasks=1  --partition=clip --account=clip --qos=medium --gres=gpu:1 \
	--cpus-per-task=8 --time 0-03:00:00 --begin=now -e $err -o $out \
	--wrap "job_nums=($nums_string); job_num=\${job_nums[\$SLURM_ARRAY_TASK_ID]};\
	python $root/scripts/main.py -c $cue --name_corpus $corpus -w $weight -b $beta -t \
	$type_input --sim_num \$job_num --epochs 100 -lr 0.0001 --dec --only_ae_eh" 
done
# done
# only use --only_ae_eh for when filtering dataset to only include 'ae' and 'eh' phones
# --dec use for decision model included

conda deactivate

