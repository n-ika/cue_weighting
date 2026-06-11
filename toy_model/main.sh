#!/bin/bash

# env setting (can be done only once)
export root=/fs/clip-projects/realspeech/projects/speech_in_noise/me_final/toy_model
export PATH=/fs/clip-realspeech/projects/speech_in_noise/miniconda/bin:$PATH
source /fs/clip-realspeech/projects/speech_in_noise/miniconda/bin/activate
conda activate /fs/clip-realspeech/projects/speech_in_noise/miniconda/envs/cue_env
module load cuda
module load gcc

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# program launch
export corpus="timit" # "wsj" "buc")
export cue="none" # "f1" or dur or none
# export betas=(0.0039 0.0078 0.0156 0.0321 0.0625 0.125 0.25 0.5 1.0)
export betas=(0.0039) # 0.0078 0.0156)

# export beta=0.0321
export weight=1.0 #0.999 
export ID="train"
export nums=({1..100})
# export num=${nums[$SLURM_ARRAY_TASK_ID]}

nums_string="${nums[*]}"
# for corpus in ${corpora[*]}
# do
for beta in ${betas[*]}
do
export resid="${corpus}_train_${beta}_${type_input}_${weight}_sim${num}"
export err=$root/tmp/err/"$resid"_%A_%a.txt
export out=$root/tmp/out/"$resid"_%A_%a.txt
# exec > "$root/tmp/out/${resid}_%A_%a.txt" 2> "$root/tmp/err/${resid}_%A_%a.txt"

sbatch --job-name=$resid --array=0-99 --ntasks=1  --partition=clip --account=clip --qos=medium --gres=gpu:1 \
--cpus-per-task=8 --time 0-03:00:00 --begin=now -e $err -o $out \
--wrap "job_nums=($nums_string); job_num=\${job_nums[\$SLURM_ARRAY_TASK_ID]}; \
python $root/scripts/main.py -c $cue --name_corpus $corpus -w $weight -b $beta --sim_num \$job_num --epochs 30" 
done
# done


conda deactivate

