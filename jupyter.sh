#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=jupyter_%j.log
##SBATCH --qos=batch
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
##Next lines requeset specific GPU type or just any GPU, respectively
##SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --gres=gpu:1
##Use -n to request number of cpus per task 
#SBATCH -n 2
#SBATCH --mem=60gb
#SBATCH --time=4:00:00

date;hostname;pwd

# Use a non-home directory that will be used for jupyter temporary space
# unset XDG_RUNTIME_DIR (legacy line from an earlier version; keeping just in case)
export XDG_RUNTIME_DIR="/fs/clip-realspeech/projects/speech_in_noise/me_final"

source /fs/clip-realspeech/projects/speech_in_noise/miniconda/bin/activate
conda activate /fs/clip-realspeech/projects/speech_in_noise/miniconda/envs/cue_env

# Set a port -- pick a 4-digit number starting with 8 or 9 that 
# you don't think someone else is using or is random. (Ugh, no way
# to check?!)
port=1080
 
echo -e "\nStarting Jupyter Notebook on port ${port} on the $(hostname) server."

# CLIP filesystem quirk: you need to cd into directories in /fs
# to mount them, otherwise they might not be visible to the shell.
# Add any /fs subdirectories you want to be able to get to via jupyter.
cd /fs/clip-realspeech/corpora/
cd /fs/clip-realspeech/projects/speech_in_noise/me_final

# By default, starting in /fs
cd /fs/clip-realspeech/projects/speech_in_noise/me_final

#Optional module loading - you want these if you are running pytorch, AllenNLP, etc.
module load cuda
module load cudnn

# Start jupyter server without launching a browser (that will be on your local machine)
# using port specified above
jupyter notebook --no-browser --port=${port} --ip=0.0.0.0

date

# run: 
# ssh -N -f -L localhost:8888:nexusclip00:1080 njurov@nexusclip00.umiacs.umd.edu