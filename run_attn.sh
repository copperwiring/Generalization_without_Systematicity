#!/bin/bash
#SBATCH --job-name=attn_exp3
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.

cd /home/vsl333/nlp-course/Generalization_without_Systematicity/
source /home/vsl333/atnlpenv/bin/activate
hostname
echo $CUDA_VISIBLE_DEVICES
python3 main.py
