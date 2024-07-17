#!/bin/bash

#SBATCH --output=./output/output.%j.test.out

#SBATCH --mail-user=xchen235@sheffield.ac.uk
#SBATCH --mail-type=FAIL

# NB Each NVIDIA A100 GPU in Stanage has 80GB of RAM
# srun --partition=gpu --qos=gpu --gres=gpu:2 --mem=164G --pty bash

#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

#SBATCH --time=48:00:00


module load cuDNN/8.0.4.30-CUDA-11.1.1

# load the module for the program we want to run
module load Anaconda3/2022.10
source activate Retinexformer

nvidia-smi -L
nvidia-smi

python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v1.yml