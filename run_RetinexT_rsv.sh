#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --mem=164G
#SBATCH --cpus-per-task=16
#SBATCH --account=dcs-mvg1
#SBATCH --reservation=dcs-mvg
#SBATCH --time=24:00:00
#SBATCH --output=./output/output.%j.test.out
#SBATCH --mail-user=xchen235@sheffield.ac.uk
#SBATCH --mail-type=FAIL

nvidia-smi -L
nvidia-smi

module load cuDNN/8.0.4.30-CUDA-11.1.1
module load GCC/12.2.0

# load the module for the program we want to run
module load Anaconda3/2022.10
source activate ResT

mkdir -p $TMPDIR/ImageNet

#cp -r /mnt/parscratch/users/acs21xc/ImageNet/train $TMPDIR/ImageNet
#cp -r /mnt/parscratch/users/acs21xc/ImageNet/val $TMPDIR/ImageNet
#echo copy to scratch done!

python -m torch.distributed.launch --nproc_per_node=2 main.py \
--model rest_lite --drop_path 0.1 \
--clip_grad 1.0 --warmup_epochs 50 --epochs 300 \
--batch_size 1024 --lr 1.5e-4 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /mnt/parscratch/users/acs21xc/ImageNet100/ \
--output_dir ./output/checkpoints/
