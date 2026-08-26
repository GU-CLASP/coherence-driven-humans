#!/bin/env bash

#SBATCH -A naiss2025-5-551-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 72
#SBATCH --gpus 1
#SBATCH -t 2:00:00
#SBATCH -J finetune_roberta_pooled
#SBATCH -o /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/scripts/ai-detect/logs/finetune_roberta_pooled_%j.out

module purge
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
module load GPU/buildenv-gcccuda/2026.03-cu13.0

CDH=/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans
export HF_HOME=/nobackup/proj/disk/naiss2024-6-297/shared/hf-home

# Slurm hands out all 4 GPUs even with --gpus 1. Without this, HF Trainer sees
# device_count>1, wraps the model in DataParallel, and dies in NCCL init
# (NET/OFI plugin unavailable). This job is single-GPU by design.
export CUDA_VISIBLE_DEVICES=0

source $CDH/envs/aidetect/bin/activate

echo "=== GPU VISIBILITY ==="
echo "CUDA_VISIBLE_DEVICES=[$CUDA_VISIBLE_DEVICES]"
python -c "import torch; print('torch:', torch.__version__, '| device_count:', torch.cuda.device_count(), '| cuda:', torch.cuda.is_available())"
echo "======================"

cd $CDH/scripts/ai-detect

echo "Running pooled classifier with GroupKFold..."
python finetune_roberta_pooled.py --prompt-type large --cv-folds 5 --epochs 20 "$@" 2>&1

echo "Training completed!"
