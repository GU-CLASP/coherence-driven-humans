#!/bin/env bash

#SBATCH -A NAISS2025-5-551
#SBATCH -p alvis
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J 'finetune_roberta_pooled'
#SBATCH -o '/mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans/scripts/ai-detect/logs/finetune_roberta_pooled.out'

module purge
module load virtualenv/20.29.2-GCCcore-14.2.0
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/13.0.2

source /mimer/NOBACKUP/groups/naiss2025-22-1187/orig-tacl/ai-detector/aigenfinetune/bin/activate

echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Change to the script directory
cd /mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans/scripts/ai-detect

# Run the training script
echo "Starting fine-tuning..."
echo "Arguments: $@"

echo "Running pooled classifier with GroupKFold..."
python finetune_roberta_pooled.py --prompt-type large --cv-folds 5 --epochs 20 "$@" 2>&1

echo "Training completed!"