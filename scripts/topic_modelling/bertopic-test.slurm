#!/bin/env bash

#SBATCH -A naiss2025-5-551-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 72
#SBATCH --gpus 1
#SBATCH -t 6:00:00
#SBATCH -J bertopic-test
#SBATCH -o /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/scripts/topic_modelling/logs_test/bertopic-test-%A_%a.out
#SBATCH --array=0-9

# Test BERTopic models on full dataset
# Array job: each task handles one bootstrap model (0-9)
# For each bootstrap, we test all topic configurations (80, 75, ..., 5)

module purge
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
module load GPU/buildenv-gcccuda/2026.03-cu13.0

# Slurm grants all 4 GPUs on the node whatever --gpus says. Anything that counts
# devices then goes multi-GPU, and NCCL cannot initialize here (OFI net plugin).
export CUDA_VISIBLE_DEVICES=0

CDH=/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans
export HF_HOME=/nobackup/proj/disk/naiss2024-6-297/shared/hf-home

source $CDH/envs/bertopic/bin/activate
cd $CDH/scripts/topic_modelling || exit 1

echo "=== GPU VISIBILITY ==="
echo "host: $(hostname)   arch: $(uname -m)"
echo "CUDA_VISIBLE_DEVICES=[$CUDA_VISIBLE_DEVICES]   SLURM_JOB_GPUS=[$SLURM_JOB_GPUS]"
python -c "import torch; print('device_count:', torch.cuda.device_count())"
echo "======================"

BASE_CACHE_DIR=${BASE_CACHE_DIR:-/nobackup/proj/disk/naiss2024-6-297/shared/bertopic_bootstrapped}
MIN_TOPIC_SIZE=${MIN_TOPIC_SIZE:-10}

MIN_TOPIC_SIZE_LABEL="default"
if [ -n "${MIN_TOPIC_SIZE}" ] && [ "${MIN_TOPIC_SIZE}" != "none" ] && [ "${MIN_TOPIC_SIZE}" != "None" ]; then
    MIN_TOPIC_SIZE_LABEL="${MIN_TOPIC_SIZE}"
fi

RUN_CACHE_DIR="${BASE_CACHE_DIR}/full_training_${MIN_TOPIC_SIZE_LABEL}"
LOG_DIR="./logs_test/full_training_${MIN_TOPIC_SIZE_LABEL}"
mkdir -p "${LOG_DIR}"

# Get bootstrap index from array task ID
BOOTSTRAP_IDX=${SLURM_ARRAY_TASK_ID}

LOG_FILE="${LOG_DIR}/bertopic-test-bootstrap_${BOOTSTRAP_IDX}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Testing BERTopic Models for Bootstrap ${BOOTSTRAP_IDX}"
echo "=========================================="
echo "Cache dir: ${RUN_CACHE_DIR}"
echo "Log file: ${LOG_FILE}"
echo "min_topic_size: ${MIN_TOPIC_SIZE}"

# Iterate through different numbers of topics (same as training: 80, 75, ..., 5)
for NR_TOPICS in $(seq 80 -5 5); do
    echo ""
    echo "Testing bootstrap ${BOOTSTRAP_IDX} with ${NR_TOPICS} topics..."

    python bertopic-test.py \
        --bootstrap-idx ${BOOTSTRAP_IDX} \
        --nr-topics ${NR_TOPICS} \
        --cache-dir "${RUN_CACHE_DIR}"
done

echo ""
echo "Bootstrap ${BOOTSTRAP_IDX}: All tests completed!"
