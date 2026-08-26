#!/bin/env bash

#SBATCH -A naiss2025-5-551-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 72
#SBATCH --gpus 1
#SBATCH -t 12:00:00
#SBATCH -J bertopic-train
#SBATCH -o /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/scripts/topic_modelling/logs_train/bertopic-train-bootstrap_%A_%a.out
#SBATCH --array=0-9

# Train BERTopic on balanced bootstrap sets
# Array job: each task handles one bootstrap set (0-9)

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

# Get bootstrap index from array task ID
BOOTSTRAP_IDX=${SLURM_ARRAY_TASK_ID}

PILOT_DISCOVERY=${PILOT_DISCOVERY:-0}
BASE_CACHE_DIR=${BASE_CACHE_DIR:-/nobackup/proj/disk/naiss2024-6-297/shared/bertopic_bootstrapped}
MIN_TOPIC_SIZE=${MIN_TOPIC_SIZE:-10}

MIN_TOPIC_SIZE_LABEL="default"

MIN_TOPIC_SIZE_ARG=""
if [ -n "${MIN_TOPIC_SIZE}" ] && [ "${MIN_TOPIC_SIZE}" != "none" ] && [ "${MIN_TOPIC_SIZE}" != "None" ]; then
    MIN_TOPIC_SIZE_ARG="--min-topic-size ${MIN_TOPIC_SIZE}"
    MIN_TOPIC_SIZE_LABEL="${MIN_TOPIC_SIZE}"
fi

if [ "${PILOT_DISCOVERY}" -eq 1 ]; then
    MODE_LABEL="pilot_discovery"
    RUN_CACHE_DIR="${BASE_CACHE_DIR}/${MODE_LABEL}_${MIN_TOPIC_SIZE_LABEL}"
    LOG_DIR="./logs_train/${MODE_LABEL}_${MIN_TOPIC_SIZE_LABEL}"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/bertopic-train-bootstrap_${BOOTSTRAP_IDX}.log"
    exec > >(tee -a "${LOG_FILE}") 2>&1

    echo "=========================================="
    echo "Training BERTopic on Bootstrap Set ${BOOTSTRAP_IDX}"
    echo "=========================================="
    echo "Running PILOT DISCOVERY mode (nr_topics=None, no reduction sweep)"
    echo "Output/cache root: ${RUN_CACHE_DIR}"
    echo "Log file: ${LOG_FILE}"
    echo "min_topic_size: ${MIN_TOPIC_SIZE}"
    python bertopic-train.py \
        --bootstrap-idx ${BOOTSTRAP_IDX} \
        --cache-dir "${RUN_CACHE_DIR}" \
        ${MIN_TOPIC_SIZE_ARG} \
        --pilot-discovery
else
    MODE_LABEL="full_training"
    RUN_CACHE_DIR="${BASE_CACHE_DIR}/${MODE_LABEL}_${MIN_TOPIC_SIZE_LABEL}"
    LOG_DIR="./logs_train/${MODE_LABEL}_${MIN_TOPIC_SIZE_LABEL}"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/bertopic-train-bootstrap_${BOOTSTRAP_IDX}.log"
    exec > >(tee -a "${LOG_FILE}") 2>&1

    echo "=========================================="
    echo "Training BERTopic on Bootstrap Set ${BOOTSTRAP_IDX}"
    echo "=========================================="
    echo "Running FULL TRAINING mode (topic reduction sweep)"
    echo "Output/cache root: ${RUN_CACHE_DIR}"
    echo "Log file: ${LOG_FILE}"
    echo "min_topic_size: ${MIN_TOPIC_SIZE}"
    python bertopic-train.py \
        --bootstrap-idx ${BOOTSTRAP_IDX} \
        --cache-dir "${RUN_CACHE_DIR}" \
        ${MIN_TOPIC_SIZE_ARG} \
        --nr-topics-start 80 \
        --nr-topics-end 5 \
        --nr-topics-step 5
fi

echo "Bootstrap ${BOOTSTRAP_IDX} completed!"
