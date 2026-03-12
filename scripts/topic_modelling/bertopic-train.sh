#!/bin/env bash

#SBATCH -A NAISS2025-5-551
#SBATCH -p alvis
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=A40:3
#SBATCH -J 'bertopic-train'
#SBATCH -o './logs_train/bertopic-train-bootstrap_%a.out'
#SBATCH --array=0-9

# Train BERTopic on balanced bootstrap sets
# Array job: each task handles one bootstrap set (0-9)

module purge
module load virtualenv/20.26.2-GCCcore-13.3.0
module load CUDA/12.9.0

source /mimer/NOBACKUP/groups/naiss2024-6-297/envs/coref/bin/activate

# Get bootstrap index from array task ID
BOOTSTRAP_IDX=${SLURM_ARRAY_TASK_ID}

PILOT_DISCOVERY=${PILOT_DISCOVERY:-0}
BASE_CACHE_DIR=${BASE_CACHE_DIR:-/mimer/NOBACKUP/groups/naiss2024-6-297/cache/bertopic_bootstrapped}
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
    /mimer/NOBACKUP/groups/naiss2024-6-297/envs/coref/bin/python bertopic-train.py \
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
    /mimer/NOBACKUP/groups/naiss2024-6-297/envs/coref/bin/python bertopic-train.py \
        --bootstrap-idx ${BOOTSTRAP_IDX} \
        --cache-dir "${RUN_CACHE_DIR}" \
        ${MIN_TOPIC_SIZE_ARG} \
        --nr-topics-start 80 \
        --nr-topics-end 5 \
        --nr-topics-step 5
fi

echo "Bootstrap ${BOOTSTRAP_IDX} completed!"
