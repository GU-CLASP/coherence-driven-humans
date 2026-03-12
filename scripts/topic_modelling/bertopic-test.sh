#!/bin/env bash

#SBATCH -A NAISS2025-5-551
#SBATCH -p alvis
#SBATCH -t 6:00:00
#SBATCH --gpus-per-node=A40:3
#SBATCH -J "bertopic-test"
#SBATCH -o "./logs_test/bertopic-test-%A_%a.out"
#SBATCH --array=0-9

# Test BERTopic models on full dataset
# Array job: each task handles one bootstrap model (0-9)
# For each bootstrap, we test all topic configurations (80, 75, ..., 5)

module purge
module load virtualenv/20.26.2-GCCcore-13.3.0
module load CUDA/12.9.0

source /mimer/NOBACKUP/groups/naiss2024-6-297/envs/coref/bin/activate

# Create log directory
BASE_CACHE_DIR=${BASE_CACHE_DIR:-/mimer/NOBACKUP/groups/naiss2024-6-297/cache/bertopic_bootstrapped}
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
    
    /mimer/NOBACKUP/groups/naiss2024-6-297/envs/coref/bin/python bertopic-test.py \
        --bootstrap-idx ${BOOTSTRAP_IDX} \
        --nr-topics ${NR_TOPICS} \
        --cache-dir "${RUN_CACHE_DIR}"
done

echo ""
echo "Bootstrap ${BOOTSTRAP_IDX}: All tests completed!"