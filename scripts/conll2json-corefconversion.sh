#!/bin/env bash

#SBATCH -A NAISS2025-5-551
#SBATCH -p alvis
#SBATCH -t 04:00:00
#SBATCH -C NOGPU
#SBATCH -J 'conll2json'
#SBATCH -o './logs-corefconversion/conll2json.out'

module purge
module load virtualenv/20.26.2-GCCcore-13.3.0
source /mimer/NOBACKUP/groups/naiss2024-6-297/envs/linkappend/bin/activate

INPUT_DIR="/mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans/models/linkappend/data-out"
OUTPUT_DIR="${INPUT_DIR}/conll_to_json"
SCRIPT_DIR="/mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans"

# Add corefconversion to PYTHONPATH so conll_transform can be found
export PYTHONPATH="${SCRIPT_DIR}/corefconversion:${PYTHONPATH}"

mkdir -p "$OUTPUT_DIR"

# Iterate through all subdirectories (model_prompt_seed folders)
for model_dir in "$INPUT_DIR"/*/; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        echo "Processing directory: $model_name"
        
        for input_fpath in "$model_dir"*_pred.conll; do
            if [ -f "$input_fpath" ]; then
                filename=$(basename -- "$input_fpath")
                output_fpath="${OUTPUT_DIR}/${model_name}_${filename%.conll}.jsonlines"

                echo "Converting $filename to $(basename "$output_fpath")..."

                python ${SCRIPT_DIR}/corefconversion/conll2jsonlines.py \
                    --token-col 1 \
                    --speaker-col "_" \
                    "$input_fpath" \
                    "$output_fpath"
            fi
        done
    fi
done
