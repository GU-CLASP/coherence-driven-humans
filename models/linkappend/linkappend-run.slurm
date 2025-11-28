#!/bin/env bash

#SBATCH -A NAISS2025-5-551
#SBATCH -p alvis
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=A100:1

module purge
module load virtualenv/20.26.2-GCCcore-13.3.0
module load CUDA/12.4.0

source /mimer/NOBACKUP/groups/naiss2024-6-297/envs/linkappend/bin/activate

which python
python --version

INPUT_DIR="/mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans/models/linkappend/data-in"
BASE_OUTPUT_DIR="/mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans/models/linkappend/data-out"
MODEL_PATH="/mimer/NOBACKUP/groups/naiss2024-6-297/hf-home/hub/models--ianporada--link-append-xxl/snapshots/16030717bb7846ef5c0380d6bd80da234fe0aa79/"
LINKAPPEND_DIR="/mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans/coref-reeval/models/decoder_based/LinkAppend"
SCRIPTS_DIR="/mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans/models/linkappend"

# Add scripts directory to Python path so conll_transform can be found
export PYTHONPATH="${SCRIPTS_DIR}:${PYTHONPATH}"

cd "$LINKAPPEND_DIR"

# Process all JSON files in the input directory
for INPUT_FILE in "$INPUT_DIR"/*.json; do
    FILENAME=$(basename "$INPUT_FILE" .json)
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${FILENAME}"
    
    echo "Processing: $FILENAME"
    echo "Input file: $INPUT_FILE"
    echo "Output dir: $OUTPUT_DIR"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Create temporary directory for this specific file
    TEMP_INPUT_DIR="/tmp/linkappend_${FILENAME}_$$"
    mkdir -p "$TEMP_INPUT_DIR"
    
    # Copy the specific input file to temp directory
    cp "$INPUT_FILE" "$TEMP_INPUT_DIR/"
    
    python main.py \
        --model_path "$MODEL_PATH" \
        --input_dir "$TEMP_INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --split test \
        --batch_size 1 \
        --max_input_size 3000 \
        --no_pound_symbol \
        --subset 0 \
        --subset_start 0
    
    # Clean up temp directory
    rm -rf "$TEMP_INPUT_DIR"
    
    echo "Completed: $FILENAME"
    echo "---"
done

echo "All files processed!"
