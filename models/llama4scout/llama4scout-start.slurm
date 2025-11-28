#!/bin/bash -l
#SBATCH -A NAISS2025-5-551
#SBATCH -t 8:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node "A100fat:4"

module purge

APPT="apptainer exec --nv"

# Get the first allocated node (works for both single and multi-node)
export HEAD_HOSTNAME=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export HF_MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct"
export HF_HOME="/mimer/NOBACKUP/groups/naiss2024-6-297/hf-home"

# Load HuggingFace token from config file
export HUGGING_FACE_HUB_TOKEN=$(python3 -c "import json; print(json.load(open('config.json'))['huggingface_token'])")
export HUGGING_FACE_HUB_TOKEN=$(python3 -c "import json; print(json.load(open('config.json'))['huggingface_token'])")

export SIF_IMAGE="/mimer/NOBACKUP/groups/naiss2024-6-297/vllm-images/vllm-v-0-11-0.sif"

# start vllm server
vllm_opts="\
  --tensor-parallel-size=${SLURM_GPUS_ON_NODE} \
  --max-model-len=32000 \
  --limit-mm-per-prompt.image 20 \
  --limit-mm-per-prompt.video 0 \
  --mm-processor-cache-gb=0 \
  --max-num-seqs=16 \
  --enforce-eager \
  --gpu-memory-utilization=0.9"

export API_PORT=$(find_ports)

echo "Starting server"
apptainer exec ${SIF_IMAGE} vllm serve ${HF_MODEL} \
  --port ${API_PORT} ${vllm_opts} \
  > vllm.out  2> vllm.err &
VLLM_PID=$!
sleep 20

# wait at most 10 min for the model to start, otherwise abort
echo "Waiting for server to start..."
for i in {1..60}; do
  if grep -qE 'Application startup complete|Uvicorn running on socket' vllm.err; then
    echo "Server started successfully"
    sleep 5
    break
  fi
  if [ $i -eq 60 ]; then
    echo "vLLM doesn't seem to start, aborting"
    kill -15 ${VLLM_PID}
    exit 1
  fi
  echo "Waiting... ($i/60)"
  sleep 10
done

echo "Proceeding to client execution"

echo "Starting client"
SERVER_URL="${HEAD_HOSTNAME}:${API_PORT}"
echo "Using server URL: ${SERVER_URL}"
echo "Current directory: $(pwd)"
echo "Contents of current directory:"
ls -la run2.py
apptainer exec ${SIF_IMAGE} bash -c "unset SSL_CERT_FILE && source /mimer/NOBACKUP/groups/naiss2024-6-297/vllm-environments/qwen3vl/bin/activate && python3 run.py \
  --csv_file ../../notebooks/collected_60.csv \
  --output_dir ./out-llama4scout-60stories \
  --template_name original \
  --server_url ${SERVER_URL}"
echo "Client script completed with exit code: $?"

echo "Terminating VLLM" && kill -15 ${VLLM_PID}
wait