#!/bin/bash -l
#SBATCH -A naiss2025-5-551-gpu
#SBATCH -J qwen3vl-original-target
#SBATCH -p gpu
#SBATCH -t 6:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=ALL
#SBATCH -o /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/qwen3vl/logs/%x-%j.out
#SBATCH -e /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/qwen3vl/logs/%x-%j.err

module purge

# Always resolve paths relative to this launcher file, regardless of sbatch CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNPY_PATH="${SCRIPT_DIR}/run.py"

# paths
export HF_MODEL="Qwen/Qwen3-VL-235B-A22B-Thinking"
export HF_HOME="/nobackup/proj/disk/naiss2024-6-297/shared/hf-home"
# client-only venv (run.py). Runs on the HOST, never inside the container.
export HOST_VENV="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64"
export SIF_IMAGE="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/vllm-v0.25.0-aarch64.sif"

# change this parameters depending on the prompt, the output folder and how many requests you want to send in parallel (16 is the max (kind of) for 4xA100-80GB)
export CLIENT_TEMPLATE_NAME="${CLIENT_TEMPLATE_NAME:-large-upper-bound}"
export CLIENT_OUTPUT_DIR="${CLIENT_OUTPUT_DIR:-./out-qwen3vl-60stories}"
export CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-16}"

# discriminative naming for the log files
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="qwen3vl-${CLIENT_TEMPLATE_NAME}" >/dev/null 2>&1 || true
  TEMPLATE_LOG_OUT="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/qwen3vl/logs/qwen3vl-${CLIENT_TEMPLATE_NAME}-${SLURM_JOB_ID}.out"
  TEMPLATE_LOG_ERR="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/qwen3vl/logs/qwen3vl-${CLIENT_TEMPLATE_NAME}-${SLURM_JOB_ID}.err"
  exec > >(tee -a "${TEMPLATE_LOG_OUT}") 2> >(tee -a "${TEMPLATE_LOG_ERR}" >&2)
fi

# container-side extra libraries, because the container's cu129 PyTorch does not ship the CUDA-13 libs (libnvrtc.so.13 / libcudart.so.13) that torchcodec loads at `vllm serve` startup.
# install them once for the container's python into this dir (bind-mounted below), which keeps the model env (models-arm64) completely separate from the container:
#   apptainer exec $SIF_IMAGE python3 -m pip install --target=$EXTRA_LIB_HOME nvidia-cuda-nvrtc nvidia-cuda-runtime
export EXTRA_LIB_HOME="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/vllm-extras"
export NVRTC_DIR="${EXTRA_LIB_HOME}/nvidia/cu13/lib"

# bind ONLY what the container needs: the CUDA-13 extras, the working dir, and the HF cache.
# models-arm64 is deliberately NOT bound, so the container cannot touch it.
APPT="apptainer exec --nv \
  --bind ${EXTRA_LIB_HOME}:${EXTRA_LIB_HOME} \
  --bind /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/qwen3vl:/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/qwen3vl \
  --bind ${HF_HOME}:${HF_HOME}"

# Environment applied inside each container command (CUDA-13 libs; NCCL diagnostics).
# NCCL_DEBUG=INFO produces a lot of output, but it is useful for debugging multi-node vLLM hangs
export INCONTAINER_ENV="export LD_LIBRARY_PATH=${NVRTC_DIR}:\${LD_LIBRARY_PATH}; export NCCL_DEBUG=INFO"

# Load HuggingFace token from environment or an optional config file
if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  TOKEN_FILE="/nobackup/proj/disk/naiss2024-6-297/personal/ilinykh/config.json"
  if [[ -f "$TOKEN_FILE" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
  else
    echo "Warning: HUGGING_FACE_HUB_TOKEN is not set and $TOKEN_FILE does not exist; relying on cached models or anonymous access"
  fi
fi

find_open_port() {
  python3 - <<'PY'
import socket
sock = socket.socket()
sock.bind(("", 0))
print(sock.getsockname()[1])
sock.close()
PY
}


# use new feature from vllm, the multi-node multiprocessing (mp) backend, which uses torch.distributed to launch vLLM on all nodes in the SLURM allocation. The head node (rank 0) hosts the API server; other ranks run with --headless.
export HEAD_HOSTNAME="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
export HEAD_IP="$(getent hosts "$HEAD_HOSTNAME" | awk '{print $1}' | head -n1)"
export API_PORT=$(find_open_port)      # OpenAI API server port on the head node (rank 0)
export MASTER_PORT=$(find_open_port)   # torch.distributed rendezvous port on the head node

# vLLM options for multi-node multiprocessing.
# NOTE: Qwen3-VL-MoE crashes at init with pipeline_parallel_size > 1 in v0.25.0
# ("No model architectures are specified" — vLLM bug #43271, fix PR #43272 not yet
# released). So use pure tensor parallel across ALL 16 GPUs (tp=16, pp=1).
# NOTE: --enforce-eager disables torch.compile + CUDA-graph capture. On this
# multi-node (tp=16) setup the vision Triton kernels JIT-compile mid-inference on
# only some ranks, which desyncs the cross-node NCCL collective under CUDA graphs
# and deadlocks the engine ("No available shared memory broadcast block ..."; see
# vLLM issue #30682). Eager execution keeps every rank on the same deterministic
# path. It is slower per token but stable, which is the right trade-off here.
vllm_common="\
  --distributed-executor-backend mp \
  --tensor-parallel-size 16 \
  --pipeline-parallel-size 1 \
  --nnodes ${SLURM_NNODES} \
  --master-addr ${HEAD_IP} \
  --master-port ${MASTER_PORT} \
  --enable-expert-parallel \
  --enforce-eager \
  --max-model-len 48000 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --distributed-timeout-seconds 1800 \
  --limit-mm-per-prompt.image 20 \
  --limit-mm-per-prompt.video 0 \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.9"


echo "Launching vLLM (mp) across ${SLURM_NNODES} nodes; head=${HEAD_HOSTNAME} (${HEAD_IP})"
: > vllm.out
: > vllm.err
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 \
  ${APPT} ${SIF_IMAGE} bash -lc "${INCONTAINER_ENV}; \
    if [ \"\${SLURM_NODEID}\" = \"0\" ]; then EXTRA=''; else EXTRA='--headless'; fi; \
    echo \"node rank \${SLURM_NODEID} on \$(hostname) headless=[\${EXTRA}]\"; \
    vllm serve ${HF_MODEL} --port ${API_PORT} --node-rank \${SLURM_NODEID} \${EXTRA} ${vllm_common}" \
  > vllm.out 2> vllm.err &
SRUN_PID=$!


echo "Waiting for server to start..."
HEALTH_URL="http://${HEAD_IP}:${API_PORT}/health"
server_ready() {
  if command -v curl >/dev/null 2>&1; then
    curl -sf -o /dev/null --max-time 5 "${HEALTH_URL}" && return 0
    return 1
  fi
  grep -qE 'Application startup complete|Uvicorn running on' vllm.err vllm.out 2>/dev/null
}
for i in {1..180}; do
  if server_ready; then
    echo "Server started successfully (health OK at ${HEALTH_URL})"
    sleep 5
    break
  fi
  if ! kill -0 ${SRUN_PID} 2>/dev/null; then
    echo "vLLM srun exited before the server became ready; see vllm.err"
    exit 1
  fi
  if [ $i -eq 180 ]; then
    echo "vLLM doesn't seem to start, aborting"
    kill -15 ${SRUN_PID}
    exit 1
  fi
  echo "Waiting... ($i/180)"
  sleep 10
done

# this runs on the host, not inside the container, so it can use the host's venv (HOST_VENV) and run run.py directly.
SERVER_URL="${HEAD_HOSTNAME}:${API_PORT}"
echo "Starting client -> ${SERVER_URL}"
echo "Client template: ${CLIENT_TEMPLATE_NAME}"
echo "Client output dir: ${CLIENT_OUTPUT_DIR}"
echo "Client concurrency: ${CLIENT_CONCURRENCY}"
ls -la "${RUNPY_PATH}"
unset SSL_CERT_FILE
source ${HOST_VENV}/bin/activate
python3 "${RUNPY_PATH}" \
  --output_dir ${CLIENT_OUTPUT_DIR} \
  --template_name ${CLIENT_TEMPLATE_NAME} \
  --concurrency ${CLIENT_CONCURRENCY} \
  --server_url ${SERVER_URL}
CLIENT_RC=$?
deactivate || true
echo "Client script completed with exit code: ${CLIENT_RC}"

# shut down
echo "Terminating vLLM"
kill -15 ${SRUN_PID} 2>/dev/null || true
wait ${SRUN_PID} 2>/dev/null || true
exit ${CLIENT_RC}