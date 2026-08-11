#!/bin/bash -l
#SBATCH -A naiss2025-5-551-gpu
#SBATCH -J llama4scout
#SBATCH -p gpu
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=ALL
#SBATCH -o /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/llama4scout/logs/%x-%j.out
#SBATCH -e /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/llama4scout/logs/%x-%j.err

set -euo pipefail

module purge

if [[ -n "${SLURM_JOB_SCRIPT:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${SLURM_JOB_SCRIPT}")" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/models/llama4scout"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"
RUNPY_PATH="${SCRIPT_DIR}/run.py"

# Shared paths on Arrhenius
export HF_MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct"
export HF_HOME="/nobackup/proj/disk/naiss2024-6-297/shared/hf-home"
export HOST_VENV="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64"
export SIF_IMAGE="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/vllm-v0.25.0-aarch64.sif"
export EXTRA_LIB_HOME="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/vllm-extras"
export NVRTC_DIR="${EXTRA_LIB_HOME}/nvidia/cu13/lib"

# Run controls (overridable via sbatch --export=ALL,...)
export CLIENT_TEMPLATE_NAME="${CLIENT_TEMPLATE_NAME:-original}"
export CLIENT_OUTPUT_DIR="${CLIENT_OUTPUT_DIR:-./out-llama4scout-60stories}"
export CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-16}"
export CLIENT_MAX_TOKENS="${CLIENT_MAX_TOKENS:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-${CLIENT_CONCURRENCY}}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="llama4scout-${CLIENT_TEMPLATE_NAME}" >/dev/null 2>&1 || true
fi

# Load Hugging Face token from env first, then fallback file.
# The fallback file may contain either:
# 1) a raw token string, or
# 2) JSON with one of keys: huggingface_token, hf_token, token.
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-/nobackup/proj/disk/naiss2024-6-297/personal/ilinykh/config.json}"
if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "${HF_TOKEN_FILE}" ]]; then
    token_value="$(python3 - <<'PY' "${HF_TOKEN_FILE}"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
raw = path.read_text(encoding='utf-8').strip()

# Raw token file (common case)
if raw.startswith('hf_'):
    print(raw)
    raise SystemExit(0)

# JSON token file
try:
    obj = json.loads(raw)
except Exception:
    print(raw)
    raise SystemExit(0)

if isinstance(obj, str):
    print(obj.strip())
elif isinstance(obj, dict):
    for key in ('huggingface_token', 'hf_token', 'token'):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            print(val.strip())
            break
PY
)"
    if [[ -n "${token_value:-}" ]]; then
      export HUGGING_FACE_HUB_TOKEN="${token_value}"
      export HF_TOKEN="${token_value}"
    else
      echo "Warning: token file ${HF_TOKEN_FILE} is present but no token could be parsed." >&2
    fi
  else
    echo "Warning: no Hugging Face token in env and token file ${HF_TOKEN_FILE} not found; relying on cached models or anonymous access" >&2
  fi
fi

# Keep both names in sync for compatibility across huggingface_hub/transformers/vLLM code paths.
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
elif [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
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

export HEAD_HOSTNAME="$(hostname)"
export HEAD_IP="$(hostname -I | awk '{print $1}')"
export API_PORT="$(find_open_port)"

VLLM_STDOUT="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/llama4scout/logs/vllm-${CLIENT_TEMPLATE_NAME}-${SLURM_JOB_ID:-manual}.out"
VLLM_STDERR="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/llama4scout/logs/vllm-${CLIENT_TEMPLATE_NAME}-${SLURM_JOB_ID:-manual}.err"

APPT="apptainer exec --nv \
  --bind ${EXTRA_LIB_HOME}:${EXTRA_LIB_HOME} \
  --bind /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/llama4scout:/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/llama4scout \
  --bind ${HF_HOME}:${HF_HOME}"

export INCONTAINER_ENV="export LD_LIBRARY_PATH=${NVRTC_DIR}:\${LD_LIBRARY_PATH}"

vllm_opts="\
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --max-model-len 32000 \
  --limit-mm-per-prompt.image 20 \
  --limit-mm-per-prompt.video 0 \
  --mm-processor-cache-gb 0 \
  --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
  --enforce-eager \
  --gpu-memory-utilization 0.9"

echo "Launching vLLM on ${HEAD_HOSTNAME} (${HEAD_IP}:${API_PORT})"
: > "${VLLM_STDOUT}"
: > "${VLLM_STDERR}"
if [[ ! -f "${NVRTC_DIR}/libnvrtc.so.13" || ! -f "${NVRTC_DIR}/libcudart.so.13" ]]; then
  echo "Missing CUDA13 libs in ${NVRTC_DIR}. Install once with:" >&2
  echo "  apptainer exec --bind ${EXTRA_LIB_HOME}:${EXTRA_LIB_HOME} ${SIF_IMAGE} python3 -m pip install --target=${EXTRA_LIB_HOME} nvidia-cuda-nvrtc nvidia-cuda-runtime" >&2
  exit 2
fi

${APPT} ${SIF_IMAGE} bash -lc "${INCONTAINER_ENV}; vllm serve ${HF_MODEL} --port ${API_PORT} ${vllm_opts}" \
  > "${VLLM_STDOUT}" 2> "${VLLM_STDERR}" &
VLLM_PID=$!

HEALTH_URL="http://${HEAD_IP}:${API_PORT}/health"
echo "Waiting for server readiness at ${HEALTH_URL}"
for i in {1..180}; do
  if command -v curl >/dev/null 2>&1 && curl -sf -o /dev/null --max-time 5 "${HEALTH_URL}"; then
    echo "Server started successfully"
    sleep 5
    break
  fi
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "vLLM exited before becoming ready; see ${VLLM_STDERR}"
    exit 1
  fi
  if [[ $i -eq 180 ]]; then
    echo "vLLM did not become ready within timeout"
    kill -15 ${VLLM_PID} 2>/dev/null || true
    exit 1
  fi
  echo "Waiting... (${i}/180)"
  sleep 10
done

SERVER_URL="${HEAD_HOSTNAME}:${API_PORT}"
echo "Starting client -> ${SERVER_URL}"
echo "Client template: ${CLIENT_TEMPLATE_NAME}"
echo "Client output dir: ${CLIENT_OUTPUT_DIR}"
echo "Client concurrency: ${CLIENT_CONCURRENCY}"
echo "Client max tokens: ${CLIENT_MAX_TOKENS}"

unset SSL_CERT_FILE
source "${HOST_VENV}/bin/activate"
python3 "${RUNPY_PATH}" \
  --output_dir "${CLIENT_OUTPUT_DIR}" \
  --template_name "${CLIENT_TEMPLATE_NAME}" \
  --concurrency "${CLIENT_CONCURRENCY}" \
  --max_tokens "${CLIENT_MAX_TOKENS}" \
  --server_url "${SERVER_URL}"
CLIENT_RC=$?
deactivate || true

echo "Client finished with exit code ${CLIENT_RC}"
echo "Terminating vLLM"
kill -15 ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} || true

exit ${CLIENT_RC}
