#!/bin/bash -l
#SBATCH -A naiss2025-5-551-gpu
#SBATCH -J internvl3
#SBATCH -p gpu
#SBATCH -t 8:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=ALL
#SBATCH -o /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/internvl3/logs/%x-%j.out
#SBATCH -e /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/internvl3/logs/%x-%j.err

set -euo pipefail

module purge

if [[ -n "${SLURM_JOB_SCRIPT:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${SLURM_JOB_SCRIPT}")" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/models/internvl3"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"
RUNPY_PATH="${SCRIPT_DIR}/run.py"

export HF_MODEL="OpenGVLab/InternVL3-78B-Instruct"
export HF_HOME="/nobackup/proj/disk/naiss2024-6-297/shared/hf-home"
export HOST_VENV="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64"
export SIF_IMAGE="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/vllm-v0.25.0-aarch64.sif"
export EXTRA_LIB_HOME="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/vllm-extras"
export NVRTC_DIR="${EXTRA_LIB_HOME}/nvidia/cu13/lib"

export CLIENT_TEMPLATE_NAME="${CLIENT_TEMPLATE_NAME:-large}"
export CLIENT_OUTPUT_DIR="${CLIENT_OUTPUT_DIR:-./out-internvl3-60stories}"
export CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-16}"
export CLIENT_MAX_TOKENS="${CLIENT_MAX_TOKENS:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-${CLIENT_CONCURRENCY}}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="internvl3-${CLIENT_TEMPLATE_NAME}" >/dev/null 2>&1 || true
  TEMPLATE_LOG_OUT="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/internvl3/logs/internvl3-${CLIENT_TEMPLATE_NAME}-${SLURM_JOB_ID}.out"
  TEMPLATE_LOG_ERR="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/internvl3/logs/internvl3-${CLIENT_TEMPLATE_NAME}-${SLURM_JOB_ID}.err"
  exec > >(tee -a "${TEMPLATE_LOG_OUT}") 2> >(tee -a "${TEMPLATE_LOG_ERR}" >&2)
fi

if [[ "${ALLOW_CONCURRENT_INTERNVL3_JOBS:-0}" != "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  active_job="$(
    squeue --noheader --user="${USER}" --states=RUNNING --format='%A|%j' |
      awk -F'|' -v current_job_id="${SLURM_JOB_ID}" \
        '$1 != current_job_id && $1 + 0 < current_job_id + 0 && $2 ~ /^internvl3-/ { print; exit }'
  )"
  if [[ -n "${active_job}" ]]; then
    echo "Another InternVL3 job is already running (${active_job}); refusing concurrent launch. Set ALLOW_CONCURRENT_INTERNVL3_JOBS=1 to override." >&2
    exit 2
  fi
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

if raw.startswith('hf_'):
    print(raw)
    raise SystemExit(0)

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

export HEAD_HOSTNAME="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
export HEAD_IP="$(getent hosts "$HEAD_HOSTNAME" | awk '{print $1}' | head -n1)"
export API_PORT=$(find_open_port)
export MASTER_PORT=$(find_open_port)

VLLM_STDOUT="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/internvl3/logs/vllm-${CLIENT_TEMPLATE_NAME}-${SLURM_JOB_ID:-manual}.out"
VLLM_STDERR="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/internvl3/logs/vllm-${CLIENT_TEMPLATE_NAME}-${SLURM_JOB_ID:-manual}.err"

if [[ ! -f "${NVRTC_DIR}/libnvrtc.so.13" || ! -f "${NVRTC_DIR}/libcudart.so.13" ]]; then
  echo "Missing CUDA13 libs in ${NVRTC_DIR}. Install once with:" >&2
  echo "  apptainer exec --bind ${EXTRA_LIB_HOME}:${EXTRA_LIB_HOME} ${SIF_IMAGE} python3 -m pip install --target=${EXTRA_LIB_HOME} nvidia-cuda-nvrtc nvidia-cuda-runtime" >&2
  exit 2
fi

APPT="apptainer exec --nv \
  --bind ${EXTRA_LIB_HOME}:${EXTRA_LIB_HOME} \
  --bind /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/internvl3:/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/internvl3 \
  --bind ${HF_HOME}:${HF_HOME}"

export INCONTAINER_ENV="export LD_LIBRARY_PATH=${NVRTC_DIR}:\${LD_LIBRARY_PATH}; export NCCL_DEBUG=INFO"

vllm_common="\
  --distributed-executor-backend mp \
  --tensor-parallel-size 16 \
  --pipeline-parallel-size 1 \
  --nnodes ${SLURM_NNODES} \
  --master-addr ${HEAD_IP} \
  --master-port ${MASTER_PORT} \
  --enforce-eager \
  --max-model-len 42000 \
  --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
  --max-num-batched-tokens ${VLLM_MAX_NUM_BATCHED_TOKENS} \
  --distributed-timeout-seconds 1800 \
  --limit-mm-per-prompt.image 20 \
  --limit-mm-per-prompt.video 0 \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code"

echo "Launching vLLM (mp) across ${SLURM_NNODES} nodes; head=${HEAD_HOSTNAME} (${HEAD_IP})"
: > "${VLLM_STDOUT}"
: > "${VLLM_STDERR}"
echo "vLLM stdout: ${VLLM_STDOUT}"
echo "vLLM stderr: ${VLLM_STDERR}"
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 \
  ${APPT} ${SIF_IMAGE} bash -lc "${INCONTAINER_ENV}; \
    if [ \"\${SLURM_NODEID}\" = \"0\" ]; then EXTRA=''; else EXTRA='--headless'; fi; \
    echo \"node rank \${SLURM_NODEID} on \$(hostname) headless=[\${EXTRA}]\"; \
    vllm serve ${HF_MODEL} --port ${API_PORT} --node-rank \${SLURM_NODEID} \${EXTRA} ${vllm_common}" \
  > "${VLLM_STDOUT}" 2> "${VLLM_STDERR}" &
SRUN_PID=$!

echo "Waiting for server to start..."
HEALTH_URL="http://${HEAD_IP}:${API_PORT}/health"
server_ready() {
  if command -v curl >/dev/null 2>&1; then
    curl -sf -o /dev/null --max-time 5 "${HEALTH_URL}" && return 0
    return 1
  fi
  grep -qE 'Application startup complete|Uvicorn running on' "${VLLM_STDERR}" "${VLLM_STDOUT}" 2>/dev/null
}
for i in {1..180}; do
  if server_ready; then
    echo "Server started successfully (health OK at ${HEALTH_URL})"
    sleep 5
    break
  fi
  if ! kill -0 ${SRUN_PID} 2>/dev/null; then
    echo "vLLM srun exited before the server became ready; see ${VLLM_STDERR}"
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

SERVER_URL="${HEAD_HOSTNAME}:${API_PORT}"
echo "Starting client -> ${SERVER_URL}"
echo "Client template: ${CLIENT_TEMPLATE_NAME}"
echo "Client output dir: ${CLIENT_OUTPUT_DIR}"
echo "Client concurrency: ${CLIENT_CONCURRENCY}"
echo "Client max tokens: ${CLIENT_MAX_TOKENS}"
echo "vLLM max sequences: ${VLLM_MAX_NUM_SEQS}"
ls -la "${RUNPY_PATH}"
unset SSL_CERT_FILE
source "${HOST_VENV}/bin/activate"
python3 "${RUNPY_PATH}" \
  --output_dir ${CLIENT_OUTPUT_DIR} \
  --template_name ${CLIENT_TEMPLATE_NAME} \
  --concurrency ${CLIENT_CONCURRENCY} \
  --max_tokens ${CLIENT_MAX_TOKENS} \
  --server_url ${SERVER_URL} &
CLIENT_PID=$!

(
  consecutive_failures=0
  while kill -0 "${CLIENT_PID}" 2>/dev/null; do
    sleep 30
    if ! kill -0 "${SRUN_PID}" 2>/dev/null; then
      echo "WATCHDOG: vLLM srun process has exited; stopping the client." >&2
      kill -TERM "${CLIENT_PID}" 2>/dev/null
      break
    fi
    if curl -sf -o /dev/null --max-time 5 "${HEALTH_URL}"; then
      consecutive_failures=0
    else
      consecutive_failures=$((consecutive_failures + 1))
      echo "WATCHDOG: vLLM /health check failed (${consecutive_failures} consecutive)." >&2
      if [[ ${consecutive_failures} -ge 4 ]]; then
        echo "WATCHDOG: vLLM unresponsive for ~120s; stopping the client to avoid burning through remaining requests." >&2
        kill -TERM "${CLIENT_PID}" 2>/dev/null
        break
      fi
    fi
  done
) &
WATCHDOG_PID=$!

wait "${CLIENT_PID}"
CLIENT_RC=$?
kill "${WATCHDOG_PID}" 2>/dev/null
wait "${WATCHDOG_PID}" 2>/dev/null || true
deactivate || true
echo "Client script completed with exit code: ${CLIENT_RC}"

echo "Terminating vLLM"
kill -15 ${SRUN_PID} 2>/dev/null || true
wait ${SRUN_PID} 2>/dev/null || true
exit ${CLIENT_RC}
