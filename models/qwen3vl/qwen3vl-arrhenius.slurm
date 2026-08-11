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

if [[ -n "${SLURM_JOB_SCRIPT:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${SLURM_JOB_SCRIPT}")" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/models/qwen3vl"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"
RUNPY_PATH="${SCRIPT_DIR}/run.py"

# paths
export HF_MODEL="Qwen/Qwen3-VL-235B-A22B-Thinking"
export HF_HOME="/nobackup/proj/disk/naiss2024-6-297/shared/hf-home"
# client-only venv (run.py). Runs on the HOST, never inside the container.
export HOST_VENV="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64"
export SIF_IMAGE="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/vllm-v0.25.0-aarch64.sif"

# Change these parameters depending on the prompt and output folder.
# Concurrency ceiling, verified from job 456664's own startup log (not a guess):
#   GPU KV cache size: 1,230,784 tokens
#   vLLM's own conservative bound: "Maximum concurrency for 48,000 tokens per
#   request: 25.64x" (i.e. capacity / max-model-len, assuming EVERY request
#   uses the full 48k context — the safest possible ceiling).
# Real observed usage is much smaller than that worst case: story prompts were
# only ~1,750-2,400 tokens (measured from num_computed_tokens - num_output_tokens
# in the dump_input.py crash dumps of jobs 427686/427692), so with the 4096
# max_tokens cap, a real story worst-case is ~6,500 tokens, not 48,000 — capacity
# alone would allow ~180 concurrent. Going that far above vLLM's own 25.64x
# guarantee is risky though: it relies on every remaining story never needing
# close to a full 48k context; if one ever does, vLLM preempts other running
# sequences to make room, which adds another source of instability on top of
# the still-unresolved JIT-recompile/NCCL desync issue below. Recommended: stay
# at or under 24 (just under the 25.64x verified-safe bound) — this is a
# capacity ceiling only, NOT a fix for that separate instability, which struck
# at concurrency=8 and is independent of batch size.
export CLIENT_TEMPLATE_NAME="${CLIENT_TEMPLATE_NAME:-large-upper-bound}"
export CLIENT_OUTPUT_DIR="${CLIENT_OUTPUT_DIR:-./out-qwen3vl-60stories}"
export CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-24}"
export CLIENT_MAX_TOKENS="${CLIENT_MAX_TOKENS:-8192}"
export CLIENT_THINKING_TOKEN_BUDGET="${CLIENT_THINKING_TOKEN_BUDGET:--1}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-${CLIENT_CONCURRENCY}}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
export VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-1}"

# discriminative naming for the log files
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="qwen3vl-${CLIENT_TEMPLATE_NAME}" >/dev/null 2>&1 || true
  TEMPLATE_LOG_OUT="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/qwen3vl/logs/qwen3vl-${CLIENT_TEMPLATE_NAME}-${SLURM_JOB_ID}.out"
  TEMPLATE_LOG_ERR="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/qwen3vl/logs/qwen3vl-${CLIENT_TEMPLATE_NAME}-${SLURM_JOB_ID}.err"
  exec > >(tee -a "${TEMPLATE_LOG_OUT}") 2> >(tee -a "${TEMPLATE_LOG_ERR}" >&2)
fi

if [[ "${ALLOW_CONCURRENT_QWEN3VL_JOBS:-0}" != "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  active_qwen3vl_job="$(
    squeue --noheader --user="${USER}" --states=RUNNING --format='%A|%j' |
      awk -F'|' -v current_job_id="${SLURM_JOB_ID}" \
        '$1 != current_job_id && $1 + 0 < current_job_id + 0 && $2 ~ /^qwen3vl-/ { print; exit }'
  )"
  if [[ -n "${active_qwen3vl_job}" ]]; then
    echo "Another Qwen3-VL job is already running (${active_qwen3vl_job}); refusing a concurrent socket-NCCL launch. Set ALLOW_CONCURRENT_QWEN3VL_JOBS=1 to override." >&2
    exit 2
  fi
fi

VLLM_STDOUT="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/qwen3vl/logs/vllm-${CLIENT_TEMPLATE_NAME}-${SLURM_JOB_ID:-manual}.out"
VLLM_STDERR="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/qwen3vl/logs/vllm-${CLIENT_TEMPLATE_NAME}-${SLURM_JOB_ID:-manual}.err"

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

case "${VLLM_ENABLE_EXPERT_PARALLEL}" in
  1) VLLM_EXPERT_PARALLEL_ARG="--enable-expert-parallel" ;;
  0) VLLM_EXPERT_PARALLEL_ARG="" ;;
  *)
    echo "VLLM_ENABLE_EXPERT_PARALLEL must be 0 or 1, got: ${VLLM_ENABLE_EXPERT_PARALLEL}" >&2
    exit 2
    ;;
esac

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
  ${VLLM_EXPERT_PARALLEL_ARG} \
  --enforce-eager \
  --max-model-len 48000 \
  --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
  --max-num-batched-tokens ${VLLM_MAX_NUM_BATCHED_TOKENS} \
  --distributed-timeout-seconds 1800 \
  --limit-mm-per-prompt.image 20 \
  --limit-mm-per-prompt.video 0 \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.9"


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
echo "Client max tokens: ${CLIENT_MAX_TOKENS}"
echo "Client thinking token budget: ${CLIENT_THINKING_TOKEN_BUDGET}"
echo "vLLM max sequences: ${VLLM_MAX_NUM_SEQS}"
echo "vLLM expert parallel: ${VLLM_ENABLE_EXPERT_PARALLEL}"
ls -la "${RUNPY_PATH}"
unset SSL_CERT_FILE
source ${HOST_VENV}/bin/activate
python3 "${RUNPY_PATH}" \
  --output_dir ${CLIENT_OUTPUT_DIR} \
  --template_name ${CLIENT_TEMPLATE_NAME} \
  --concurrency ${CLIENT_CONCURRENCY} \
  --max_tokens ${CLIENT_MAX_TOKENS} \
  --thinking_token_budget ${CLIENT_THINKING_TOKEN_BUDGET} \
  --server_url ${SERVER_URL} &
CLIENT_PID=$!

# Watchdog: if the vLLM server process dies or stops answering /health while the
# client is still running, kill the client immediately. Without this, a mid-run
# EngineDeadError crash (see logs/vllm-*.err) leaves the client looping through
# every remaining (seed, story) combination as an instant, guaranteed
# "Connection error" for the rest of the job — observed as two full wasted
# 60-story passes after the server died.
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
wait "${WATCHDOG_PID}" 2>/dev/null
deactivate || true
echo "Client script completed with exit code: ${CLIENT_RC}"

# shut down
echo "Terminating vLLM"
kill -15 ${SRUN_PID} 2>/dev/null || true
wait ${SRUN_PID} 2>/dev/null || true
exit ${CLIENT_RC}