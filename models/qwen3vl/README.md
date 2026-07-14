# Qwen3-VL-235B multi-node serving on Arrhenius - runbook

This is the practical runbook for serving Qwen/Qwen3-VL-235B-A22B-Thinking
with vLLM across 4 Arrhenius GPU nodes, then running run.py for story generation.

If you only need to launch a run, go straight to Quick start.

---

## Quick start

Prerequisites (one-time setup) are in Section 1. If those are already done,
run this from the project root:

1) Submit a large-target run (for example)

  sbatch --export=ALL,CLIENT_TEMPLATE_NAME=large-target,CLIENT_OUTPUT_DIR=./models/qwen3vl/out-qwen3vl-60stories-large-target,CLIENT_CONCURRENCY=16 models/qwen3vl/qwen3vl-arrhenius.slurm

2) Watch status

  squeue --me
  tail -f models/qwen3vl/logs/qwen3vl-large-target-<jobid>.out

3) Confirm files are landing

  find models/qwen3vl/out-qwen3vl-60stories-large-target -name '*.parquet' | wc -l

If the count stays 0, check the launcher log first. The most common reason is
that the client has not started yet or failed before calling run.py.

---

## 0. Overview and architecture

- **Model**: `Qwen/Qwen3-VL-235B-A22B-Thinking` (MoE, ~439 GB bf16, 97 safetensors).
  Too big for one node (GH200 = 96 GB/GPU × 4 = 384 GB), so **multi-node is required**.
- **Serving**: vLLM `v0.25.0`, from an **arm64 Apptainer container** (`vllm-v0.25.0-aarch64.sif`).
- **Distribution**: vLLM **multiprocessing** multi-node backend (`--distributed-executor-backend mp`).
  **No Ray** — v0.25.0 removed Ray from the image; MP multi-node works via
  `--nnodes / --node-rank / --master-addr / --master-port`, worker ranks use `--headless`.
- **Parallelism**: **tensor parallel = 16, pipeline parallel = 1** (all 16 GPUs, one TP group).
  Pipeline parallel is intentionally NOT used (see Gotcha #6).
- **Execution mode**: **`--enforce-eager`** (no `torch.compile` / CUDA graphs). Required for
  stable multi-node decode here — see Gotcha #9.
- **Throughput**: driven by concurrency, not single-stream speed. Server --max-num-seqs 16
  + client --concurrency 16 batches 16 stories at once. This is what makes it fast
  (see Gotcha #10): the 60-story run takes about 38 minutes instead of about
  8.5 hours.
- **Client**: `run.py` runs on the **host** (not the container) in the `models-arm64` venv,
  talks to the server over HTTP (OpenAI-compatible API), fires stories concurrently.

### Filesystem layout (Arrhenius)

| Purpose | Path |
|---|---|
| Project root | `/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans` |
| This workdir | `.../models/qwen3vl` |
| Container image (SIF) | `.../models/vllm-v0.25.0-aarch64.sif` |
| Client venv (host) | `.../envs/models-arm64` |
| Container extra libs | `.../envs/vllm-extras` |
| HF model cache | `/nobackup/proj/disk/naiss2024-6-297/shared/hf-home` |
| Prompts | `.../data/prompts/prompt-original-target-{w,wo}-names.txt` |
| Story images | `.../data/sampled_60/images` |
| Character images | `.../data/sampled_60/characters` |
| Story metadata / word counts | `.../data/sampled_60/sampled_60_stories.json` |
| Character-name CSV | `.../data/vwp-acl2025-subset.csv` |

> Login node is amd64, container and venvs are arm64.
> You cannot run the SIF or import compiled packages from those venvs on arrhenius1.
> Always validate container and venv behavior on a GPU node.

---

## 1. One-time setup (persists on shared disk)

Container steps must run on an **arm64 GPU node**. Grab an interactive allocation:

```bash
interactive -A naiss2025-5-551-gpu -p gpu --gpus=1 -n 1 -c 8 -t 00:30:00
```

### 1a. Build the arm64 vLLM container (if the SIF doesn't exist)

Definition file `../vllm-0.25.def`:

```
Bootstrap: docker
From: vllm/vllm-openai:v0.25.0-aarch64-cu129-ubuntu2404

%environment
    export PYTHONUNBUFFERED=1

%runscript
    exec vllm "$@"
```

Build (on an arm64 node):

```bash
apptainer build /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/vllm-v0.25.0-aarch64.sif \
  /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/vllm-0.25.def
```

### 1b. Install the CUDA-13 libs the container needs

The container's `torchcodec` is CUDA-13 built and loads `libnvrtc.so.13` /
`libcudart.so.13` at `vllm serve` startup, but the bundled PyTorch is cu12.9 and does
**not** ship those. Install into `vllm-extras` with the **container's** Python.
`/nobackup` is read-only in the container, so create the dir on the host first and
bind it rw for the pip step:

```bash
SIF=/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/vllm-v0.25.0-aarch64.sif
EXTRA=/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/vllm-extras

mkdir -p "$EXTRA"
apptainer exec --bind "$EXTRA:$EXTRA" "$SIF" \
  python3 -m pip install --target="$EXTRA" nvidia-cuda-nvrtc nvidia-cuda-runtime

ls "$EXTRA"/nvidia/cu13/lib/libnvrtc.so.13 "$EXTRA"/nvidia/cu13/lib/libcudart.so.13
```

### 1c. Create the host client venv (`models-arm64`) for run.py

Runs on the host (GPU node), never inside the container; only HTTP-client deps:

```bash
module purge
module load GPU/buildenv-nvhpc/25.9-cu13.0
python3 -m venv /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64
source /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install openai pandas tqdm nltk pyarrow
deactivate
```

### 1d. Create the HF cache dir (must be writable)

```bash
mkdir -p /nobackup/proj/disk/naiss2024-6-297/shared/hf-home
```

### 1e. Pre-download the model (recommended)

Compute nodes may lack internet. Download once from a shell that has it, into the
shared cache, using a throwaway venv (don't pollute `models-arm64`):

```bash
python3 -m venv /nobackup/proj/disk/naiss2024-6-297/personal/<you>/hf-dl-venv
source /nobackup/proj/disk/naiss2024-6-297/personal/<you>/hf-dl-venv/bin/activate
pip install -U "huggingface_hub[cli]"
export HF_HOME=/nobackup/proj/disk/naiss2024-6-297/shared/hf-home
export HUGGING_FACE_HUB_TOKEN=$(tr -d '\r\n' < /nobackup/proj/disk/naiss2024-6-297/personal/<you>/config.json)
hf download Qwen/Qwen3-VL-235B-A22B-Thinking
# verify: no *.incomplete files, ~439 GB, 97 safetensors shards
```

---

## 2. Verify prerequisites (on a GPU node)

```bash
SIF=/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/vllm-v0.25.0-aarch64.sif
EXTRA=/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/vllm-extras

ls "$EXTRA"/nvidia/cu13/lib/libnvrtc.so.13 "$EXTRA"/nvidia/cu13/lib/libcudart.so.13
ls -d /nobackup/proj/disk/naiss2024-6-297/shared/hf-home/hub/models--Qwen--Qwen3-VL-235B*

apptainer exec --nv --bind "$EXTRA:$EXTRA" "$SIF" bash -lc \
  "export LD_LIBRARY_PATH=$EXTRA/nvidia/cu13/lib:\$LD_LIBRARY_PATH; vllm serve --help=all 2>&1" \
  | grep -iE 'nnodes|node-rank|headless|master-addr|master-port'

/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64/bin/python3 \
  -c "import openai, pandas, tqdm, nltk, pyarrow; print('client deps OK')"
```

---

## 3. Run

Default run from models/qwen3vl:

```bash
cd /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/models/qwen3vl
sbatch qwen3vl-arrhenius.slurm
```

Recommended run from project root with explicit template and output dir:

```bash
cd /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans
sbatch --export=ALL,CLIENT_TEMPLATE_NAME=large-target,CLIENT_OUTPUT_DIR=./models/qwen3vl/out-qwen3vl-60stories-large-target,CLIENT_CONCURRENCY=16 models/qwen3vl/qwen3vl-arrhenius.slurm
```

`qwen3vl-arrhenius.slurm` does, in order:
1. Computes head node IP + free ports (API server + torch.distributed).
2. Truncates `vllm.out` / `vllm.err` (so the readiness check can't match stale output
   from a previous run — see Gotcha #10), then `srun` launches `vllm serve` on all 4
   nodes; rank 0 hosts the API server, ranks 1-3 run `--headless`. Backend `mp`,
   `tp=16`, `pp=1`, `--enforce-eager`, `--max-num-seqs 16`.
3. Waits until the server is actually ready by polling its HTTP `/health` endpoint
   (`curl -sf http://$HEAD_IP:$API_PORT/health`), not just grepping the log.
4. Runs `run.py` on the **host** (in `models-arm64`), `--concurrency 16`
   (matches server `--max-num-seqs 16`).
5. Shuts down the server.

Nothing needs re-installing between runs.

---

## 4. Monitor progress

```bash
squeue --me
sacct -j <jobid> --format=JobID,JobName%22,State,Elapsed,ExitCode

# launcher log (first place to look)
tail -f models/qwen3vl/logs/qwen3vl-<template>-<jobid>.out

# server bring-up and engine health
cd models/qwen3vl
tail -f vllm.err vllm.out

# check decode throughput and active requests
grep 'Avg generation throughput' vllm.out | tail

# output files (bursty writes are normal with concurrency)
find <output_dir>/prompt-<template>-outputs -name '*.parquet' | wc -l
```

Healthy signs:
- launcher log shows Server started successfully then Starting client
- vllm.out shows non-zero Avg generation throughput and Running: 16 reqs
- parquet count increases in bursts

Unhealthy signs:
- launcher log repeats Waiting... for too long
- no Starting client line
- repeated No available shared memory broadcast block found in 60 seconds

`vllm.out` / `vllm.err` are overwritten each run; `logs/%x-%j.*` are per-job history.

---

## 5. Key files

| File | Role |
|---|---|
| `qwen3vl-arrhenius.slurm` | Launcher: starts the multi-node server + runs the client |
| `run.py` | Story-generation client (HTTP → OpenAI API). Host, in `models-arm64` |
| `../vllm-0.25.def` | Apptainer definition to build the arm64 SIF |
| `README.md` | This document |

`run.py` behavior:
- `--csv_file` is **optional**. If omitted, story IDs come from the target source
  (`sampled_60_stories.json` for `original*` templates).
- Any template passed via `--template_name` auto-selects `-w-names` / `-wo-names`
  per story based on whether that story has character images (for example,
  `original-target`, `large-target`, `large-upper-bound`).
- `--concurrency N` sends N stories at once (thread pool; client is thread-safe, each
  story writes its own parquet). Pair with server `--max-num-seqs N`.

### Inspecting generated stories

Each story is one `.parquet` row with columns: `story_id`, `num_story_images`,
`num_character_images`, `target_words`, `instruction_text`, `model_output`, `seed`,
`elapsed_time`.

Important: this is a Thinking model. model_output often contains a reasoning trace
before the final story. Many samples include a literal </think> separator, but some
do not.

Recommended: use scripts/collect_and_clean_data.py for post-processing. It handles:
- standard extraction after </think>
- fallback extraction when </think> is missing
- normalization of [SEP] spacing

Read parquet **on the login node** (amd64) — the arm64 client venv won't import there.
Use the SciPy-bundle module and add `pyarrow` once to your user site:

```bash
module load Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
python3 -m pip install --user pyarrow      # one-time
python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("out-qwen3vl-60stories/prompt-original-target-outputs/345.parquet")
out = str(df.iloc[0]["model_output"])
story = out.split("</think>", 1)[-1].strip()   # final story only
print(story)
PY
```

---

## 6. Gotchas we hit (and the fixes)

1. **Login node is amd64, container/venvs are arm64** → false "not found" errors on
   `arrhenius1`. Test on a GPU node.
2. **Container has no `ray`** (removed in v0.25.0). Use MP multi-node instead of Ray.
3. **torchcodec needs CUDA-13 libs** not shipped with the cu12.9 torch. Install
   `nvidia-cuda-nvrtc` + `nvidia-cuda-runtime` into `vllm-extras`, add to `LD_LIBRARY_PATH`.
4. **`/nobackup` is read-only inside the container.** Anything the container writes to
   (HF cache, pip `--target`) needs an explicit `--bind` (dir created on host first).
5. **HF cache dir didn't exist** → "Read-only file system" on download. Create it.
6. **Pipeline parallel (`pp>1`) crashes Qwen3-VL-MoE** at init with
   "No model architectures are specified" (vLLM bug #43271; fix PR #43272 not in v0.25.0).
   Use `tp=16, pp=1`. Cross-node NCCL for TP works fine here.
7. **`run.py` had hardcoded `/mimer/...` (Alvis) paths** → repointed to `/nobackup`
   equivalents (images, characters, prompts, VWP CSV).
8. **`default_target_source_for_template` walked up one dir too many** (`../../..` →
   `/shared`). Fixed to the absolute `coherence-driven-humans` root.
9. **Decode deadlock → must use `--enforce-eager`** (vLLM issue #30682). With CUDA
   graphs on, the server starts fine, generates ~1 token, then hangs at 0 tok/s while
   `EngineCore` spams `No available shared memory broadcast block found in 60 seconds`.
   Trigger: the vision Triton kernels (`_bilinear_pos_embed_kernel`,
   `_compute_slot_mapping_kernel`) JIT-compile mid-inference on only some ranks, which
   desyncs the cross-node NCCL collective under CUDA graphs. `--enforce-eager` gives
   every rank the same deterministic path → stable sustained decode. No container change.
10. **Launcher readiness race (premature client start).** The old wait loop grepped
    `vllm.out`/`vllm.err` for "Application startup complete", but those files still held
    that line from the **previous** run, so the client launched against a server that
    was only partially loaded → all 60 stories `Connection error`, job "COMPLETED" in
    ~2 min. Fix: **truncate the logs before `srun`** *and* poll the real HTTP `/health`
    endpoint instead of trusting a log string. Symptom to recognize: the batch `.out`
    jumps straight to "Server started successfully" with no "Waiting... (n/180)" lines.

---

## 7. References (issues, PRs, docs we relied on)

vLLM issues / PRs:
- **Ray removed from the container image** (why MP multi-node instead of Ray):
  https://github.com/vllm-project/vllm/issues/38113
- **Qwen3-VL-MoE crashes with pipeline parallelism** ("No model architectures are
  specified") — the reason we use `tp=16, pp=1`:
  https://github.com/vllm-project/vllm/issues/43271
- **Fix PR for the above** (not yet in v0.25.0; needed only if you want `pp>1`):
  https://github.com/vllm-project/vllm/pull/43272
- **Multi-node decode deadlock / "No available shared memory broadcast block"** — why we
  use `--enforce-eager`:
  https://github.com/vllm-project/vllm/issues/30682
- Qwen3-VL-235B support tracking issue:
  https://github.com/vllm-project/vllm/issues/25582

vLLM docs:
- Parallelism & scaling (Ray vs. MultiProcessing multi-node):
  https://docs.vllm.ai/en/latest/serving/parallelism_scaling/
- `vllm serve` CLI reference (v0.25.0) — `--headless`, `--nnodes`, `--node-rank`,
  `--master-addr`, `--master-port`, `ParallelConfig`:
  https://docs.vllm.ai/en/v0.25.0/cli/serve/
- Distributed serving overview:
  https://docs.vllm.ai/en/latest/serving/distributed_serving.html
- Distributed serving (direct multi-node section):
  https://docs.vllm.ai/en/v0.8.0/serving/distributed_serving.html#running-vllm-on-multiple-nodes

Ray on SLURM (background; not used in the final setup):
- https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html

External libraries pulled in as fixes:
- `nvidia-cuda-nvrtc`, `nvidia-cuda-runtime` (PyPI) — provide `libnvrtc.so.13` /
  `libcudart.so.13` for the container's CUDA-13 `torchcodec`.
- `huggingface_hub[cli]` — `hf download` for pre-fetching the model.
