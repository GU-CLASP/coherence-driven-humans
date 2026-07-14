## 1. How to run on Arrhenius from scratch

Start with an interactive GPU allocation, then create a fresh environment inside that shell.
Use a dedicated arm64 venv for model-serving jobs, separate from the older shared env used by other project jobs:

```bash
interactive \
	-A naiss2025-5-551-gpu \
	-p gpu \
	--gpus=1 \
	-n 1 \
	-c 8 \
	-t 01:00:00
```

Inside the interactive GPU shell, load the GPU build environment and create the venv before installing Python packages:

```bash
module purge
module load GPU/buildenv-nvhpc/25.9-cu13.0
python3 -m venv /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64
source /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install openai pandas tqdm nltk pyarrow
```

This creates a fresh environment at `/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64`.

## 2. How to create a CPU-node environment (for notebooks and preprocessing)

Use a separate CPU-only environment for notebook work and lightweight preprocessing.
Do this on the login/CPU side, not inside a GPU allocation.

```bash
module purge
module load buildenv-intel/2025b-eb

python3 -m venv /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/jupyter-cpu
source /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/jupyter-cpu/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install pandas tqdm nltk pyarrow ipykernel
```

This creates:

`/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/jupyter-cpu`

Recommended usage split:

- `models-arm64`: GPU-node model serving and generation scripts.
- `jupyter-cpu`: CPU-node notebook analysis and data inspection.

## 3. How to run on Arrhenius with an existing environment

For collaborators using the current Arrhenius setup, the dedicated arm64 model-serving environment should live at:

```bash
/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64
```

If that environment is available, they can skip creating a new venv and just activate it after entering an interactive GPU allocation:

```bash
interactive \
	-A naiss2025-5-551-gpu \
	-p gpu \
	--gpus=1 \
	-n 1 \
	-c 8 \
	-t 01:00:00

module purge
module load GPU/buildenv-nvhpc/25.9-cu13.0
source /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64/bin/activate
```

If they want to recreate the environment instead of reusing it, they should follow the scratch setup in the previous section.

## 4. How to run the model scripts on Arrhenius

For the model-generation scripts, use a separate arm64 environment created on an Arrhenius GPU node.
Keep the roles separate: use the older `tacl-gpu` environment for general project work, and use `models-arm64` only for model-serving scripts.
This keeps the model-serving dependencies isolated from the shared general-purpose environment.

Create the environment from scratch:

```bash
interactive \
	-A naiss2025-5-551-gpu \
	-p gpu \
	--gpus=1 \
	-n 1 \
	-c 8 \
	-t 01:00:00

module purge
module load GPU/buildenv-nvhpc/25.9-cu13.0
python3 -m venv /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64
source /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install openai pandas tqdm nltk pyarrow
```

If the environment already exists, just activate it after entering an interactive GPU allocation:

```bash
interactive \
	-A naiss2025-5-551-gpu \
	-p gpu \
	--gpus=1 \
	-n 1 \
	-c 8 \
	-t 01:00:00

module purge
module load GPU/buildenv-nvhpc/25.9-cu13.0
source /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/envs/models-arm64/bin/activate
```

## 5. How to reproduce the project on a different system

This project can be reproduced on another GPU system without the exact NAISS module names, as long as the target machine provides the same capabilities:

- Linux on a compatible CPU architecture
- NVIDIA GPUs for the GPU jobs
- a CUDA-capable driver/runtime
- Python 3.13-compatible host tooling
- a way to create virtual environments
- Apptainer or another container runtime for the VLM serving jobs
- access to the external model checkpoints used by the scripts