## 1. How to run on Arrhenius from scratch

Start with an interactive GPU allocation, then create a fresh environment inside that shell:

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
python3 -m venv ../envs/tacl-gpu
source ../envs/tacl-gpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

This creates a fresh environment at `../envs/tacl-gpu` inside the repository workspace.

## 2. How to run on Arrhenius with an existing environment

For collaborators using the current Arrhenius setup, there is already a shared environment at:

```bash
/nobackup/proj/disk/naiss2024-6-297/shared/envs/tacl-gpu
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
source /nobackup/proj/disk/naiss2024-6-297/shared/envs/tacl-gpu/bin/activate
```

If they want to recreate the environment instead of reusing it, they should follow the scratch setup in the previous section.

## 3. How to reproduce the project on a different system

This project can be reproduced on another GPU system without the exact NAISS module names, as long as the target machine provides the same capabilities:

- Linux on a compatible CPU architecture
- NVIDIA GPUs for the GPU jobs
- a CUDA-capable driver/runtime
- Python 3.13-compatible host tooling
- a way to create virtual environments
- Apptainer or another container runtime for the VLM serving jobs
- access to the external model checkpoints used by the scripts