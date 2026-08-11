# Llama-4-Scout Model (Arrhenius)

Supported template names:

- `original`
- `large`
- `original-target`
- `large-target`
- `large-upper-bound`

## Submit a run

From repo root:

```bash
cd /nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans

sbatch --export=ALL,CLIENT_TEMPLATE_NAME=original,CLIENT_OUTPUT_DIR=./models/llama4scout/out-llama4scout-60stories,CLIENT_CONCURRENCY=16 \
  models/llama4scout/llama4scout-start.slurm
```