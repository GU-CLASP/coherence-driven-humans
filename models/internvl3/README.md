# InternVL3 Model (Arrhenius)

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

sbatch --export=ALL,CLIENT_TEMPLATE_NAME=large,CLIENT_OUTPUT_DIR=./models/internvl3/out-internvl3-60stories,CLIENT_CONCURRENCY=16 \
  models/internvl3/internvl3-start.slurm
```