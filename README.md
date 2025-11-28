# coherence-driven-humans

## Setup

Running on Alvis, use existing environment:

```
module purge
module load virtualenv/20.29.2-GCCcore-14.2.0
module load Python/3.13.1-GCCcore-14.2.0
source /mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-tacl/envs/coherence_tacl/bin/activate
```

### Prepare data

Download full images and images of characters for 60 visual sequences by running the following script:

```
python download_data.py --csv-file ./vwp-acl2025-subset.csv --output-dir ./sampled_60
```

### Data collection on MTurk

See `mturk/README.md` for setup instructions and workflow.

### Running models and generating stories

We generated stories for the collected human descriptions using 5 models:
- **InternVL3-78B** - See `models/internvl3/README.md`
- **Qwen3-VL-235B** - See `models/qwen3vl/README.md`
- **Claude-4.5-Sonnet** - See `models/claude/claude45.ipynb`
- **GPT-4o** - See `models/gpt/gpt4o.ipynb`
- **Llama-4-Scout** - See `models/llama4scout/README.md`

Each model folder contains the necessary scripts and instructions for generating stories.

### Post-processing

