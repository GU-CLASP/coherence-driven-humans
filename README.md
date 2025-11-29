# coherence-driven-humans

## Setup

Running on Alvis, use existing environment:

```
module purge
module load virtualenv/20.29.2-GCCcore-14.2.0
module load Python/3.13.1-GCCcore-14.2.0
source /mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-tacl/envs/coherence_tacl/bin/activate
```

## Visualisation

To explore individual stories with coreference annotations, use the interactive notebook or pre-generated HTML files. Run `notebooks/visualize_stories_with_coref.ipynb` or download HTML files under `examine_stories/`.


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

After generating stories with all models, process the outputs for analysis:

1. **Collect model outputs**: Gather all model-generated and human-written stories into a single JSON file.

```bash
cd scripts
python collect_data.py \
    --qwen3vl-out ../models/qwen3vl/out-qwen3vl-60stories/ \
    --internvl3-out ../models/internvl3/out-internvl3-60stories/ \
    --llama4-out ../models/llama4scout/out-llama4scout-60stories/ \
    --gpt4o-out ../models/gpt/out-gpt4o-60stories/ \
    --claude45-out ../models/claude/out-claude45-60stories/ \
    --human-large-csv ../notebooks/collected_60.csv \
    --human-original-csv ../data/vwp-acl2025-subset.csv \
    --output-json ../data/post-processing/collected_outputs.json
```

2. **Clean model outputs**: Apply model-specific cleaning functions to remove artifacts and normalise formatting.

```bash
python clean_data.py \
    --input-json ../data/post-processing/collected_outputs.json \
    --output-json ../data/post-processing/cleaned_outputs.json
```

3. **Prepare LinkAppend inputs**: Convert cleaned stories into LinkAppend-compatible format for coreference resolution.

```bash
python prepare_linkappend_inputs.py \
    --input-json ../data/post-processing/cleaned_outputs.json \
    --output-dir ../models/linkappend/data-in
```

This creates separate JSON files for each model/prompt/seed combination.

4. **Run LinkAppend coreference resolution**: Process all stories through LinkAppend to identify coreference chains.

```bash
cd ../models/linkappend
sbatch linkappend-run.slurm
```

This submits a SLURM job that processes all JSON files in `data-in/` and outputs coreference-annotated results to `data-out/`.

5. **Convert CoNLL to JSON format**: Convert LinkAppend's CoNLL output to jsonlines format for analysis.

```bash
cd ../../scripts
sbatch conll2json-corefconversion.sh
```

This processes all `.conll` files from `data-out/` subdirectories and creates jsonlines files in `data-out/conll_to_json/`.

