# Topic Modelling

A guide to replicate topic modelling experiments.

## Core Files

- `scripts/topic_modelling/bertopic-train.py`
- `scripts/topic_modelling/bertopic-train.sh`
- `scripts/topic_modelling/bertopic-test.py`
- `scripts/topic_modelling/bertopic-test.sh`
- `scripts/topic_modelling/technical_details.md`
- `notebooks/create_inputs_for_berttopic.ipynb`
- `analysis/topic_modelling_profiles.ipynb`

Useful companion notebooks:

- `notebooks/bertopic_inference_analysis.ipynb`
- `notebooks/pilot_discovery_summary.ipynb`

## Inputs Used By Current Scripts

Both train and test scripts now use:

- `/mimer/NOBACKUP/groups/naiss2024-6-297/cache/bertopic_data/`

Expected files:

- `all_stories_texts.json`
- `all_stories_metadata.csv`
- `balanced_train_sets/` with
	- `train_texts_bootstrap_00.json` to `train_texts_bootstrap_09.json`
	- `train_metadata_bootstrap_00.csv` to `train_metadata_bootstrap_09.csv`
	- `human_large_seed_choices.json`

## What The Pipeline Does

1. Build BERTopic inputs from the story dataset.
2. Train BERTopic on 10 balanced bootstrap sets.
3. Run inference for each bootstrap and topic-reduction setting.
4. Compute topic-switch metrics from sentence-level assignments.

Defaults in current runs:

- `min_topic_size=10`
- topic reduction sweep: `nr_topics` from 80 down to 5 (step 5)

## Output Locations

External cache/output root:

- `/mimer/NOBACKUP/groups/naiss2024-6-297/cache/bertopic_bootstrapped/`

Main run example:

- `/mimer/NOBACKUP/groups/naiss2024-6-297/cache/bertopic_bootstrapped/full_training_10/`

Key artifacts:

- `models/bootstrap_XX/topics_YY/` (trained BERTopic models)
- `inference/bootstrap_XX/topics_YY/` with
	- `full_results.csv`
	- `sentence_topics.npy`
	- `sentence_probs.npy`
	- `inference_info.json`

Analysis output:

- `analysis/analysis_data/topic_modelling/`

## Run Commands

From the repo root:

```bash
# Optional pilot discovery
PILOT_DISCOVERY=1 MIN_TOPIC_SIZE=10 sbatch scripts/topic_modelling/bertopic-train.sh

# Full training
MIN_TOPIC_SIZE=10 sbatch scripts/topic_modelling/bertopic-train.sh

# Inference
MIN_TOPIC_SIZE=10 sbatch scripts/topic_modelling/bertopic-test.sh
```