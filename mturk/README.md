# MTurk Data Collection

Scripts and tools for running Amazon Mechanical Turk (AMT) data collection tasks

## Setup

activate the environment:
```bash
module load virtualenv/20.26.2-GCCcore-13.3.0
source /mimer/NOBACKUP/groups/naiss2024-6-297/envs/amt/bin/activate
```

**Important:** use `--prod` flag to publish HITs to production (real money). Without this flag, HITs are published to the sandbox environment for testing.

## Qualifications

### Sandbox qualifications
- describers: `3TSO6UD8GEFI6WGMKI7TODSYHRWWQE`
- masters: `2ARFPLSP75KLA8M8DH1HTEQVJT3SY6`

### Production qualifications
- describers: `3VIW7BLKQFCIG5E6CXLL1HDYHN07PT`
- masters: `2F1QJWKUDD8XADTFD2Q0G6UTO95ALH`

## Workflow

### 0. Generate HTML files for HITs
Run the `prepare_amt_tasks.ipynb` notebook to generate HTML files for each story from the sampled data

### 1. Launch HITs
```bash
python scripts/launch_hits.py --hit_properties_file data/properties/deploy-2025-10-08.json --html_dir meta/input_amt_samples_20 --prod
```
Publishes one HIT every 30 minutes to keep them visible on the first page of AMT
