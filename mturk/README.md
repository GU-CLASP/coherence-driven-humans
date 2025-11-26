# MTurk Data Collection

Scripts and tools for running Amazon Mechanical Turk (AMT) data collection tasks

## Setup

Activate the environment:
```bash
module load virtualenv/20.26.2-GCCcore-13.3.0
source /mimer/NOBACKUP/groups/naiss2024-6-297/envs/amt/bin/activate
```

**Important:** 
- Use `--prod` flag to publish HITs to production (real money)
- Without `--prod`, HITs are published to the sandbox environment for testing
- Create `scripts/config.json` with your AWS credentials:
  ```bash
  cp scripts/config.json.example scripts/config.json
  # Then edit config.json and add your AWS access key and secret key
  ```

## Dependencies

Required Python packages:
- boto3 (AWS SDK)
- jinja2 (templating)
- pandas (for prepare_amt_tasks.ipynb)
- tqdm (for prepare_amt_tasks.ipynb)

## Qualifications

### Sandbox qualifications
- describers: `3TSO6UD8GEFI6WGMKI7TODSYHRWWQE`
- masters: `2ARFPLSP75KLA8M8DH1HTEQVJT3SY6`

### Production qualifications
- describers: `3VIW7BLKQFCIG5E6CXLL1HDYHN07PT`
- masters: `2F1QJWKUDD8XADTFD2Q0G6UTO95ALH`

## Workflow

### 0. Generate HTML files for HITs
Run the `prepare_amt_tasks.ipynb` notebook to generate HTML files for each story from the sampled data. This creates files in `data/input_amt_samples_20/` and `data/input_amt_samples_40/`

### 1. Launch HITs
```bash
python scripts/launch_hits.py --hit_properties_file data/properties/deploy-2025-10-08.json --html_dir data/input_amt_samples_20 --prod
```
- Publishes one HIT every 30 minutes to keep them visible on the first page of AMT
- Creates a timestamped folder in `data/results/` with `hit_ids.txt`
- for 20 stories use `data/input_amt_samples_20`, for 40 stories use `data/input_amt_samples_40`


