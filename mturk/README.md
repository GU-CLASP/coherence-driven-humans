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
- For 20 stories use `data/input_amt_samples_20`, for 40 stories use `data/input_amt_samples_40`

### 2. Monitor submissions and assign qualifications

**Option A: Monitor a specific run**
```bash
python scripts/monitor_and_qualify.py --rejected --prod --hit_ids_file data/results/True-publish-YYYY-MM-DD-HH-MM/hit_ids.txt
```
- Runs continuously (checks every 30 seconds)
- Monitors only the specific run you launched
- Collects submitted assignments and saves them to `results.csv`
- Automatically assigns blocking qualifications to workers who exceed 6 tasks within this run
- Saves `worker_counts.csv` tracking tasks per worker
- Use `--rejected` to include rejected assignments in counts

**Option B: Monitor all runs globally**
```bash
python scripts/monitor_workers.py --prod
```
- Runs continuously (checks every 30 seconds)
- Monitors ALL run folders in `data/results/`
- Counts total tasks per worker across all runs
- Automatically assigns blocking qualifications to workers who exceed 6 tasks globally
- Use this to enforce limits across multiple concurrent runs

### 3. Approve/reject assignments and pay bonuses
```bash
python scripts/approve_and_pay.py --prod --results_file data/results/True-publish-YYYY-MM-DD-HH-MM/results.csv
```
- Reviews each submission
- Shows the text descriptions, word count, and survey responses
- Prompts to approve or reject each assignment
- Automatically calculates and offers to pay bonuses for descriptions exceeding 85 words ($0.04 per extra word)
- Saves decisions to `decisions.csv`

## Additional Scripts

### Manually block specific workers
Edit `scripts/assign_qualifications.py` to add worker IDs to the `WORKERS_TO_BLOCK` list, then run:
```bash
python scripts/assign_qualifications.py --prod
```

### Create story-specific qualifications
If you need to block specific workers from specific stories:
```bash
# 1. Edit STORY_IDS list in create_story_qualifications.py
python scripts/create_story_qualifications.py --prod

# 2. This creates qualifications and saves IDs to story_qualification_ids.txt
```


