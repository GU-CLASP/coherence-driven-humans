# MTurk Data Collection

Scripts and tools for running Amazon Mechanical Turk (AMT) data collection tasks.

## Data Collection Overview

We collected human-generated descriptions for 60 visual story sequences through Amazon Mechanical Turk (AMT). Each sequence was described by three different crowd workers, resulting in 180 descriptions in total. Each worker could describe a given story only once.

### Recruitment

Participation was restricted to workers meeting the following criteria:

1. Master qualification status on Amazon Mechanical Turk
2. Residence in an English-speaking country (US, UK, Canada, Ireland, Australia, New Zealand)
3. Approval rate ≥ 95%
4. At least 500 previously approved AMT tasks

Data collection involved 55 unique workers, who completed an average of 3.27 stories each (SD = 1.86, range = 1–8).

### Compensation

- **Base reward:** $4.00 per story
- **Bonus:** $0.04 per word beyond the required minimum of 85 words
- **Time limit:** 30 minutes per task

On average, descriptions contained 162.46 words (SD = 80.46, range = 85–487), corresponding to an average payment of $10.50 per submission. Total compensation to workers was $1,889.68, with an additional $413.94 in platform fees (20% on base reward and bonuses, plus 5% Masters qualification surcharge), for a total data-collection cost of **$2,303.62**.

### Worker and description distributions


[Worker distribution](../figures/worker_distribution.pdf)

[Word count distribution](../figures/word_count_distribution.pdf)

[Task interface](./figures/mturk_instructions.png)

[Detailed instructions](./figures/mturk_detailed_instructions.png)


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


