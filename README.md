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

