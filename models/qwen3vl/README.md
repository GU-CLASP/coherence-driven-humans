# Qwen3-VL Model

## Start vLLM Server

```bash
sbatch qwen3vl-start.slurm
```

## Run the Script on the Same Node

Find the node where the job is running:

```bash
scontrol show job JOBID | grep -E 'BatchHost|NodeList'
```

Connect to that node:

```bash
srun --jobid=JOBID -N1 -n1 -w NODEID --cpus-per-task=1 --overlap --pty bash -l
```

## Run the Model

```bash
module purge
module load virtualenv/20.32.0-GCCcore-14.3.0
source /mimer/NOBACKUP/groups/naiss2024-6-297/vllm-environments/qwen3vl/bin/activate
python run.py \
  --csv_file ../../notebooks/collected_60.csv \
  --output_dir ./out-qwen3vl-60stories \
  --template_name original|medium|large \
  --server_url HOST:PORT
```
