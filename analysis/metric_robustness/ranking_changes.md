# Are there any changes in model ranking across metrics when removing possibly AI-generated texts?

Baseline uses 60 stories.
Pangram columns use 58 stories. RoBERTa columns use 53 stories.

### scenarios

| scenario/method | Excluded IDs (from 60) |
|---|---|
| pangram | 5540, 9418 |
| roberta | 1214, 3761, 3891, 5047, 5540, 7195, 10499 |

## Pangram

### original

| metric | model | 60_rank | 58_rank | 58_rank_shift |
|---|---|---:|---:|---:|
| topic_switch_rate | claude45 | 4 | 5 | 1 |
| topic_switch_rate | internvl3 | 5 | 4 | -1 |

### large

| metric | model | 60_rank | 58_rank | 58_rank_shift |
|---|---|---:|---:|---:|
| char_coherence | internvl3 | 4 | 5 | 1 |
| char_coherence | claude45 | 5 | 4 | -1 |
| groovist | internvl3 | 3 | 4 | 1 |
| groovist | human | 4 | 3 | -1 |

## RoBERTa

### original

| metric | model | 60_rank | 53_rank | 53_rank_shift |
|---|---|---:|---:|---:|
| char_coherence | internvl3 | 2 | 3 | 1 |
| char_coherence | gpt4o | 3 | 2 | -1 |
| discourse_diversity | qwen3vl | 5 | 6 | 1 |
| discourse_diversity | internvl3 | 6 | 5 | -1 |
| MCC | gpt4o | 1 | 2 | 1 |
| MCC | qwen3vl | 2 | 1 | -1 |
| ncs_geometric | gpt4o | 2 | 3 | 1 |
| ncs_geometric | qwen3vl | 3 | 2 | -1 |
| topic_switch_rate | gpt4o | 3 | 4 | 1 |
| topic_switch_rate | claude45 | 4 | 3 | -1 |

### large

| metric | model | 60_rank | 53_rank | 53_rank_shift |
|---|---|---:|---:|---:|
| char_coherence | internvl3 | 4 | 5 | 1 |
| char_coherence | claude45 | 5 | 4 | -1 |
| ncs_arithmetic | internvl3 | 4 | 5 | 1 |
| ncs_arithmetic | claude45 | 5 | 4 | -1 |
| ncs_geometric | internvl3 | 4 | 5 | 1 |
| ncs_geometric | claude45 | 5 | 4 | -1 |
| topic_switch_rate | qwen3vl | 5 | 6 | 1 |
| topic_switch_rate | claude45 | 6 | 5 | -1 |