# Technical details, Topic modelling experiments

## General description

We use BERTopic (Grootendorst, M., 2022) to assign topic labels to story segments for the topic switch metric.
Our objective is not to optimise a single topic model, but to get stable segment-level topic assignments for comparative analysis across systems and prompt conditions.
Accordingly, we train the topic model on texts from both prompt conditions and keep the modelling procedure fixed across analyses.

## Balanced Bootstrap Training Design

Because the long prompt condition contains three human variants per sequence, including all variants in training would over-represent long prompt human data.
To avoid this, we trained BERTopic on 10 balanced bootstrap sets and re-sampled the human long story choice across bootstraps (one human variant per sequence per bootstrap).

Each bootstrap set contains 720 stories:
- 360 from the short prompt condition
- 360 from the long prompt condition
- that is, 60 per source across 6 sources per prompt condition

Stories are split into segments using the __[SEP]__ boundaries, resulting in approximately 4.1k segments per bootstrap.

## Embeddings

We embed segments with `https://huggingface.co/Salesforce/SFR-Embedding-Mistral`.

## BERTopic Configuration

We use default BERTopic settings, with pilot runs to select a suitable **min\_topic\_size**.
We evaluated **min\_topic\_size** $\in \{10,20,30,40\}$ based on (1) the number of discovered topics, (2) the outlier rate (topic $-1$), (3) and manual inspection of topic keywords.
We set **min\_topic\_size** to 10 (default value), which produced a favourable trade-off between low outlier rate and interpretable topics (typically 82-88 topics in pilot runs).

## Topic Granularity Robustness

To eavluate robustness to topic granularity, we apply BERTopic's topic reduction and evaluate **nr\_topics** from 80 down to 5 (step size 5).

## Metric Computation

For each story, we assign a topic label to each segment under each bootstrap model and each **nr\_topics** setting, and compute the topic switch rate.
We then average topic switch rates across **nr\_topics** settings and across bootstraps.
Averaging across topic granularities reduces dependence on any single clustering resolution and results in a more stable estimate of topical progression.

## Reference

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure.
