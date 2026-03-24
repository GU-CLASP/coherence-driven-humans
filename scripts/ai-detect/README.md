# AI-detection quality control, long prompt

This code is used to screen crowd-sourced long-prompt human stories for possible AI assistance.

We fine-tuned a RoBERTa-based binary classifier initialised from `openai-community/roberta-base-openai-detector` on Hugging Face (Solaiman et al., 2019; Liu et al., 2019). The classifier is trained to distinguish human-written stories from model-generated stories in this dataset.

The training set contains 480 stories in total:
- 180 long-prompt human stories (3 seeds x 60 visual sequences)
- 300 model-generated stories (5 VLMs x 60 visual sequences, seed 42)

This setup combines all available long-prompt human texts with the corresponding model outputs from the same visual prompts.

We evaluate the detector with 5-fold GroupKFold cross-validation grouped by visual sequence (`story_id`). This keeps all stories from the same sequence in the same fold and prevents sequence-level leakage between train and validation. In practice, each evaluated story is scored by a model that did not train on that story or any other text from the same sequence.

Training uses weighted cross-entropy to handle class imbalance, with class weights 1.33 for human and 0.80 for AI. Each fold is trained for 20 epochs, and the checkpoint with best validation F1 is retained.

All filtering decisions are based on out-of-fold (OOF) predictions, meaning each story is scored only in the fold where it is in validation. These OOF scores are used as a conservative signal to flag candidate cases for removal.

## Implementation

Main training script:
- [scripts/ai-detect/finetune_roberta_pooled.py](finetune_roberta_pooled.py)

Cluster launcher:
- [scripts/ai-detect/finetune_roberta.slurm](finetune_roberta.slurm)

OOF analysis notebook:
- [scripts/ai-detect/analyze_oof_predictions.ipynb](analyze_oof_predictions.ipynb)

Data source:
- [data/post-processing/cleaned_outputs.json](../../data/post-processing/cleaned_outputs.json)

## Run

```bash
python finetune_roberta_pooled.py --prompt-type large --cv-folds 5 --epochs 20
```

## Output Files

Cross-validation metrics and predictions:
- [scripts/ai-detect/roberta-finetuned-pooled/cv/cv_metrics.json](roberta-finetuned-pooled/cv/cv_metrics.json)
- [scripts/ai-detect/roberta-finetuned-pooled/cv/oof_predictions.json](roberta-finetuned-pooled/cv/oof_predictions.json)
- [scripts/ai-detect/roberta-finetuned-pooled/cv/oof_predictions.csv](roberta-finetuned-pooled/cv/oof_predictions.csv)

Per-fold best checkpoints:
- [scripts/ai-detect/roberta-finetuned-pooled/cv](roberta-finetuned-pooled/cv)

Fold split metadata:
- [scripts/ai-detect/split_metadata](split_metadata)

Logs:
- [scripts/ai-detect/finetune_pooled.log](finetune_pooled.log)
- [scripts/ai-detect/logs](logs)

## References

- Solaiman, I., Brundage, M., Clark, J., Askell, A., Herbert-Voss, A., Wu, J., Radford, A., Krueger, G., Kim, J. W., Kreps, S., et al. (2019). Release strategies and the social impacts of language models. arXiv:1908.09203.
- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., and Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv:1907.11692.
