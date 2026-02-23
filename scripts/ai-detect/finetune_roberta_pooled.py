#!/usr/bin/env python3
"""
Fine-tune a pooled human-vs-AI RoBERTa classifier with GroupKFold by sequence.

180 human texts (3 seeds x 60 sequences) + 300 AI texts (seed 42, 5 models x 60
sequences) = 480 total. All texts from the same visual sequence are kept together
in the same fold. Uses weighted cross-entropy to handle the 3:5 class imbalance.

Usage:
    python finetune_roberta_pooled.py --prompt-type large --cv-folds 5
"""

import sys
import argparse
import os
import json
import glob
import shutil
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("finetune_pooled.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

MODEL_NAME = "openai-community/roberta-base-openai-detector"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pooled human-vs-AI RoBERTa detector with GroupKFold."
    )
    parser.add_argument(
        "--prompt-type",
        choices=["original", "large"],
        default="large",
        help="Which prompt_type subset to train on.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of GroupKFold folds (default 5).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs per fold.",
    )
    parser.add_argument(
        "--ai-seed",
        type=int,
        default=42,
        help="Which seed to use for AI-generated stories (default 42).",
    )
    return parser.parse_args()


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class weights to the CE loss."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_pooled_data(prompt_type: str, ai_seed: int):
    """
    Return parallel lists: texts, labels (0=human, 1=AI), groups (story_id),
    and meta dicts.  AI texts are filtered to *ai_seed* only.
    """
    data_path = "/mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans/data/post-processing/cleaned_outputs.json"

    with open(data_path, "r") as f:
        raw = json.load(f)

    # Filter prompt type
    filtered = [d for d in raw if d["prompt_type"] == prompt_type]
    logger.info(f"Total records: {len(raw)}, {prompt_type}: {len(filtered)}")

    texts, labels, groups, meta = [], [], [], []

    for item in filtered:
        model_type = item["model_type"]
        is_human = model_type == "human"

        # For AI, keep only the requested seed
        if not is_human and item["seed"] != ai_seed:
            continue

        text = item["cleaned_model_output"].replace("[SEP] ", "")
        texts.append(text)
        labels.append(0 if is_human else 1)
        groups.append(item["story_id"])
        meta.append(
            {
                "story_id": item["story_id"],
                "seed": item["seed"],
                "model_type": model_type,
                "prompt_type": item["prompt_type"],
            }
        )

    n_human = sum(1 for l in labels if l == 0)
    n_ai = sum(1 for l in labels if l == 1)
    n_seq = len(set(groups))
    logger.info(
        f"Pooled dataset: {n_human} human + {n_ai} AI = {len(texts)} total, "
        f"{n_seq} unique sequences"
    )

    ai_models = Counter(m["model_type"] for m in meta if m["model_type"] != "human")
    logger.info(f"AI model breakdown: {dict(ai_models)}")

    return texts, labels, groups, meta


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def per_model_eval(trainer, val_texts, val_labels, val_meta, tokenizer, fold_tag):
    """
    Break down detection accuracy by model type (human, claude45, gpt4o, etc.)
    on the validation fold. Also collects out-of-fold (OOF) predictions with
    probabilities for every text, so we can later inspect which specific
    human texts got misclassified as AI.

    Returns (per_model_results, oof_predictions_list).
    """
    from collections import defaultdict
    import scipy.special as sp

    buckets = defaultdict(lambda: {"preds": [], "labels": []})
    ds = TextClassificationDataset(val_texts, val_labels, tokenizer)
    output = trainer.predict(ds)
    logits = output.predictions                       # (N, 2)
    probs = sp.softmax(logits, axis=1)                # (N, 2)
    preds = np.argmax(logits, axis=1)

    # Build per-text prediction records
    oof_predictions = []
    for i, (pred, label, m) in enumerate(zip(preds, val_labels, val_meta)):
        mtype = m["model_type"]
        buckets[mtype]["preds"].append(pred)
        buckets[mtype]["labels"].append(label)
        oof_predictions.append({
            "fold": fold_tag,
            "story_id": m["story_id"],
            "seed": m["seed"],
            "model_type": mtype,
            "true_label": int(label),
            "predicted_label": int(pred),
            "prob_human": float(probs[i, 0]),
            "prob_ai": float(probs[i, 1]),
        })

    results = {}
    for mtype, data in sorted(buckets.items()):
        p = np.array(data["preds"])
        l = np.array(data["labels"])
        acc = accuracy_score(l, p)
        prec, rec, f1, _ = precision_recall_fscore_support(
            l, p, average="binary" if mtype != "human" else "binary",
            pos_label=1 if mtype != "human" else 0,
            zero_division=0,
        )
        results[mtype] = {
            "n": len(l),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        if mtype == "human":
            fpr = float(np.mean(p == 1))
            results[mtype]["false_positive_rate"] = fpr
            logger.info(f"  {mtype}: n={len(l)}, acc={acc:.4f}, FPR={fpr:.4f}")
        else:
            det = float(np.mean(p == 1))
            results[mtype]["detection_rate"] = det
            logger.info(f"  {mtype}: n={len(l)}, acc={acc:.4f}, det_rate={det:.4f}")

    return results, oof_predictions


def _log_split_metadata(fold_tag: str, prompt_type: str, train_meta, val_meta):
    meta_dir = Path("split_metadata") / prompt_type / "pooled"
    meta_dir.mkdir(parents=True, exist_ok=True)

    def summarise(entries):
        return [
            {"story_id": m["story_id"], "seed": m["seed"], "model_type": m["model_type"]}
            for m in entries
        ]

    payload = {
        "fold": fold_tag,
        "train_sequences": sorted(set(m["story_id"] for m in train_meta)),
        "val_sequences": sorted(set(m["story_id"] for m in val_meta)),
        "train_records": len(train_meta),
        "val_records": len(val_meta),
    }
    out = meta_dir / f"{fold_tag}_split.json"
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Logged split metadata to {out}")


def balanced_class_weights(labels) -> torch.Tensor:
    """Inverse-frequency weights: w_c = N / (n_classes * n_c)."""
    counts = Counter(labels)
    n = len(labels)
    n_classes = len(counts)
    weights = [n / (n_classes * counts[c]) for c in range(n_classes)]
    logger.info(f"Class weights: human(0)={weights[0]:.4f}, AI(1)={weights[1]:.4f}")
    return torch.tensor(weights, dtype=torch.float32)


def _build_training_args(output_dir: str, epochs: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_strategy="steps",
        report_to=None,
        seed=42,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )


def _select(values, indices):
    return [values[i] for i in indices]


def train_fold(
    texts, labels, meta, groups,
    train_idx, val_idx,
    output_dir, prompt_type, epochs,
    fold_tag="fold-1",
    fold_num=1,
    total_folds=5,
):
    fold_start = time.time()
    logger.info(f"\n--- FOLD {fold_num}/{total_folds} ({fold_tag}) ---")

    train_texts = _select(texts, train_idx)
    val_texts = _select(texts, val_idx)
    train_labels = _select(labels, train_idx)
    val_labels = _select(labels, val_idx)
    train_meta = _select(meta, train_idx)
    val_meta = _select(meta, val_idx)

    logger.info(
        f"{fold_tag} — Train: {len(train_texts)} "
        f"(human {sum(1 for l in train_labels if l==0)}, "
        f"AI {sum(1 for l in train_labels if l==1)}), "
        f"Val: {len(val_texts)} "
        f"(human {sum(1 for l in val_labels if l==0)}, "
        f"AI {sum(1 for l in val_labels if l==1)})"
    )
    _log_split_metadata(fold_tag, prompt_type, train_meta, val_meta)

    # Tokenizer & model (fresh each fold)
    logger.info(f"[{fold_tag}] Loading tokenizer & model from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{fold_tag}] Device: {device}")
    model = model.to(device)

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)

    # Class weights from training split
    class_weights = balanced_class_weights(train_labels)

    training_args = _build_training_args(output_dir, epochs)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info(f"[{fold_tag}] Starting training ({epochs} epochs)...")
    train_start = time.time()
    trainer.train()
    train_elapsed = time.time() - train_start
    logger.info(f"[{fold_tag}] Training completed in {train_elapsed/60:.1f} min")

    # Overall eval
    logger.info(f"[{fold_tag}] Running overall evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"[{fold_tag}] Overall eval: acc={eval_results.get('eval_accuracy', 0):.4f}, "
                f"f1={eval_results.get('eval_f1', 0):.4f}, "
                f"precision={eval_results.get('eval_precision', 0):.4f}, "
                f"recall={eval_results.get('eval_recall', 0):.4f}")

    # Per-model breakdown + out-of-fold predictions
    logger.info(f"[{fold_tag}] Running per-model evaluation...")
    model_results, oof_predictions = per_model_eval(
        trainer, val_texts, val_labels, val_meta, tokenizer, fold_tag
    )

    # Save best model
    logger.info(f"[{fold_tag}] Saving best model...")
    model_dir = Path("roberta-finetuned-pooled") / "cv" / fold_tag
    best_dir = model_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info(f"[{fold_tag}] Best model saved to {best_dir}")

    fold_elapsed = time.time() - fold_start
    logger.info(f"[{fold_tag}] Fold completed in {fold_elapsed/60:.1f} min")

    return {**eval_results, "per_model": model_results, "oof_predictions": oof_predictions}


def _aggregate(fold_results):
    metric_keys = ["eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]
    agg = {"num_folds": len(fold_results), "fold_results": fold_results}

    for key in metric_keys:
        vals = [float(r[key]) for r in fold_results if key in r]
        agg[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{key}_std"] = float(np.std(vals)) if vals else float("nan")

    # Per-model aggregate
    all_models = set()
    for r in fold_results:
        all_models.update(r.get("per_model", {}).keys())

    per_model_agg = {}
    for mtype in sorted(all_models):
        bucket = {}
        for metric in ["accuracy", "f1", "detection_rate", "false_positive_rate"]:
            vals = [
                r["per_model"][mtype][metric]
                for r in fold_results
                if mtype in r.get("per_model", {}) and metric in r["per_model"][mtype]
            ]
            if vals:
                bucket[f"{metric}_mean"] = float(np.mean(vals))
                bucket[f"{metric}_std"] = float(np.std(vals))
        per_model_agg[mtype] = bucket

    agg["per_model_aggregate"] = per_model_agg
    return agg


def main():
    args = parse_args()
    if args.cv_folds < 2:
        logger.error("--cv-folds must be >= 2 for GroupKFold")
        sys.exit(1)

    logger.info("Pooled Human-vs-AI RoBERTa fine-tuning with GroupKFold")
    logger.info(f"prompt_type={args.prompt_type}, cv_folds={args.cv_folds}, "
                f"epochs={args.epochs}, ai_seed={args.ai_seed}")

    texts, labels, groups, meta = load_pooled_data(args.prompt_type, args.ai_seed)

    groups_arr = np.array(groups)
    gkf = GroupKFold(n_splits=args.cv_folds)

    fold_results = []
    temp_base = "./temp-training-pooled"

    total_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(np.zeros(len(labels)), labels, groups=groups_arr), start=1
    ):
        fold_tag = f"fold-{fold_idx}"
        fold_output_dir = os.path.join(temp_base, fold_tag)

        try:
            result = train_fold(
                texts, labels, meta, groups,
                train_idx.tolist(), val_idx.tolist(),
                fold_output_dir, args.prompt_type, args.epochs,
                fold_tag=fold_tag,
                fold_num=fold_idx,
                total_folds=args.cv_folds,
            )
            fold_results.append({"fold": fold_idx, **result})
        except Exception as e:
            logger.error(f"Error in {fold_tag}: {e}", exc_info=True)
            continue
        finally:
            if os.path.exists(fold_output_dir):
                shutil.rmtree(fold_output_dir)
                logger.info(f"Cleaned up {fold_output_dir}")

    # ---- Collect out-of-fold predictions across all folds ----
    all_oof = []
    for r in fold_results:
        all_oof.extend(r.pop("oof_predictions", []))

    # Aggregate
    aggregated = _aggregate(fold_results)

    out_dir = Path("roberta-finetuned-pooled") / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "cv_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"Saved CV metrics to {metrics_path}")

    # Save all out-of-fold predictions (JSON)
    oof_json_path = out_dir / "oof_predictions.json"
    with oof_json_path.open("w") as f:
        json.dump(all_oof, f, indent=2)
    logger.info(f"Saved {len(all_oof)} out-of-fold predictions to {oof_json_path}")

    # Save as CSV for quick inspection
    oof_csv_path = out_dir / "oof_predictions.csv"
    if all_oof:
        import csv
        fieldnames = list(all_oof[0].keys())
        with oof_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_oof)
        logger.info(f"Saved CSV to {oof_csv_path}")

    # Human-text summary
    human_oof = [p for p in all_oof if p["model_type"] == "human"]
    if human_oof:
        n_human = len(human_oof)
        n_flagged = sum(1 for p in human_oof if p["predicted_label"] == 1)
        logger.info(f"")
        logger.info(f"HUMAN TEXT PREDICTIONS (out-of-fold, no leakage):")
        logger.info(f"  Total human texts: {n_human}")
        logger.info(f"  Flagged as AI:     {n_flagged} ({100*n_flagged/n_human:.1f}%)")
        logger.info(f"  Classified human:  {n_human - n_flagged} ({100*(n_human-n_flagged)/n_human:.1f}%)")

    # Summary
    logger.info("\n--- TRAINING SUMMARY (pooled classifier) ---")
    logger.info(
        f"  Accuracy: {aggregated['eval_accuracy_mean']:.4f} "
        f"± {aggregated['eval_accuracy_std']:.4f}"
    )
    logger.info(
        f"  F1:       {aggregated['eval_f1_mean']:.4f} "
        f"± {aggregated['eval_f1_std']:.4f}"
    )
    logger.info(
        f"  Precision:{aggregated['eval_precision_mean']:.4f} "
        f"± {aggregated['eval_precision_std']:.4f}"
    )
    logger.info(
        f"  Recall:   {aggregated['eval_recall_mean']:.4f} "
        f"± {aggregated['eval_recall_std']:.4f}"
    )

    logger.info("")
    logger.info("Per-model breakdown (mean ± std across folds):")
    for mtype, stats in aggregated.get("per_model_aggregate", {}).items():
        parts = [f"{k}={v:.4f}" for k, v in stats.items()]
        logger.info(f"  {mtype}: {', '.join(parts)}")

    total_elapsed = time.time() - total_start
    logger.info(f"Total wall time: {total_elapsed/60:.1f} min")
    logger.info("")
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
