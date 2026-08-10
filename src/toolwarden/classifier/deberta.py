"""Fine-tunes microsoft/deberta-v3-base for injection/benign classification.

Exact base model per the build spec's locked decisions — the benchmark
comparison against ProtectAI's DeBERTa-v3-base baseline only holds if this
exact base model is used, not a different DeBERTa variant.

Checkpoint goes to config.DEBERTA_CHECKPOINT_DIR (Apps and Models, never
this repo). Model selection uses a val slice carved out of train only
(data.train_fit_val_split) — test and held_out_novel are never touched
during training, only in evaluate.py.

Run with: python -m toolwarden.classifier.deberta
"""

from __future__ import annotations

from pathlib import Path

from toolwarden import config

config.configure_hf_cache_env()  # must run before importing transformers

BASE_MODEL = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_PATH = REPO_ROOT / "datasets" / "processed" / "records.jsonl"


def _build_dataset(records, tokenizer):
    import torch

    class _TorchDataset(torch.utils.data.Dataset):
        def __init__(self, records, tokenizer):
            self.encodings = tokenizer(
                [r["text"] for r in records],
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
            )
            from toolwarden.classifier.data import LABEL2ID

            self.labels = [LABEL2ID[r["label"]] for r in records]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    return _TorchDataset(records, tokenizer)


def _compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def train() -> None:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from toolwarden.classifier.data import ID2LABEL, LABEL2ID, load_records, train_fit_val_split

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    records = load_records(PROCESSED_PATH)
    fit_records, val_records = train_fit_val_split(records)
    print(f"Fine-tuning on {len(fit_records)} records, validating on {len(val_records)} (carved from train only)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="single_label_classification",
    )

    fit_dataset = _build_dataset(fit_records, tokenizer)
    val_dataset = _build_dataset(val_records, tokenizer)

    config.DEBERTA_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = -(-len(fit_dataset) // BATCH_SIZE)  # ceil div
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)

    args = TrainingArguments(
        output_dir=str(config.DEBERTA_CHECKPOINT_DIR / "training_runs"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=torch.cuda.is_available(),
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=fit_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    final_dir = config.DEBERTA_CHECKPOINT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    train()
