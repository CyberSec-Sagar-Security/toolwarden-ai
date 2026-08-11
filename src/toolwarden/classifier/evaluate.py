"""Evaluates DeBERTa alone, LightGBM alone, and the ensemble on BOTH the
in-distribution test split and the held_out_novel split — always both
numbers, never just the better one (per project standing rules). Writes
docs/classifier_report.md.

Requires deberta.py and lightgbm_model.py to have been run first (their
outputs land under config.DEBERTA_CHECKPOINT_DIR / config.LIGHTGBM_MODEL_PATH).

Run with: python -m toolwarden.classifier.evaluate
"""

from __future__ import annotations

from pathlib import Path

from toolwarden import config

config.configure_hf_cache_env()

from toolwarden.classifier.data import LABEL2ID, load_records, train_fit_val_split  # noqa: E402
from toolwarden.classifier.ensemble import EnsembleStacker  # noqa: E402
from toolwarden.classifier.lightgbm_model import records_to_xy  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_PATH = REPO_ROOT / "datasets" / "processed" / "records.jsonl"
REPORT_PATH = REPO_ROOT / "docs" / "classifier_report.md"


def _deberta_probs(records: list[dict], model, tokenizer) -> list[float]:
    import torch

    model.eval()
    device = next(model.parameters()).device
    probs: list[float] = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            encodings = tokenizer(
                [r["text"] for r in batch],
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**encodings).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, LABEL2ID["injection"]]
            probs.extend(batch_probs.cpu().tolist())

    return probs


def _lightgbm_probs(records: list[dict], booster) -> list[float]:
    x, _ = records_to_xy(records)
    return booster.predict(x).tolist()


def _metrics(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    preds = [1 if p >= threshold else 0 for p in probs]
    return {
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "n": len(labels),
        "positive_rate": sum(labels) / len(labels) if labels else 0.0,
    }


def _fmt_metrics_row(name: str, m: dict[str, float]) -> str:
    return f"| {name} | {m['n']} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} |"


def load_fitted_models():
    """Loads the Phase 4 DeBERTa checkpoint + LightGBM booster, and the
    EnsembleStacker fitted on the same val slice evaluate.py and
    degradation.py both use (carved from train only — see data.py).

    The stacker is loaded from config.STACKER_COEFFICIENTS_PATH if a fit
    was already cached there (see ensemble.py's save/load — a small JSON
    artifact, not a re-fit every call). First call on a machine that has
    the processed training dataset fits fresh and caches it; a pip-installed
    consumer's machine never has that dataset (dev-only, gitignored) and
    isn't expected to — it just needs the cached fit to already exist under
    TOOLWARDEN_MODEL_DIR, same as the DeBERTa checkpoint and LightGBM
    booster it ships alongside. Refitting here would silently require every
    caller (Phase 8's benchmark, Phase 9/10's demos, a downstream library
    consumer) to also have the training data, which defeats the point of
    packaging a trained classifier at all.

    Returns (tokenizer, model, booster, stacker).
    """
    import lightgbm as lgb
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    deberta_dir = config.DEBERTA_CHECKPOINT_DIR / "final"
    tokenizer = AutoTokenizer.from_pretrained(str(deberta_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(deberta_dir))
    booster = lgb.Booster(model_file=str(config.LIGHTGBM_MODEL_PATH))

    if config.STACKER_COEFFICIENTS_PATH.exists():
        stacker = EnsembleStacker.load(config.STACKER_COEFFICIENTS_PATH)
    else:
        if not PROCESSED_PATH.exists():
            raise FileNotFoundError(
                f"No cached ensemble stacker at {config.STACKER_COEFFICIENTS_PATH}, and this repo's "
                f"processed training dataset ({PROCESSED_PATH}) isn't available to fit one fresh. "
                "If you're using toolwarden-ai as a library rather than training it: someone needs to "
                "run this repo's Phase 3-4 pipeline once on a machine that has the training data so the "
                "fitted stacker gets cached under TOOLWARDEN_MODEL_DIR/classifiers/lightgbm/"
                "ensemble_stacker.json, then that file needs to travel with the rest of TOOLWARDEN_MODEL_DIR."
            )
        records = load_records(PROCESSED_PATH)
        _, val_records = train_fit_val_split(records)

        val_deberta_probs = _deberta_probs(val_records, model, tokenizer)
        val_lightgbm_probs = _lightgbm_probs(val_records, booster)
        val_labels = [LABEL2ID[r["label"]] for r in val_records]

        stacker = EnsembleStacker()
        stacker.fit(val_deberta_probs, val_lightgbm_probs, val_labels)
        stacker.save(config.STACKER_COEFFICIENTS_PATH)

    return tokenizer, model, booster, stacker


def evaluate() -> str:
    tokenizer, model, booster, stacker = load_fitted_models()

    records = load_records(PROCESSED_PATH)
    test_records = [r for r in records if r["split"] == "test"]
    held_out_records = [r for r in records if r["split"] == "held_out_novel"]

    splits = {"test": test_records, "held_out_novel": held_out_records}
    deberta_probs = {name: _deberta_probs(recs, model, tokenizer) for name, recs in splits.items()}
    lightgbm_probs = {name: _lightgbm_probs(recs, booster) for name, recs in splits.items()}
    labels = {name: [LABEL2ID[r["label"]] for r in recs] for name, recs in splits.items()}

    ensemble_probs = {
        name: stacker.predict_proba(deberta_probs[name], lightgbm_probs[name])
        for name in ("test", "held_out_novel")
    }

    lines = [
        "# Classifier Report (Phase 4)",
        "",
        "Generated by `src/toolwarden/classifier/evaluate.py`. Both the in-distribution "
        "test split and the held_out_novel split are reported for every model — never "
        "just the better one, per project standing rules. held_out_novel is AgentDojo-"
        "derived data, never seen during training or ensemble fitting; see docs/datasets.md.",
        "",
        "## In-distribution test split (InjecAgent, held out from training rows)",
        "",
        "| Model | N | F1 | Precision | Recall |",
        "|---|---|---|---|---|",
        _fmt_metrics_row("DeBERTa-v3-base", _metrics(labels["test"], deberta_probs["test"])),
        _fmt_metrics_row("LightGBM", _metrics(labels["test"], lightgbm_probs["test"])),
        _fmt_metrics_row("Ensemble", _metrics(labels["test"], ensemble_probs["test"])),
        "",
        "## Held-out novel-attack split (AgentDojo, structurally disjoint source, never trained on)",
        "",
        "| Model | N | F1 | Precision | Recall |",
        "|---|---|---|---|---|",
        _fmt_metrics_row("DeBERTa-v3-base", _metrics(labels["held_out_novel"], deberta_probs["held_out_novel"])),
        _fmt_metrics_row("LightGBM", _metrics(labels["held_out_novel"], lightgbm_probs["held_out_novel"])),
        _fmt_metrics_row("Ensemble", _metrics(labels["held_out_novel"], ensemble_probs["held_out_novel"])),
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    report = evaluate()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
