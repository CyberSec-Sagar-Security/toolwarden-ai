"""Single-text classifier scoring for the live demo loop. Reuses the exact
Phase 4/8 model-loading and ensemble-fitting path (load_fitted_models) —
the demo scores content with the same trained models the degradation
curve was measured against, not a separate/simplified copy.
"""

from __future__ import annotations

from toolwarden.classifier.evaluate import _deberta_probs, _lightgbm_probs, load_fitted_models


class Classifier:
    def __init__(self) -> None:
        self.tokenizer, self.model, self.booster, self.stacker = load_fitted_models()

    def score(self, text: str) -> float:
        """Returns the ensemble's P(injection) for a single piece of text."""
        records = [{"text": text, "label": "benign"}]  # label unused for inference, required by records_to_xy's shared shape
        deberta_probs = _deberta_probs(records, self.model, self.tokenizer)
        lightgbm_probs = _lightgbm_probs(records, self.booster)
        ensemble_probs = self.stacker.predict_proba(deberta_probs, lightgbm_probs)
        return ensemble_probs[0]
