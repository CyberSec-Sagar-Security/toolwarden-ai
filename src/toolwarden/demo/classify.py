"""Single-text classifier scoring, used by every guarded adapter (Phase 9's
guarded_loop, Phase 10's mcp_proxy). Reuses the exact Phase 4/8
model-loading and ensemble-fitting path (load_fitted_models) — scoring
happens against the same trained models the degradation curve was measured
against, not a separate/simplified copy.
"""

from __future__ import annotations

from toolwarden.classifier.evaluate import _deberta_probs, _lightgbm_probs, load_fitted_models

DETECTOR_MODES = ("ensemble", "deberta_only")


class Classifier:
    def __init__(self, detector_mode: str = "ensemble") -> None:
        """detector_mode="deberta_only" is a direct, documented consequence
        of the ensemble-combination ablation (docs/degradation_curve_report.md):
        the fitted stacker weighs LightGBM almost as heavily as DeBERTa despite
        LightGBM being measurably less reliable on held-out data, which
        measurably suppresses real attacks DeBERTa alone would have caught
        (3/35 on AgentDojo, 6/300 on the synthetic set). Fixing the stacker
        itself needs a third calibration split never touched by either
        benchmark (not yet built — see known_limitations.md); until then,
        this gives a caller a documented way to opt out of that specific
        failure mode today, at the cost of losing LightGBM's contribution
        entirely rather than a partial reweighting.
        """
        if detector_mode not in DETECTOR_MODES:
            raise ValueError(f"detector_mode must be one of {DETECTOR_MODES}, got {detector_mode!r}")
        self.detector_mode = detector_mode
        self.tokenizer, self.model, self.booster, self.stacker = load_fitted_models()

    def score(self, text: str) -> float:
        """Returns P(injection) for a single piece of text, per detector_mode."""
        records = [{"text": text, "label": "benign"}]  # label unused for inference, required by records_to_xy's shared shape
        deberta_probs = _deberta_probs(records, self.model, self.tokenizer)

        if self.detector_mode == "deberta_only":
            return deberta_probs[0]

        lightgbm_probs = _lightgbm_probs(records, self.booster)
        ensemble_probs = self.stacker.predict_proba(deberta_probs, lightgbm_probs)
        return ensemble_probs[0]
