"""Combines DeBERTa and LightGBM injection-probability scores into one
ensemble score via a small logistic-regression stacker (2 inputs: each base
model's P(injection)). The stacker is fit on the val slice only (carved
from train, see data.py) — never on test or held_out_novel, so the
ensemble's reported numbers on those splits are as honest as the base
models' own.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class EnsembleStacker:
    def __init__(self) -> None:
        self._model = LogisticRegression()

    def fit(self, deberta_probs: list[float], lightgbm_probs: list[float], labels: list[int]) -> None:
        x = np.column_stack([deberta_probs, lightgbm_probs])
        self._model.fit(x, labels)

    def predict_proba(self, deberta_probs: list[float], lightgbm_probs: list[float]) -> list[float]:
        x = np.column_stack([deberta_probs, lightgbm_probs])
        return self._model.predict_proba(x)[:, 1].tolist()

    def coefficients(self) -> dict[str, float]:
        """Fitted weights: how much the stacker actually trusts each base
        model's score relative to the other, plus the intercept. Exposed so
        downstream reporting can check whether the combination is naively
        equal-weighting two models of very different out-of-distribution
        reliability, rather than assuming it from black-box predictions
        alone (see docs/degradation_curve_report.md's ensemble-combination
        ablation section).
        """
        coef = self._model.coef_[0]
        return {"deberta_weight": float(coef[0]), "lightgbm_weight": float(coef[1]), "intercept": float(self._model.intercept_[0])}
