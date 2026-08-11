"""Combines DeBERTa and LightGBM injection-probability scores into one
ensemble score via a small logistic-regression stacker (2 inputs: each base
model's P(injection)). The stacker is fit on the val slice only (carved
from train, see data.py) — never on test or held_out_novel, so the
ensemble's reported numbers on those splits are as honest as the base
models' own.
"""

from __future__ import annotations

import json
from pathlib import Path

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

    def save(self, path: str | Path) -> None:
        """Persists the fitted weights only (3 floats + classes_), not a
        pickled sklearn object — avoids coupling a loaded classifier's
        correctness to matching sklearn versions across the machine that
        trained it and the machine that later just runs inference. See
        load_fitted_models() in evaluate.py for why this matters: fitting
        requires the processed training dataset, which is dev-only and not
        distributed with the pip package, but loading a fitted classifier
        for inference should not.
        """
        coef = self._model.coef_[0]
        data = {
            "deberta_weight": float(coef[0]),
            "lightgbm_weight": float(coef[1]),
            "intercept": float(self._model.intercept_[0]),
            "classes": self._model.classes_.tolist(),
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleStacker":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        stacker = cls()
        stacker._model.coef_ = np.array([[data["deberta_weight"], data["lightgbm_weight"]]])
        stacker._model.intercept_ = np.array([data["intercept"]])
        stacker._model.classes_ = np.array(data["classes"])
        stacker._model.n_features_in_ = 2
        return stacker
