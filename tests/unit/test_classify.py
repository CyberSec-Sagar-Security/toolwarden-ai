"""Classifier's detector_mode branching, tested without loading real model
weights: load_fitted_models/_deberta_probs/_lightgbm_probs are monkeypatched
at the module level Classifier actually calls them through, same pattern
every other unit test in this project uses to avoid a real DeBERTa/LightGBM
load in a fast unit test.
"""

from __future__ import annotations

import pytest

from toolwarden.demo import classify as classify_module
from toolwarden.demo.classify import Classifier


class FakeStacker:
    def __init__(self, result: float):
        self._result = result
        self.calls: list[tuple[list[float], list[float]]] = []

    def predict_proba(self, deberta_probs, lightgbm_probs):
        self.calls.append((deberta_probs, lightgbm_probs))
        return [self._result]


def test_ensemble_mode_combines_both_models(monkeypatch):
    stacker = FakeStacker(result=0.5)
    monkeypatch.setattr(classify_module, "load_fitted_models", lambda: (None, None, None, stacker))
    monkeypatch.setattr(classify_module, "_deberta_probs", lambda records, model, tokenizer: [0.9])
    monkeypatch.setattr(classify_module, "_lightgbm_probs", lambda records, booster: [0.1])

    clf = Classifier(detector_mode="ensemble")

    assert clf.score("some text") == 0.5
    assert stacker.calls == [([0.9], [0.1])]


def test_deberta_only_mode_skips_lightgbm_and_stacker_entirely(monkeypatch):
    monkeypatch.setattr(classify_module, "load_fitted_models", lambda: (None, None, None, None))
    monkeypatch.setattr(classify_module, "_deberta_probs", lambda records, model, tokenizer: [0.77])

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("deberta_only mode must not call LightGBM")

    monkeypatch.setattr(classify_module, "_lightgbm_probs", _fail_if_called)

    clf = Classifier(detector_mode="deberta_only")

    assert clf.score("some text") == 0.77


def test_default_mode_is_ensemble(monkeypatch):
    monkeypatch.setattr(classify_module, "load_fitted_models", lambda: (None, None, None, FakeStacker(result=0.3)))
    monkeypatch.setattr(classify_module, "_deberta_probs", lambda records, model, tokenizer: [0.4])
    monkeypatch.setattr(classify_module, "_lightgbm_probs", lambda records, booster: [0.6])

    clf = Classifier()

    assert clf.detector_mode == "ensemble"
    assert clf.score("some text") == 0.3


def test_invalid_detector_mode_raises_before_loading_models(monkeypatch):
    def _fail_if_called():
        raise AssertionError("should validate detector_mode before loading any model weights")

    monkeypatch.setattr(classify_module, "load_fitted_models", _fail_if_called)

    with pytest.raises(ValueError, match="detector_mode"):
        Classifier(detector_mode="bogus")
