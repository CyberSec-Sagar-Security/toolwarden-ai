"""Explainability layer (Phase 5): for any flagged detection, show which
features/tokens drove it — not just a bare score. Two independent views:

- SHAP over the LightGBM engineered features (which signal — imperative
  phrasing, base64 blobs, etc. — pushed the score toward injection).
- Attention over the DeBERTa tokens (which tokens the classifier's pooled
  representation attended to most).

Neither explainer changes the classifier's decision — see docs/architecture.md's
component table: explainability attaches evidence, it doesn't own the verdict.
"""

from __future__ import annotations

import numpy as np

from toolwarden.features.extractors import extract_all


def explain_lightgbm(text: str, booster, feature_names: list[str], top_k: int = 10) -> list[tuple[str, float]]:
    """Returns (feature_name, shap_value) pairs, sorted by |shap_value| descending.
    Positive values push the score toward 'injection'.
    """
    import shap

    features = extract_all(text)
    x = np.array([[features[name] for name in feature_names]])

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(x)
    values = np.asarray(shap_values).reshape(-1)

    pairs = list(zip(feature_names, values.tolist()))
    pairs.sort(key=lambda pair: -abs(pair[1]))
    return pairs[:top_k]


def explain_deberta(text: str, model, tokenizer, top_k: int = 10) -> list[tuple[str, float]]:
    """Returns (token, attention_weight) pairs: how much the pooled [first-
    token] representation attends to each other token in the last layer,
    averaged over heads. Special tokens are excluded from the ranking.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    encoding = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**encoding, output_attentions=True)

    last_layer_attn = outputs.attentions[-1][0]  # [heads, seq, seq]
    pooled_token_attn = last_layer_attn[:, 0, :].mean(dim=0)  # avg over heads, attn FROM first token

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    special = set(tokenizer.all_special_tokens)
    pairs = [
        (token, weight)
        for token, weight in zip(tokens, pooled_token_attn.cpu().tolist())
        if token not in special
    ]
    pairs.sort(key=lambda pair: -pair[1])
    return pairs[:top_k]


def explain(text: str, deberta_model, tokenizer, booster, feature_names: list[str], top_k: int = 10) -> dict:
    return {
        "text": text,
        "lightgbm_top_features": explain_lightgbm(text, booster, feature_names, top_k),
        "deberta_top_tokens": explain_deberta(text, deberta_model, tokenizer, top_k),
    }
