from toolwarden.classifier.ensemble import EnsembleStacker


def test_coefficients_exposes_fitted_weights_not_just_predictions():
    stacker = EnsembleStacker()
    # A tiny separable fit: label tracks deberta_prob, lightgbm_prob is pure noise.
    deberta = [0.1, 0.2, 0.8, 0.9, 0.15, 0.85]
    lightgbm = [0.9, 0.1, 0.9, 0.1, 0.5, 0.5]
    labels = [0, 0, 1, 1, 0, 1]

    stacker.fit(deberta, lightgbm, labels)
    coef = stacker.coefficients()

    assert set(coef) == {"deberta_weight", "lightgbm_weight", "intercept"}
    assert all(isinstance(v, float) for v in coef.values())
    # deberta_prob is the actual signal here, lightgbm_prob is noise — the
    # fitted weight should reflect that asymmetry, not treat both equally.
    assert coef["deberta_weight"] > coef["lightgbm_weight"]
