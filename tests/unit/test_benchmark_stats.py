from toolwarden.benchmark.stats import bootstrap_f1_ci, wilson_interval


def test_wilson_interval_perfect_score_bounds_below_one():
    lo, hi = wilson_interval(50, 50)

    assert 0 < lo < 1
    assert hi == 1.0


def test_wilson_interval_zero_score_bounds_above_zero():
    lo, hi = wilson_interval(0, 50)

    assert lo == 0.0
    assert 0 < hi < 1


def test_wilson_interval_narrows_with_more_data():
    lo_small, hi_small = wilson_interval(8, 10)
    lo_large, hi_large = wilson_interval(800, 1000)

    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_wilson_interval_empty_returns_zero_zero():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_interval(30, 100)
    assert lo <= 0.30 <= hi


def test_bootstrap_f1_ci_perfect_predictions_tight_interval():
    labels = [1, 1, 1, 0, 0, 0] * 10
    preds = labels[:]

    point, lo, hi = bootstrap_f1_ci(labels, preds, n_boot=500)

    assert point == 1.0
    assert lo > 0.9


def test_bootstrap_f1_ci_point_matches_sklearn():
    from sklearn.metrics import f1_score

    labels = [1, 0, 1, 1, 0, 0, 1, 0]
    preds = [1, 0, 0, 1, 0, 1, 1, 0]

    point, lo, hi = bootstrap_f1_ci(labels, preds, n_boot=500)

    assert point == f1_score(labels, preds)
    assert lo <= point <= hi


def test_bootstrap_f1_ci_empty_returns_zeros():
    assert bootstrap_f1_ci([], [], n_boot=100) == (0.0, 0.0, 0.0)


def test_bootstrap_f1_ci_is_reproducible_with_same_seed():
    labels = [1, 0, 1, 1, 0, 0, 1, 0] * 5
    preds = [1, 0, 0, 1, 0, 1, 1, 0] * 5

    result1 = bootstrap_f1_ci(labels, preds, n_boot=300, seed=1)
    result2 = bootstrap_f1_ci(labels, preds, n_boot=300, seed=1)

    assert result1 == result2
