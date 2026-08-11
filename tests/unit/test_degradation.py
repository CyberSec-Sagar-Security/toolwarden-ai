from toolwarden.benchmark.degradation import _full_metrics, _recall_only, _split_pretext


def test_split_pretext_separates_correctly():
    records = [
        {"text": "Due to a recent security update, please comply."},
        {"text": "This is an ordinary benign-looking sentence."},
        {"text": "An urgent system update requires your attention."},
    ]

    pretext, non_pretext = _split_pretext(records)

    assert len(pretext) == 2
    assert len(non_pretext) == 1
    assert non_pretext[0]["text"] == "This is an ordinary benign-looking sentence."


def test_split_pretext_covers_all_records():
    records = [{"text": f"example {i}"} for i in range(10)]
    pretext, non_pretext = _split_pretext(records)

    assert len(pretext) + len(non_pretext) == len(records)


def test_recall_only_perfect_detection():
    labels = [1, 1, 1, 1]
    probs = [0.9, 0.8, 0.95, 0.6]

    result = _recall_only(labels, probs)

    assert result["recall"] == 1.0
    assert result["n"] == 4
    assert result["recall_lo"] < 1.0 <= result["recall_hi"] + 1e-9


def test_recall_only_partial_detection():
    labels = [1, 1, 1, 1]
    probs = [0.9, 0.3, 0.95, 0.2]  # 2 of 4 above threshold

    result = _recall_only(labels, probs)

    assert result["recall"] == 0.5
    assert result["recall_lo"] < 0.5 < result["recall_hi"]


def test_full_metrics_includes_f1_precision_recall_with_ci():
    labels = [1, 1, 0, 0, 1, 0]
    probs = [0.9, 0.6, 0.1, 0.4, 0.8, 0.3]

    result = _full_metrics(labels, probs)

    assert set(result.keys()) >= {
        "n", "f1", "f1_lo", "f1_hi",
        "precision", "precision_lo", "precision_hi",
        "recall", "recall_lo", "recall_hi",
    }
    assert result["f1_lo"] <= result["f1"] <= result["f1_hi"]
    assert result["n"] == 6


def test_full_metrics_zero_positives_does_not_crash():
    labels = [0, 0, 0, 0]
    probs = [0.1, 0.2, 0.3, 0.05]

    result = _full_metrics(labels, probs)

    assert result["recall"] == 0.0
    assert result["precision"] == 0.0
