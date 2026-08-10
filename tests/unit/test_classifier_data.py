from toolwarden.classifier.data import by_split, train_fit_val_split

RECORDS = [
    {"id": f"train-{i}", "text": "x", "label": "benign", "source": "s", "attack_type": None, "tool_context": None, "split": "train", "metadata": {}}
    for i in range(200)
] + [
    {"id": "test-1", "text": "x", "label": "benign", "source": "s", "attack_type": None, "tool_context": None, "split": "test", "metadata": {}},
    {"id": "held-1", "text": "x", "label": "injection", "source": "s", "attack_type": None, "tool_context": None, "split": "held_out_novel", "metadata": {}},
]


def test_by_split_filters_correctly():
    assert len(by_split(RECORDS, "train")) == 200
    assert len(by_split(RECORDS, "test")) == 1
    assert len(by_split(RECORDS, "held_out_novel")) == 1


def test_train_fit_val_split_only_touches_train_and_is_disjoint():
    fit, val = train_fit_val_split(RECORDS)

    fit_ids = {r["id"] for r in fit}
    val_ids = {r["id"] for r in val}

    assert fit_ids.isdisjoint(val_ids)
    assert len(fit) + len(val) == 200
    assert all(r["id"].startswith("train-") for r in fit + val)


def test_train_fit_val_split_is_deterministic():
    fit1, val1 = train_fit_val_split(RECORDS)
    fit2, val2 = train_fit_val_split(RECORDS)

    assert [r["id"] for r in fit1] == [r["id"] for r in fit2]
    assert [r["id"] for r in val1] == [r["id"] for r in val2]


def test_train_fit_val_split_roughly_90_10():
    fit, val = train_fit_val_split(RECORDS)

    val_fraction = len(val) / (len(fit) + len(val))
    assert 0.03 < val_fraction < 0.20
