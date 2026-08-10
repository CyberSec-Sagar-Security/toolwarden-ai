from toolwarden.datasets.injecagent import _clean_template, _group_split
from toolwarden.datasets.schema import Split


def test_group_split_is_deterministic():
    key = "direct_harm:AugustSmartLockGrantGuestAccess"

    assert _group_split(key) == _group_split(key)


def test_group_split_returns_valid_split_values():
    for i in range(50):
        result = _group_split(f"group-{i}")
        assert result in (Split.TRAIN, Split.TEST)


def test_group_split_roughly_80_20_over_many_groups():
    splits = [_group_split(f"group-{i}") for i in range(500)]
    test_fraction = splits.count(Split.TEST) / len(splits)

    assert 0.10 < test_fraction < 0.30


def test_clean_template_removes_placeholder():
    template = "{'note': {'snippet': '<Attacker Instruction>'}}"

    cleaned = _clean_template(template)

    assert "<Attacker Instruction>" not in cleaned
