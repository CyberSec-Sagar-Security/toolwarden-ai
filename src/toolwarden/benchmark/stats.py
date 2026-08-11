"""Uncertainty estimates for the Phase 8 degradation curve. Every point
estimate reported alongside a confidence interval, not bare — per Sagar's
explicit instruction (2026-08-11).
"""

from __future__ import annotations

import math
import random


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n. Preferred
    over the normal approximation for small n or proportions near 0/1 —
    both occur here (some buckets are under 60 examples, some near-perfect).
    """
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def bootstrap_f1_ci(
    labels: list[int], preds: list[int], n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """Returns (point_estimate, lo, hi) for F1 via percentile bootstrap.
    F1 combines precision and recall nonlinearly, so there's no closed-form
    interval the way there is for a single proportion — resampling is the
    standard honest approach.
    """
    from sklearn.metrics import f1_score

    if len(labels) == 0:
        return (0.0, 0.0, 0.0)

    point = f1_score(labels, preds, zero_division=0)

    rng = random.Random(seed)
    n = len(labels)
    boot_scores = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boot_labels = [labels[i] for i in idx]
        boot_preds = [preds[i] for i in idx]
        boot_scores.append(f1_score(boot_labels, boot_preds, zero_division=0))

    boot_scores.sort()
    lo = boot_scores[int(0.025 * n_boot)]
    hi = boot_scores[min(int(0.975 * n_boot), n_boot - 1)]
    return (point, lo, hi)
