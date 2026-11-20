"""Tests for the extended metric suite."""

from __future__ import annotations

import numpy as np

from compute_metrics_suite import compute


def test_perfect_predictions_have_max_metrics() -> None:
    probabilities = np.eye(3)
    truths = np.asarray([0, 1, 2])
    metrics = compute(probabilities, truths, bootstrap=False)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["auroc_macro"] == 1.0
    assert metrics["ece"] == 0.0


def test_metric_ranges() -> None:
    rng = np.random.default_rng(7)
    probabilities = rng.dirichlet([1.0, 1.0, 1.0], size=60)
    truths = rng.integers(0, 3, size=60)
    metrics = compute(probabilities, truths, bootstrap=False)
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "auroc_macro", "auprc_macro"):
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["n"] == 60
    assert 0.0 <= metrics["ece"] <= 1.0
    assert metrics["brier"] >= 0.0
    assert metrics["nll"] >= 0.0
