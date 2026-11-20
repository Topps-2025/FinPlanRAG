"""Regression tests for the real-evidence benchmark pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_real_benchmark import make_pipeline  # noqa: E402


def test_categorical_encoder_targets_the_categorical_column() -> None:
    numeric_count = 11
    pipeline = make_pipeline(numeric_count, c_value=1.0, class_weight=None)
    preprocess = pipeline.named_steps["preprocess"]
    numeric_columns = preprocess.transformers[0][2]
    categorical_columns = preprocess.transformers[1][2]
    assert numeric_columns == list(range(numeric_count))
    assert categorical_columns == [numeric_count]
