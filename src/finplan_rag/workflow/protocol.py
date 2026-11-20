"""Single source of truth for the paper protocol constants.

The legacy runner scripts import these instead of re-declaring the same
constants. The workflow config in ``configs/paper.json`` documents the same
values; ``scripts/workflow.py`` can compare them if a non-default protocol is
needed.
"""

from __future__ import annotations

from dataclasses import replace

from finplan_rag.decision.loss import LossConfig

TEST_YEARS = (2022, 2023, 2024, 2025)
FOLDS = 5
FOLD_SEEDS = (1, 2, 3, 4, 5)
DECISION_SEEDS = tuple(range(1, 11))
VERIFIED_LABELS = frozenset({"xbrl_search", "xbrl_verified", "xbrl_override"})
ALPHA_GRID = (0.05, 0.10, 0.20)
RF_TREES = 200

BASE = LossConfig()
CONFIGS = [
    ("base", BASE),
    ("symmetric", replace(BASE, over=10.0, under=10.0, wrong=10.0)),
    ("over_averse", replace(BASE, over=20.0, under=5.0)),
    ("under_averse", replace(BASE, over=5.0, under=20.0)),
    ("cheap_abstain", replace(BASE, abstain=2.0)),
    ("expensive_abstain", replace(BASE, abstain=10.0)),
    ("narrow_penalty", replace(BASE, width=0.5)),
    ("wide_penalty", replace(BASE, width=4.0)),
    ("alpha_05", replace(BASE, alpha=0.05)),
    ("alpha_20", replace(BASE, alpha=0.2)),
]
CONFIG_NAMES = tuple(name for name, _ in CONFIGS)
