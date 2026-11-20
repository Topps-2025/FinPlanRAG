"""Loss-aware portfolio over fixed, abstain, and planner policies.

The portfolio does not choose a policy from test outcomes.  It compares the
validation loss of a conservative default (fixed adjudication) with abstention
and the belief-driven planner, and switches only when a candidate improves
validation loss by at least ``switching_margin``.  The margin is a
conservatism/simplicity choice, not a learned parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

POLICIES = ("fixed", "abstain", "planner")


@dataclass(frozen=True)
class PortfolioConfig:
    switching_margin: float = 0.15
    default: str = "fixed"

    def __post_init__(self) -> None:
        if self.default not in POLICIES:
            raise ValueError(f"unknown default policy: {self.default}")
        if self.switching_margin < 0:
            raise ValueError("switching_margin must be non-negative")


def select_policy(
    validation_losses: Mapping[str, float],
    config: PortfolioConfig | None = None,
) -> str:
    """Select a policy by validation loss with a conservative margin."""
    config = config or PortfolioConfig()
    if config.default not in validation_losses:
        raise ValueError(f"missing default policy loss: {config.default}")
    selected = config.default
    for candidate in POLICIES:
        if candidate == selected:
            continue
        if candidate not in validation_losses:
            continue
        if (
            validation_losses[candidate] + config.switching_margin
            < validation_losses[selected]
        ):
            selected = candidate
    return selected
