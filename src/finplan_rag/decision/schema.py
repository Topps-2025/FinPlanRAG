"""Decision-level contracts for the FinPlan-RAG v2 pilot.

The pilot reuses real financial records but abstracts them into a decision
problem: a hypothesis space, typed evidence with availability times and costs,
a cutoff or an endogenous decision time, and a gold outcome.  The contracts are
deliberately small so that the pilot can be audited and re-used by the full
experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class EvidenceItem:
    """One piece of evidence with an explicit availability time and price."""

    evidence_id: str
    available_at: datetime
    cost: float
    kind: str  # soft | hard | vacuous | adversarial | stale
    origin: str
    source_type: str
    reliability: float
    payload: Mapping[str, Any] = field(default_factory=dict)
    text: str = ""


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    label: str


@dataclass(frozen=True)
class DecisionCase:
    """A decision-level case derived from a financial claim."""

    case_id: str
    task_type: str  # determinate | vague | mixed | mode_ambiguous
    query: str
    hypotheses: tuple[Hypothesis, ...]
    evidence: tuple[EvidenceItem, ...]
    cutoff: datetime
    gold_hypothesis: str
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionOutput:
    """The auditable result of one policy on one case."""

    case_id: str
    policy: str
    action: str  # answer | adjudicate | abstain
    prediction: str | None
    credible_set: tuple[str, ...]
    belief: Mapping[str, float]
    decision_loss: float
    evidence_cost: float
    delay_cost: float
    total_cost: float
    decision_days: float
    used_evidence: tuple[str, ...]
    leaked: bool
    note: str = ""
