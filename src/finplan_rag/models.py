"""Small, serializable contracts shared by the FinPlan-RAG components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Tuple


@dataclass(frozen=True)
class DocumentObligation:
    """One filing that must be covered by a point-in-time question."""

    entity: str
    fiscal_year: int
    filing_type: str
    fiscal_period: str
    document_id: str = ""

    @property
    def key(self) -> Tuple[str, int, str, str]:
        return (self.entity, self.fiscal_year, self.filing_type, self.fiscal_period)


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage with an explicit availability timestamp."""

    chunk_id: str
    document_id: str
    entity: str
    available_at: datetime
    text: str
    terms: Tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class EvidenceState:
    """Auditable planner state after a retrieval step."""

    obligations: Tuple[DocumentObligation, ...]
    covered_document_ids: Tuple[str, ...]
    queries: int
    closed: bool
    future_document_ids: Tuple[str, ...] = ()
    retrieved_chunks: Tuple[Chunk, ...] = ()
