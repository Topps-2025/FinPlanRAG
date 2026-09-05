"""Non-oracle obligation planning and auditable retrieval loops."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping, Sequence

from .models import Chunk, DocumentObligation, EvidenceState
from .retrieval import BM25Index


def plan_obligations(question: str, filings: Sequence[Mapping[str, object]], entities: Sequence[str], cutoff: object | None = None) -> tuple[DocumentObligation, ...]:
    """Create filing obligations from the question and visible metadata."""
    from .temporal import build_obligations

    return build_obligations(question, entities, filings, cutoff)


class ObligationPlanner:
    """Retrieve one targeted passage per unresolved filing obligation.

    The planner only sees question text, filing metadata, the index, and the
    cutoff.  It never receives gold document IDs or answer labels.
    """

    def __init__(self, index: BM25Index, budget: int = 4) -> None:
        if budget < 1:
            raise ValueError("budget must be positive")
        self.index, self.budget = index, budget

    def retrieve(self, question: str, obligations: Sequence[DocumentObligation], cutoff: datetime) -> EvidenceState:
        used_chunks: list[Chunk] = []
        used_ids: set[str] = set()
        future_ids: list[str] = []
        for obligation in obligations:
            if len(used_chunks) >= self.budget:
                break
            query = f"{question} {obligation.entity} {obligation.fiscal_year} {obligation.filing_type} {obligation.fiscal_period}"
            required = (obligation.entity.lower(), str(obligation.fiscal_year))
            required_metadata = {
                "entity": obligation.entity,
                "fiscal_year": obligation.fiscal_year,
                "filing_type": obligation.filing_type,
                "fiscal_period": obligation.fiscal_period,
            }
            if obligation.document_id:
                required_metadata["document_id"] = obligation.document_id
            hits = self.index.search(
                query,
                cutoff,
                limit=1,
                exclude=used_ids,
                required_terms=required,
                required_metadata=required_metadata,
            )
            if not hits:
                continue
            hit = hits[0].chunk
            used_chunks.append(hit)
            used_ids.add(hit.chunk_id)
            if hit.available_at > cutoff:
                future_ids.append(hit.document_id)
        covered = tuple(sorted({chunk.document_id for chunk in used_chunks}))
        required_ids = {item.document_id for item in obligations if item.document_id}
        return EvidenceState(
            tuple(obligations),
            covered,
            len(used_chunks),
            required_ids.issubset(covered) if required_ids else bool(covered),
            tuple(sorted(set(future_ids))),
            tuple(used_chunks),
        )

    def run(self, question: str, obligations: Sequence[DocumentObligation], cutoff: datetime) -> EvidenceState:
        return self.retrieve(question, obligations, cutoff)
