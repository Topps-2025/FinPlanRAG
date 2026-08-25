"""Dependency-free BM25 retrieval with point-in-time filtering."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable, Sequence

from .models import Chunk, RetrievalResult

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(text.lower()))


def parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def make_chunks(documents: Iterable[dict], window: int = 160, stride: int = 120) -> list[Chunk]:
    """Split document records into stable, overlapping retrieval passages."""

    chunks: list[Chunk] = []
    for document in documents:
        words = str(document.get("text", "")).split()
        for start in range(0, len(words), stride):
            text = " ".join(words[start : start + window])
            if not text:
                break
            chunks.append(Chunk(
                chunk_id=f"{document['document_id']}::c{start}",
                document_id=str(document["document_id"]),
                entity=str(document.get("entity", "")),
                available_at=parse_timestamp(document["available_at"]),
                text=text,
                terms=tokens(text),
                metadata={k: v for k, v in document.items() if k not in {"text", "available_at"}},
            ))
            if start + window >= len(words):
                break
    return chunks


class BM25Index:
    """Small in-memory BM25 index suitable for deterministic experiments."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.term_frequencies = [Counter(c.terms) for c in self.chunks]
        self.document_frequency = Counter()
        for tf in self.term_frequencies:
            self.document_frequency.update(tf.keys())
        self.average_length = sum(len(c.terms) for c in self.chunks) / max(1, len(self.chunks))

    def search(self, query: str, cutoff: datetime | None = None, *, limit: int = 5, exclude: Iterable[str] = (), required_terms: Iterable[str] = ()) -> list[RetrievalResult]:
        excluded = set(exclude)
        required = set(required_terms)
        qterms = tokens(query)
        n = len(self.chunks)
        scored: list[RetrievalResult] = []
        for chunk, tf in zip(self.chunks, self.term_frequencies):
            if chunk.chunk_id in excluded or (cutoff is not None and chunk.available_at > cutoff):
                continue
            if required and not required.issubset(tf):
                continue
            length = len(chunk.terms)
            score = 0.0
            for term in qterms:
                frequency = tf.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency.get(term, 0)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                norm = frequency + self.k1 * (1.0 - self.b + self.b * length / max(1.0, self.average_length))
                score += idf * frequency * (self.k1 + 1.0) / norm
            if score > 0:
                scored.append(RetrievalResult(chunk, score))
        scored.sort(key=lambda item: (-item.score, item.chunk.available_at, item.chunk.chunk_id))
        return scored[:limit]
