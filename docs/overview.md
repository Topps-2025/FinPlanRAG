# FinPlan-RAG overview

## Research question

When a financial question requires evidence across entities, relationships, filings, and time, does an explicit obligation state improve the next retrieval and stopping decision under a fixed cutoff and budget?

## State and action contract

The planner keeps four dimensions explicit: entity, filing type, fiscal year, and fiscal period. A retrieval action targets one unresolved obligation. A result is usable only when its `available_at` timestamp is no later than the question cutoff. The planner returns the covered document IDs, query count, closure flag, and any future-document IDs observed by diagnostics.

## What is and is not claimed

The package demonstrates a reproducible planning mechanism and exposes controlled ablations. It does not claim that obligation-aware retrieval is universally better than strong adaptive RAG, that document closure guarantees answer accuracy, or that pilot results establish a benchmark win. Those claims require larger frozen datasets, stronger readers, and independent reproduction.
