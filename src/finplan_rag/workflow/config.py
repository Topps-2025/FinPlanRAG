"""Workflow configuration.

All paths are resolved relative to the repository root. API keys are never
stored in the config; the LLM section names an environment variable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class WorkflowConfig:
    raw: dict[str, Any]
    source: Path

    def path(self, key: str) -> Path:
        value = self.raw.get(key)
        if not value:
            raise KeyError(f"missing config key: {key}")
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def data_path(self) -> Path:
        return self.path("data")

    @property
    def results_root(self) -> Path:
        return self.path("results_root")

    @property
    def llm(self) -> dict[str, Any]:
        return dict(self.raw.get("llm") or {})

    @property
    def sequential(self) -> dict[str, Any]:
        return dict(self.raw.get("sequential") or {})

    def section(self, key: str) -> Any:
        return self.raw.get(key)


def load_config(path: str | Path) -> WorkflowConfig:
    source = Path(path)
    if not source.is_absolute():
        source = REPO_ROOT / source
    payload = json.loads(source.read_text(encoding="utf-8"))
    return WorkflowConfig(raw=payload, source=source)
