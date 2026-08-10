"""Central storage paths for the lightweight FinPlanRAG code repository.

Large datasets, model weights and generated results live outside the GitHub
working tree.  Set ``FINPLANRAG_STORAGE_ROOT`` to override the default data
location on another machine.
"""

from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
DEFAULT_STORAGE_ROOT = Path(r"D:\Engineering\FinPlanRAG\database")
STORAGE_ROOT = Path(os.environ.get("FINPLANRAG_STORAGE_ROOT", DEFAULT_STORAGE_ROOT)).expanduser()
PREEXPERIMENTS_STORAGE = STORAGE_ROOT / "preexperiments"
DATA_ROOT = PREEXPERIMENTS_STORAGE / "data"
RESULTS_ROOT = PREEXPERIMENTS_STORAGE / "results"
MODELS_ROOT = PREEXPERIMENTS_STORAGE / "models"
TMP_ROOT = STORAGE_ROOT / "tmp"
ARCHIVE_ROOT = STORAGE_ROOT / "archive"


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def result_path(*parts: str) -> Path:
    return RESULTS_ROOT.joinpath(*parts)


def model_path(*parts: str) -> Path:
    return MODELS_ROOT.joinpath(*parts)


def require_storage_root() -> Path:
    """Return the configured root or raise a precise setup error."""
    if not STORAGE_ROOT.exists():
        raise FileNotFoundError(
            f"FinPlanRAG external storage does not exist: {STORAGE_ROOT}. "
            "Set FINPLANRAG_STORAGE_ROOT to the migrated database directory."
        )
    return STORAGE_ROOT
