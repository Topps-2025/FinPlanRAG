"""Build a SHA-256 inventory for the external FinPlanRAG database."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from storage_paths import STORAGE_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not STORAGE_ROOT.exists():
        raise FileNotFoundError(STORAGE_ROOT)
    output = STORAGE_ROOT / "storage_manifest_v1.json"
    rows = []
    for path in sorted(item for item in STORAGE_ROOT.rglob("*") if item.is_file() and item != output):
        rows.append(
            {
                "relative_path": path.relative_to(STORAGE_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    result = {
        "schema": "finplanrag-external-storage-manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "storage_root": str(STORAGE_ROOT),
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "assets": rows,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": result["files"], "bytes": result["bytes"], "manifest": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
