"""Wrapper that starts the AnchorEvolve MCP stdio server.

Claude Code's project ``.mcp.json`` points here.  The wrapper loads ``.env``
before importing the server so that the API key is available to the engine,
and changes to the repository root so relative case and result paths resolve
consistently regardless of the caller's working directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from anchor_engine import load_env  # noqa: E402

load_env(REPO_ROOT / ".env")

from anchor_mcp import main  # noqa: E402

if __name__ == "__main__":
    main()
