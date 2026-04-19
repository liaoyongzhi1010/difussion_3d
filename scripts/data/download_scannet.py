"""Pointer shim for scannet.download.sh. See docs/superpowers/specs/datasets/README.md."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    script = Path(__file__).resolve().parents[2] / "docs" / "superpowers" / "specs" / "datasets" / "scannet.download.sh"
    if not script.exists():
        print(f"missing {script}")
        return 1
    return subprocess.call(["bash", str(script), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
