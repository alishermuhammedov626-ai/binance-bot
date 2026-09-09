"""Bundle the bot into a single runnable file.

``zipapp`` archives the real package rather than concatenating sources, so the
bundle is byte-for-byte the code that the test suite runs -- no import
rewriting, no chance of the single-file build drifting from the repo.

    python scripts/build_pyz.py
    python dist/smcbot.pyz backtest --synthetic --days 60
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipapp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "dist", "smcbot.pyz")

MAIN = '''import sys

from smcbot.cli import main

if __name__ == "__main__":
    sys.exit(main())
'''


def build(out: str = OUT) -> str:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        shutil.copytree(
            os.path.join(ROOT, "smcbot"),
            os.path.join(staging, "smcbot"),
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        with open(os.path.join(staging, "__main__.py"), "w", encoding="utf-8") as fh:
            fh.write(MAIN)
        zipapp.create_archive(staging, out, interpreter="/usr/bin/env python3")
    os.chmod(out, 0o755)
    return out


if __name__ == "__main__":
    path = build()
    print(f"built {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    sys.exit(0)
