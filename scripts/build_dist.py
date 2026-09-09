"""Build the distributable artifacts.

    python scripts/build_dist.py

Produces:
    dist/smcbot.pyz        single runnable file (zipapp of the real package)
    dist/smcbot-bot.tar.gz full source; tar preserves the executable bit on
                           scripts/*.sh, which a zip does not survive
    dist/sample_m1.csv     example of the expected CSV input format
"""
from __future__ import annotations

import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

INCLUDE_DIRS = ["smcbot", "tests", "scripts", "deploy"]
INCLUDE_FILES = ["README.md", "pyproject.toml", "requirements.txt", ".gitignore"]
PREFIX = "smcbot-bot"


def _skip(path: str) -> bool:
    return "__pycache__" in path or path.endswith(".pyc")


def build_tarball(out: str) -> str:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for d in INCLUDE_DIRS:
            for root, dirs, files in os.walk(os.path.join(ROOT, d)):
                dirs[:] = [x for x in dirs if x != "__pycache__"]
                for f in sorted(files):
                    p = os.path.join(root, f)
                    if _skip(p):
                        continue
                    rel = os.path.relpath(p, ROOT)
                    tar.add(p, arcname=os.path.join(PREFIX, rel))
        for f in INCLUDE_FILES:
            p = os.path.join(ROOT, f)
            if os.path.exists(p):
                tar.add(p, arcname=os.path.join(PREFIX, f))
        sample = os.path.join(ROOT, "dist", "sample_m1.csv")
        if os.path.exists(sample):
            tar.add(sample, arcname=os.path.join(PREFIX, "data", "sample_m1.csv"))
    return out


def main() -> int:
    from smcbot.data import loader, synthetic
    from scripts.build_pyz import build as build_pyz

    dist = os.path.join(ROOT, "dist")
    os.makedirs(dist, exist_ok=True)

    sample = os.path.join(dist, "sample_m1.csv")
    rows = loader.save_csv(sample, synthetic.generate(60 * 24 * 3, seed=42))
    print(f"sample   : {sample} ({rows} rows)")

    pyz = build_pyz()
    print(f"bundle   : {pyz} ({os.path.getsize(pyz) / 1024:.0f} KB)")

    tar = build_tarball(os.path.join(dist, "smcbot-bot.tar.gz"))
    print(f"source   : {tar} ({os.path.getsize(tar) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
