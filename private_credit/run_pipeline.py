#!/usr/bin/env python3
"""Run private-credit data generation → credit models → ML → HTML report."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    py = root / "python"
    repo = root.parent
    scripts = [
        py / "01_data_generator.py",
        py / "02_credit_models.py",
        py / "03_ml_classifier.py",
    ]
    exe = sys.executable
    for s in scripts:
        print(f"\n--- Running {s.name} ---\n")
        subprocess.run([exe, str(s)], cwd=str(repo), check=True)

    sys.path.insert(0, str(repo / "src"))
    from private_credit_report import build_fragility_report

    build_fragility_report()


if __name__ == "__main__":
    main()
