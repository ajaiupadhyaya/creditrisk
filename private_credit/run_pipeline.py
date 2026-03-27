#!/usr/bin/env python3
"""Run private-credit data generation → credit models → ML → HTML report."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _venv_python(repo: Path) -> Path | None:
    if sys.platform == "win32":
        p = repo / ".venv" / "Scripts" / "python.exe"
    else:
        p = repo / ".venv" / "bin" / "python"
    return p if p.exists() else None


def _maybe_reexec_with_venv() -> None:
    """If the user invoked this with system Python but `.venv` exists, re-run under the venv."""
    override = os.environ.get("PRIVATE_CREDIT_PYTHON")
    if override:
        return
    venv_py = _venv_python(_repo_root())
    if venv_py is None:
        return
    if Path(sys.executable).resolve() == venv_py.resolve():
        return
    script = Path(__file__).resolve()
    os.execv(str(venv_py), [str(venv_py), str(script)] + sys.argv[1:])


def _python_for_subprocess(repo: Path) -> Path:
    env = os.environ.get("PRIVATE_CREDIT_PYTHON")
    if env:
        return Path(env).expanduser().resolve()
    v = _venv_python(repo)
    return v if v is not None else Path(sys.executable)


def main() -> None:
    root = Path(__file__).resolve().parent
    py = root / "python"
    repo = root.parent
    scripts = [
        py / "01_data_generator.py",
        py / "02_credit_models.py",
        py / "03_ml_classifier.py",
    ]
    exe = _python_for_subprocess(repo)
    for s in scripts:
        print(f"\n--- Running {s.name} ---\n")
        subprocess.run([str(exe), str(s)], cwd=str(repo), check=True)

    sys.path.insert(0, str(repo / "src"))
    from private_credit_report import build_fragility_report

    build_fragility_report()


if __name__ == "__main__":
    _maybe_reexec_with_venv()
    main()
