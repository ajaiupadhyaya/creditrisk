"""Project paths for the private-credit fragility pipeline."""
from __future__ import annotations

import os
from pathlib import Path


def private_credit_root() -> Path:
    """Directory `private_credit/` (parent of `python/`)."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Default: `private_credit/data/`. Override with env `PRIVATE_CREDIT_DATA_DIR`."""
    override = os.environ.get("PRIVATE_CREDIT_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return private_credit_root() / "data"


def ensure_data_dir() -> Path:
    d = get_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d
