"""Portable defaults; set SHARED_ONSET_ROOT or SHARED_ONSET_DATA_ROOT to override."""
import os
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("SHARED_ONSET_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    return source_root if (source_root / "pyproject.toml").is_file() else Path.cwd()


def shared_data_root() -> Path:
    return Path(os.environ.get("SHARED_ONSET_DATA_ROOT", project_root() / "data")).expanduser().resolve()


def processed_root() -> Path:
    return shared_data_root() / "processed" / "paper2_tri"
