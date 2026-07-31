from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "configs" / "default_ja.yaml"


def experiments_root() -> Path:
    path = project_root() / "data" / "experiments"
    path.mkdir(parents=True, exist_ok=True)
    return path

