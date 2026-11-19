"""Configuration loading. No machine-specific paths are stored in the repository.

Dataset roots come from the environment variable ECG_DATA_ROOT (a directory with one subdirectory per dataset,
see docs/DATASETS.md). Run outputs go to ECG_RUN_ROOT. Neither variable has a default that points at a real location.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_root() -> Path:
    v = os.environ.get("ECG_DATA_ROOT")
    if not v:
        raise EnvironmentError("ECG_DATA_ROOT is not set. Point it at the directory holding the dataset subdirectories.")
    p = Path(v)
    if not p.is_dir():
        raise NotADirectoryError(f"ECG_DATA_ROOT does not exist: {p}")
    return p


def run_root() -> Path:
    v = os.environ.get("ECG_RUN_ROOT")
    if not v:
        raise EnvironmentError("ECG_RUN_ROOT is not set. Point it at a writable directory for run outputs.")
    return Path(v)


def dataset_config() -> dict:
    return load_yaml(CONFIG_DIR / "datasets.yaml")


def label_config() -> dict:
    return load_yaml(CONFIG_DIR / "label_map_aami.yaml")
