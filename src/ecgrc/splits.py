"""Patient-level splits with integrity checks.

Rules:
- A patient is assigned to exactly one split. A patient that appears in more than one dataset is assigned once,
  using its first dataset in sorted order, and is reported.
- Splits are stratified by dataset (each dataset contributes roughly the same fractions).
- Leave-one-dataset-out: with holdout set, every patient of that dataset goes to test and is absent from train and validation.
- Integrity is checked after construction: no patient in two splits, every beat's patient present in the split map.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


class SplitIntegrityError(RuntimeError):
    pass


def build_splits(patient_datasets: dict[str, list[str]], seed: int,
                 fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
                 holdout_dataset: str | None = None) -> dict[str, str]:
    """patient_datasets: patient_id -> list of datasets it appears in. Returns patient_id -> split."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    # group patients by their primary dataset (first in sorted order) for stratification
    primary = defaultdict(list)
    for pid in sorted(patient_datasets):
        dsets = sorted(set(patient_datasets[pid]))
        if holdout_dataset and holdout_dataset in dsets:
            assignment[pid] = "test"
            continue
        primary[dsets[0]].append(pid)
    for ds, pids in sorted(primary.items()):
        order = rng.permutation(len(pids))
        n = len(pids)
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        for k, idx in enumerate(order):
            pid = pids[idx]
            assignment[pid] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    return assignment


def check_split_integrity(split_map: dict[str, str], beats: pd.DataFrame) -> dict:
    """Raise SplitIntegrityError on any violation. Returns a summary dict."""
    bad_split = sorted({s for s in split_map.values() if s not in SPLITS})
    if bad_split:
        raise SplitIntegrityError(f"unknown split labels: {bad_split}")
    missing = sorted(set(beats["patient_id"]) - set(split_map))
    if missing:
        raise SplitIntegrityError(f"{len(missing)} patients have beats but no split (e.g. {missing[:3]})")
    # one split per patient is guaranteed by the dict; check that each patient's records agree
    per_patient = beats.groupby("patient_id")["dataset"].nunique()
    shared = per_patient[per_patient > 1]
    summary = {
        "patients": len(split_map),
        "patients_per_split": {s: sum(1 for v in split_map.values() if v == s) for s in SPLITS},
        "patients_in_more_than_one_dataset": int(len(shared)),
    }
    return summary


def holdout_leak_check(split_map: dict[str, str], beats: pd.DataFrame, holdout_dataset: str) -> None:
    """In leave-one-dataset-out runs, the held-out dataset must not appear in train or validation."""
    held = beats[beats["dataset"] == holdout_dataset]
    leaked = [p for p in set(held["patient_id"]) if split_map.get(p) != "test"]
    if leaked:
        raise SplitIntegrityError(f"{len(leaked)} held-out {holdout_dataset} patients are not in test")
