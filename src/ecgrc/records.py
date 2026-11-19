"""WFDB record loading for the beat-labelled datasets, with explicit lead, rate, label and patient handling.

Every returned beat carries its provenance: dataset, record, lead, native sample index, original symbol, class and patient.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

from .config import CONFIG_DIR, dataset_config
from .labels import SymbolMap


class DataError(RuntimeError):
    pass


@dataclass
class Record:
    dataset: str
    name: str
    patient_id: str
    lead: str
    native_fs: float
    fs: float                   # processing rate
    signal: np.ndarray          # (n,) float32 in mV, resampled to fs
    beats: pd.DataFrame         # one row per beat symbol that is a beat in the label map
    n_nonbeat: int              # annotation symbols that are not beats (kept for audit)
    duration_s: float


@lru_cache(maxsize=1)
def patient_map() -> dict:
    path = CONFIG_DIR / "patient_map.csv"
    if not path.exists():
        raise DataError(f"patient map missing: {path}")
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["record"])
            if key in out:
                raise DataError(f"duplicate patient map entry {key}")
            out[key] = (row["patient_id"], row["evidence"])
    return out


def patient_of(dataset: str, record: str) -> tuple[str, str]:
    pm = patient_map()
    key = (dataset, record)
    if key not in pm:
        raise DataError(f"no patient mapping for {dataset}/{record}; add it to configs/patient_map.csv")
    return pm[key]


def _resample_factors(native_fs: float, fs: float) -> tuple[int, int]:
    r = Fraction(fs / native_fs).limit_denominator(1000)
    return r.numerator, r.denominator


def choose_lead(sig_names: list[str], preference: list[str]) -> int:
    for want in preference:
        if want in sig_names:
            return sig_names.index(want)
    raise DataError(f"none of the preferred leads {preference} present in {sig_names}")


def load_record(dataset: str, base: Path, fs_processing: float, labels: SymbolMap, cfg: dict | None = None) -> Record:
    cfg = cfg or dataset_config()
    dcfg = cfg["datasets"][dataset]
    rec = wfdb.rdrecord(str(base))
    name = base.name
    if abs(rec.fs - dcfg["native_fs"]) > 1e-6:
        raise DataError(f"{dataset}/{name}: header fs {rec.fs} != configured native_fs {dcfg['native_fs']}")
    li = choose_lead(list(rec.sig_name), dcfg["lead_preference"])
    x = rec.p_signal[:, li].astype(np.float64)
    if not np.all(np.isfinite(x)):
        raise DataError(f"{dataset}/{name}: non-finite samples in lead {rec.sig_name[li]}")
    up, down = _resample_factors(rec.fs, fs_processing)
    y = resample_poly(x, up, down).astype(np.float32)

    ann = wfdb.rdann(str(base), "atr")
    rows, n_nonbeat = [], 0
    for samp, sym in zip(ann.sample, ann.symbol):
        cls = labels.classify(sym, dataset=dataset, record=name, sample=int(samp))
        if cls is None:
            n_nonbeat += 1
            continue
        rows.append((int(round(samp * fs_processing / rec.fs)), int(samp), sym, cls))
    patient_id, _ = patient_of(dataset, name)
    beats = pd.DataFrame(rows, columns=["sample", "native_sample", "symbol", "aami"])
    beats["dataset"] = dataset
    beats["record"] = name
    beats["lead"] = rec.sig_name[li]
    beats["patient_id"] = patient_id
    beats = beats[beats["sample"] < len(y)].reset_index(drop=True)
    return Record(dataset=dataset, name=name, patient_id=patient_id, lead=rec.sig_name[li], native_fs=rec.fs,
                  fs=fs_processing, signal=y, beats=beats, n_nonbeat=n_nonbeat, duration_s=len(x) / rec.fs)


def list_records(dataset: str, root: Path, cfg: dict | None = None) -> tuple[list[Path], dict]:
    """Return (records to load, skipped records with reasons). Excluded records are listed in the config."""
    cfg = cfg or dataset_config()
    dcfg = cfg["datasets"][dataset]
    d = root / dcfg["subdir"]
    if not d.is_dir():
        raise DataError(f"dataset folder missing: {d}")
    excluded = {str(k): str(v) for k, v in (dcfg.get("excluded_records") or {}).items()}
    bases, skipped = [], {}
    for hea in sorted(d.glob("*.hea")):
        name = hea.stem
        if name in excluded:
            skipped[name] = f"excluded by config: {excluded[name]}"
        elif not hea.with_suffix(".dat").exists():
            skipped[name] = "signal file missing"
        else:
            bases.append(hea.with_suffix(""))
    return bases, skipped
