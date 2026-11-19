#!/usr/bin/env python
"""Validate the data pipeline before any training. Read-only on the dataset folders.

Checks: label-map consistency; dataset folders and records; native sampling rate per record; lead availability;
annotation symbols (no unmapped symbols); patient-map coverage; patient-level split integrity; beat counts per class.

Usage:
  ECG_DATA_ROOT=/path/to/data ECG_RUN_ROOT=/path/to/runs \
    python scripts/validate_data.py --config configs/experiment_recall_primary.yaml [--datasets mitdb svdb]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgrc.config import data_root, dataset_config, load_yaml, run_root  # noqa: E402
from ecgrc.engine import build_beat_table  # noqa: E402
from ecgrc.labels import SymbolMap, check_map_consistency  # noqa: E402
from ecgrc.splits import build_splits, check_split_integrity  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="report path (default: $ECG_RUN_ROOT/validation/data_report.json)")
    a = ap.parse_args()

    cfg = dataset_config()
    exp = load_yaml(a.config)["experiment"]
    datasets = a.datasets or exp["datasets"]
    report = {"config": a.config, "datasets": datasets, "problems": []}

    problems = check_map_consistency()
    report["label_map_problems"] = problems
    if problems:
        report["problems"].extend(problems)

    labels = SymbolMap.load(allow_provisional=bool(exp.get("allow_provisional_labels", False)))
    try:
        beats, signals, info, skipped = build_beat_table(datasets, data_root(), labels, cfg, cfg["processing_fs"])
    except Exception as e:  # report and stop: a failed load is a failed validation
        report["problems"].append(f"load failed: {type(e).__name__}: {e}")
        return _write(report, a.out, exit_code=1)

    report["records_loaded"] = info.groupby("dataset").size().to_dict()
    report["records_skipped"] = skipped
    report["native_fs_seen"] = {k: sorted(set(v)) for k, v in info.groupby("dataset")["native_fs"].apply(list).items()}
    report["leads_seen"] = {k: sorted(set(v)) for k, v in info.groupby("dataset")["lead"].apply(list).items()}
    report["beats_by_class"] = {ds: beats[beats["dataset"] == ds]["aami"].value_counts().to_dict() for ds in datasets}
    report["hours"] = (info.groupby("dataset")["duration_s"].sum() / 3600).round(3).to_dict()

    patient_datasets = beats.groupby("patient_id")["dataset"].apply(lambda s: sorted(set(s))).to_dict()
    split_map = build_splits(patient_datasets, seed=int(exp.get("split_seed") or 0),
                             fractions=tuple(load_yaml(a.config)["splits"]["fractions"]),
                             holdout_dataset=exp.get("holdout_dataset"))
    try:
        report["split_summary"] = check_split_integrity(split_map, beats)
    except Exception as e:
        report["problems"].append(f"split integrity: {e}")

    report["status"] = "PASS" if not report["problems"] else "FAIL"
    return _write(report, a.out, exit_code=0 if report["status"] == "PASS" else 1)


def _write(report: dict, out: str | None, exit_code: int) -> int:
    path = Path(out) if out else run_root() / "validation" / "data_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("status", "problems", "records_loaded", "split_summary")}, indent=1, default=str))
    print(f"report: {path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
