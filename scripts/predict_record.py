#!/usr/bin/env python
"""Predict per-beat probabilities for one WFDB record from an external R-peak list.

The R-peak list is a CSV with a column `sample` (native-rate sample indices of the record). It can come from an annotation
export or from a detector. This script does not detect QRS complexes.

Usage:
  python scripts/predict_record.py --config configs/experiment_recall_primary.yaml \
      --checkpoint $ECG_RUN_ROOT/run01/ckpt_best.pt --dataset mitdb \
      --record /path/to/mitdb/100 --peaks peaks_100.csv --out predictions_100.csv \
      [--thresholds $ECG_RUN_ROOT/run01/frozen_thresholds.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgrc.checkpoint import load_checkpoint  # noqa: E402
from ecgrc.config import dataset_config, load_yaml  # noqa: E402
from ecgrc.labels import CLASSES  # noqa: E402
from ecgrc.model import CNNGRU  # noqa: E402
from ecgrc.predict import load_signal, predict_record  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--record", required=True, help="WFDB record path without extension")
    ap.add_argument("--peaks", required=True, help="CSV with column 'sample' (native-rate R-peak indices)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default=None, help="frozen_thresholds.json (optional; otherwise argmax)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    exp = load_yaml(a.config)
    mdl = exp["model"]
    cfg = dataset_config()
    fs = float(cfg["processing_fs"])
    pre, post = int(cfg["beat_pre"]), int(cfg["beat_post"])

    signal, lead, native_fs = load_signal(Path(a.record), a.dataset, fs, cfg)
    peaks_native = pd.read_csv(a.peaks)["sample"].to_numpy()
    peaks = (peaks_native * fs / native_fs).round().astype("int64")

    device = torch.device(a.device)
    model = CNNGRU(window=mdl["window"], n_classes=len(CLASSES), emb=mdl["emb"], hidden=mdl["hidden"],
                   n_layers=mdl["n_layers"], dropout=mdl["dropout"]).to(device)
    load_checkpoint(Path(a.checkpoint), model, None, map_location=device)

    thresholds = None
    if a.thresholds:
        fz = json.loads(Path(a.thresholds).read_text())
        class_index = {c: i for i, c in enumerate(CLASSES)}
        thresholds = {class_index[c]: float(t) for c, t in fz["thresholds"].items()}

    df = predict_record(model, signal, peaks, pre=pre, post=post, seq_len=mdl["sequence_length"], fs=fs,
                        device=device, thresholds=thresholds)
    df["native_sample"] = (df["sample"] * native_fs / fs).round().astype("int64")
    df["lead"] = lead
    df["fs_processing"] = fs
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} beats to {a.out}; {int(df['next_peak_inside_window'].sum())} with the next R peak inside the window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
