#!/usr/bin/env python
"""Evaluation in two strict phases.

Phase 1 (validation only): select S and V thresholds from validation data with the recall floors and FP budget in the
config. Writes frozen_thresholds.json with hashes of the config, checkpoint and validation counts. Refuses to overwrite.

Phase 2 (test, ONE time): apply the frozen thresholds to the test split and write test_report.json.
Refuses if the thresholds are missing or stale (config or checkpoint changed), or if test_report.json already exists.

Usage:
  python scripts/evaluate.py --config configs/experiment_recall_primary.yaml --run-name run01 --phase validate
  python scripts/evaluate.py --config configs/experiment_recall_primary.yaml --run-name run01 --phase test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgrc.checkpoint import load_checkpoint  # noqa: E402
from ecgrc.config import data_root, dataset_config, load_yaml, run_root  # noqa: E402
from ecgrc.engine import build_beat_table, predict_probs, scored_only, split_beats  # noqa: E402
from ecgrc.labels import CLASSES, SymbolMap  # noqa: E402
from ecgrc.metrics import confusion, patient_bootstrap_ci, per_class, precision_stat, recall_stat  # noqa: E402
from ecgrc.model import CNNGRU  # noqa: E402
from ecgrc.splits import build_splits  # noqa: E402
from ecgrc.thresholds import apply_rule, select_threshold  # noqa: E402
from ecgrc.windows import assemble  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_split_data(run_dir: Path, exp_cfg: dict, cfg: dict, labels: SymbolMap, split: str):
    """Rebuild the split from the same seed and datasets the training run used (recorded in split_map.json)."""
    split_map = json.loads((run_dir / "split_map.json").read_text())
    beats, signals, info, _ = build_beat_table(exp_cfg["experiment"]["datasets"], data_root(), labels, cfg,
                                              cfg["processing_fs"])
    sb = scored_only(split_beats(beats, split_map))
    part = sb[sb["split"] == split]
    class_index = {c: i for i, c in enumerate(CLASSES)}
    mdl = exp_cfg["model"]
    x, rr, m, y, meta = assemble(part, class_index, signals, cfg["beat_pre"], cfg["beat_post"],
                                 mdl["sequence_length"], cfg["processing_fs"])
    hours = float(info[info["record"].isin(part["record"].unique())]["duration_s"].sum() / 3600.0)
    return x, rr, m, y, meta, hours


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--phase", choices=["validate", "test"], required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    exp_cfg = load_yaml(a.config)
    tsel = exp_cfg["threshold_selection"]
    cfg = dataset_config()
    run_dir = run_root() / a.run_name
    ckpt = run_dir / "ckpt_best.pt"
    if not ckpt.exists():
        sys.exit(f"missing checkpoint {ckpt}")
    mdl = exp_cfg["model"]
    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    model = CNNGRU(window=mdl["window"], n_classes=len(CLASSES), emb=mdl["emb"], hidden=mdl["hidden"],
                   n_layers=mdl["n_layers"], dropout=mdl["dropout"]).to(device)
    load_checkpoint(ckpt, model, None, map_location=device)
    labels = SymbolMap.load(allow_provisional=bool(exp_cfg["experiment"].get("allow_provisional_labels", False)))
    class_index = {c: i for i, c in enumerate(CLASSES)}
    tr = exp_cfg["training"]
    ckpt_hash, cfg_hash = sha256(ckpt), sha256_text(exp_cfg)

    if a.phase == "validate":
        if (run_dir / "frozen_thresholds.json").exists():
            sys.exit("frozen_thresholds.json already exists; refusing to re-select (delete only with a documented reason)")
        x, rr, m, y, meta, hours = load_split_data(run_dir, exp_cfg, cfg, labels, "val")
        probs, yt = predict_probs(model, x, rr, m, y, tr["batch_size"], device)
        chosen = {}
        for cls, floor_key in (("S", "S_recall_floor"), ("V", "V_recall_floor")):
            k = class_index[cls]
            sel = select_threshold(probs[:, k], (yt == k), hours, tsel[floor_key], tsel["fp_per_hour_budget"])
            chosen[cls] = sel
        if not all(chosen[c]["feasible"] for c in chosen):
            (run_dir / "threshold_selection_infeasible.json").write_text(json.dumps(chosen, indent=1, default=float))
            sys.exit("threshold selection infeasible on validation data; nothing frozen")
        out = {"thresholds": {c: chosen[c]["threshold"] for c in chosen}, "selection": chosen, "validation_hours": hours,
               "config_sha256": cfg_hash, "checkpoint_sha256": ckpt_hash, "checkpoint": str(ckpt.name),
               "validation_sequences": int(len(yt)), "rule": "see docs/PROTOCOL.md (operating rule)"}
        (run_dir / "frozen_thresholds.json").write_text(json.dumps(out, indent=1, default=float))
        print(json.dumps(out["thresholds"]), flush=True)
        return 0

    # ---- test phase
    if (run_dir / "test_report.json").exists():
        sys.exit("test_report.json already exists: the test set is evaluated once per run; refusing")
    fz = run_dir / "frozen_thresholds.json"
    if not fz.exists():
        sys.exit("frozen_thresholds.json missing: run --phase validate first")
    fzj = json.loads(fz.read_text())
    if fzj["config_sha256"] != cfg_hash or fzj["checkpoint_sha256"] != ckpt_hash:
        sys.exit("frozen thresholds are stale (config or checkpoint changed); refusing")
    thr = {class_index[c]: float(t) for c, t in fzj["thresholds"].items()}
    x, rr, m, y, meta, hours = load_split_data(run_dir, exp_cfg, cfg, labels, "test")
    probs, yt = predict_probs(model, x, rr, m, y, tr["batch_size"], device)
    pred = apply_rule(probs, thr, class_index)
    patients = meta["patient_id"].to_numpy()
    cm = confusion(yt, pred)
    report = {"run": a.run_name, "thresholds": fzj["thresholds"], "test_hours": hours,
              "per_class": per_class(cm, hours=hours), "config_sha256": cfg_hash, "checkpoint_sha256": ckpt_hash,
              "patient_ci": {}, "per_dataset": {}, "per_lead": {}}
    B = tsel["bootstrap"]["B"]
    for cls in ("S", "V"):
        report["patient_ci"][f"{cls}_recall"] = patient_bootstrap_ci(yt, pred, patients, recall_stat(cls), B=B, seed=0)
        report["patient_ci"][f"{cls}_precision"] = patient_bootstrap_ci(yt, pred, patients, precision_stat(cls), B=B, seed=0)
    for ds in meta["dataset"].unique():
        msk = (meta["dataset"] == ds).to_numpy()
        report["per_dataset"][ds] = {"n": int(msk.sum()), "per_class": per_class(confusion(yt[msk], pred[msk]))}
    for ld in meta["lead"].unique():
        msk = (meta["lead"] == ld).to_numpy()
        report["per_lead"][ld] = {"n": int(msk.sum()), "per_class": per_class(confusion(yt[msk], pred[msk]))}
    (run_dir / "test_report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({"S_recall": report["per_class"]["S"]["recall"], "V_recall": report["per_class"]["V"]["recall"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
