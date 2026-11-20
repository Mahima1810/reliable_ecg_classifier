#!/usr/bin/env python
"""Train the recall-primary CNN-GRU. Does NOT select thresholds and does NOT read test results.

Selection: the epoch with the highest validation statistic (mean average precision for S and V) is kept as
ckpt_best.pt. ckpt_last.pt is written every epoch so that an interrupted run can resume with --resume.

Refuses to run while experiment.seeds, splits.split_seed or training settings needed for reproducibility are null.

Usage:
  python scripts/train.py --config configs/experiment_recall_primary.yaml --run-name run01 --seed 101 [--resume]
  python scripts/train.py --config ... --run-name smoke --seed 0 --dry-run        # data and shapes only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgrc.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from ecgrc.config import CONFIG_DIR, data_root, dataset_config, load_yaml, run_root  # noqa: E402
from ecgrc.engine import (build_beat_table, class_weights, predict_probs, scored_only,  # noqa: E402
                          split_beats, train_epoch)
from ecgrc.labels import CLASSES, SymbolMap  # noqa: E402
from ecgrc.model import CNNGRU, count_parameters  # noqa: E402
from ecgrc.seed import set_seed  # noqa: E402
from ecgrc.splits import build_splits, check_split_integrity, holdout_leak_check  # noqa: E402
from ecgrc.windows import assemble  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="load data, build inputs and run one batch; no training")
    ap.add_argument("--max-epochs", type=int, default=None)
    a = ap.parse_args()

    exp_cfg = load_yaml(a.config)
    exp, tr, mdl = exp_cfg["experiment"], exp_cfg["training"], exp_cfg["model"]
    if not a.dry_run and exp.get("seeds") is None:
        sys.exit("experiment.seeds is null in the config; set the approved seed list before training")
    if not a.dry_run and exp_cfg["splits"].get("split_seed") is None:
        sys.exit("splits.split_seed is null in the config; set it before training so the split is recorded")
    cfg = dataset_config()
    run_dir = run_root() / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.json").write_text(json.dumps({"config": exp_cfg, "seed": a.seed}, indent=1))

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not a.dry_run and device.type != "cuda":
        sys.exit("no CUDA device visible; refusing to train on CPU silently")

    labels = SymbolMap.load(allow_provisional=bool(exp.get("allow_provisional_labels", False)))
    beats, signals, info, skipped = build_beat_table(exp["datasets"], data_root(), labels, cfg, cfg["processing_fs"])
    info.to_csv(run_dir / "records.csv", index=False)
    (run_dir / "skipped_records.json").write_text(json.dumps(skipped, indent=1))

    patient_datasets = beats.groupby("patient_id")["dataset"].apply(lambda s: sorted(set(s))).to_dict()
    split_map = build_splits(patient_datasets, seed=int(exp_cfg["splits"]["split_seed"] or a.seed),
                             fractions=tuple(exp_cfg["splits"]["fractions"]), holdout_dataset=exp.get("holdout_dataset"))
    summary = check_split_integrity(split_map, beats)
    if exp.get("holdout_dataset"):
        holdout_leak_check(split_map, beats, exp["holdout_dataset"])
    (run_dir / "split_summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "split_map.json").write_text(json.dumps(split_map, indent=0))

    sb = split_beats(beats, split_map)
    sb = scored_only(sb)
    class_index = {c: i for i, c in enumerate(CLASSES)}
    L, W = mdl["sequence_length"], mdl["window"]
    data = {}
    for s in ("train", "val"):
        part = sb[sb["split"] == s]
        x, rr, m, y, meta = assemble(part, class_index, signals, cfg["beat_pre"], cfg["beat_post"], L, cfg["processing_fs"])
        data[s] = (x, rr, m, y, meta)
    x, rr, m, y, _ = data["train"]
    weights = class_weights(y[:, -1], cap=tr["class_weight_cap"])
    print(f"train sequences {len(x)}, val sequences {len(data['val'][0])}, window {x.shape[-1]} (expected {W})", flush=True)

    model = CNNGRU(window=W, n_classes=len(CLASSES), emb=mdl["emb"], hidden=mdl["hidden"], n_layers=mdl["n_layers"],
                   dropout=mdl["dropout"]).to(device)
    print(f"parameters: {count_parameters(model)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])

    if a.dry_run:
        b = min(8, len(x))
        out = model(torch.as_tensor(x[:b], device=device), torch.as_tensor(rr[:b], device=device),
                    torch.as_tensor(m[:b], device=device))
        assert tuple(out.shape) == (b, L, len(CLASSES)), out.shape
        print(f"dry run OK: logits {tuple(out.shape)}", flush=True)
        return 0

    rng = np.random.default_rng(a.seed)
    start_epoch, best = 0, {"epoch": -1, "statistic": -np.inf}
    last_path, best_path = run_dir / "ckpt_last.pt", run_dir / "ckpt_best.pt"
    if a.resume and last_path.exists():
        payload = load_checkpoint(last_path, model, opt, map_location=device)
        start_epoch, best = payload["epoch"] + 1, payload["best"]
        print(f"resumed from epoch {payload['epoch']}", flush=True)
    log_path = run_dir / "train_log.csv"
    new_log = not log_path.exists()
    log = open(log_path, "a", newline="")
    wr = csv.writer(log)
    if new_log:
        wr.writerow(["epoch", "train_loss", "val_AP_S", "val_AP_V", "statistic", "seconds"])
    max_epochs = a.max_epochs or tr["epochs"]
    stale = 0
    for ep in range(start_epoch, max_epochs):
        t0 = time.time()
        loss = train_epoch(model, opt, x, rr, m, y, tr["batch_size"], weights.to(device), device, rng,
                           amp=(tr.get("mixed_precision", "none") == "fp16"),
                           max_grad_norm=float(tr.get("max_grad_norm", 5.0)))
        pv, yv = predict_probs(model, data["val"][0], data["val"][1], data["val"][2], data["val"][3], tr["batch_size"], device)
        ap_s = float(average_precision_score((yv == class_index["S"]).astype(int), pv[:, class_index["S"]]))
        ap_v = float(average_precision_score((yv == class_index["V"]).astype(int), pv[:, class_index["V"]]))
        stat = 0.5 * (ap_s + ap_v)
        wr.writerow([ep, f"{loss:.6f}", f"{ap_s:.6f}", f"{ap_v:.6f}", f"{stat:.6f}", f"{time.time() - t0:.1f}"])
        log.flush()
        if stat > best["statistic"]:
            best = {"epoch": ep, "statistic": stat}
            save_checkpoint(best_path, model, opt, ep, best)
            stale = 0
        else:
            stale += 1
        save_checkpoint(last_path, model, opt, ep, best)
        print(f"epoch {ep} loss {loss:.4f} AP_S {ap_s:.4f} AP_V {ap_v:.4f} stat {stat:.4f}", flush=True)
        if stale >= tr["patience"]:
            print("early stop", flush=True)
            break
    log.close()
    (run_dir / "train.done.json").write_text(json.dumps({"best": best, "seed": a.seed}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
