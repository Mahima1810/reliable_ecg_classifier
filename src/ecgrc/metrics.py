"""Metrics for recall-primary abnormal-beat evaluation.

Primary: S recall and V recall. Secondary: per-class precision, F1, confusion, false negatives, false positives per hour.
Uncertainty: patient-cluster bootstrap (resample patients, recompute pooled counts). Q beats are never scored here.
"""
from __future__ import annotations

import numpy as np

from .labels import CLASSES


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = len(CLASSES)) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), np.int64)
    np.add.at(cm, (y_true.astype(int), y_pred.astype(int)), 1)
    return cm


def per_class(cm: np.ndarray, hours: float | None = None) -> dict:
    """Per-class recall, precision, F1, support, FN, FP. FP per hour if hours is given (per recording-hour)."""
    out = {}
    for k, name in enumerate(CLASSES):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        support = cm[k, :].sum()
        recall = tp / support if support else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision and recall and np.isfinite(precision) and np.isfinite(recall)) else float("nan")
        rec = {"recall": float(recall), "precision": float(precision), "f1": float(f1),
               "support": int(support), "false_negatives": int(fn), "false_positives": int(fp)}
        if hours:
            rec["fp_per_hour"] = float(fp / hours)
        out[name] = rec
    f1s = [out[c]["f1"] for c in CLASSES if np.isfinite(out[c]["f1"])]
    out["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    out["confusion"] = cm.tolist()
    return out


def patient_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, patients: np.ndarray, stat, B: int = 2000,
                         seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CI from resampling patients with replacement. stat(cm) -> float. Returns point, lo, hi, n_patients."""
    pids, inv = np.unique(patients, return_inverse=True)
    rng = np.random.default_rng(seed)
    # per-patient confusion counts
    per_p = np.zeros((len(pids), len(CLASSES), len(CLASSES)), np.int64)
    np.add.at(per_p, (inv, y_true.astype(int), y_pred.astype(int)), 1)
    point = float(stat(per_p.sum(0)))
    vals = []
    for _ in range(B):
        draw = rng.integers(0, len(pids), len(pids))
        vals.append(stat(per_p[draw].sum(0)))
    vals = np.array(vals, float)
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_patients": int(len(pids)), "B": B}


def recall_stat(cls: str):
    k = CLASSES.index(cls)

    def f(cm):
        s = cm[k, :].sum()
        return cm[k, k] / s if s else float("nan")
    return f


def precision_stat(cls: str):
    k = CLASSES.index(cls)

    def f(cm):
        s = cm[:, k].sum()
        return cm[k, k] / s if s else float("nan")
    return f
