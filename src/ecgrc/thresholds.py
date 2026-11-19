"""Operating thresholds for S and V alarms. Selection uses VALIDATION data only.

Operating rule (fixed, documented in docs/PROTOCOL.md):
    triggered = {c in {S, V} : p_c >= t_c}
    if triggered: predict the triggered class with the highest probability
    else:         predict the class with the highest probability among {N, F}

Threshold selection for one class c:
- The recall floor R_c and the false-positive budget are REQUIRED inputs from the experiment config.
  If either is None, selection refuses to run. Nothing in this module chooses a clinical value.
- Among thresholds with validation recall >= R_c (and FP/hour within budget, if given), choose the one with the
  highest precision; ties go to the higher threshold (fewer alarms).
- If no threshold meets the constraints, the result is infeasible and nothing is selected.
"""
from __future__ import annotations

import numpy as np


class ThresholdError(RuntimeError):
    pass


def curve(scores: np.ndarray, is_class: np.ndarray, hours: float, grid: np.ndarray | None = None) -> list[dict]:
    """Recall, precision and alarms per hour over a threshold grid (validation data only)."""
    if grid is None:
        grid = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 401)))
    rows = []
    pos = is_class.astype(bool)
    n_pos = pos.sum()
    for t in grid:
        alarm = scores >= t
        tp = int((alarm & pos).sum())
        fp = int((alarm & ~pos).sum())
        rows.append({"threshold": float(t),
                     "recall": tp / n_pos if n_pos else float("nan"),
                     "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
                     "alarms_per_hour": (tp + fp) / hours if hours else float("nan"),
                     "tp": tp, "fp": fp, "fn": int(n_pos - tp)})
    return rows


def select_threshold(scores: np.ndarray, is_class: np.ndarray, hours: float, recall_floor: float | None,
                     fp_per_hour_budget: float | None = None) -> dict:
    if recall_floor is None:
        raise ThresholdError("recall_floor is not set in the experiment config; threshold selection refuses to run")
    if not (0.0 < recall_floor <= 1.0):
        raise ThresholdError(f"recall_floor {recall_floor} out of range")
    if fp_per_hour_budget is not None and fp_per_hour_budget < 0:
        raise ThresholdError("fp_per_hour_budget must be non-negative")
    rows = curve(scores, is_class, hours)
    feasible = [r for r in rows if np.isfinite(r["recall"]) and r["recall"] >= recall_floor
                and (fp_per_hour_budget is None or (np.isfinite(r["alarms_per_hour"]) and
                     (r["fp"] / hours) <= fp_per_hour_budget))]
    if not feasible:
        return {"feasible": False, "recall_floor": recall_floor, "fp_per_hour_budget": fp_per_hour_budget,
                "threshold": None, "note": "no validation threshold meets the constraints"}
    best = max(feasible, key=lambda r: (np.nan_to_num(r["precision"], nan=-1.0), r["threshold"]))
    return {"feasible": True, "recall_floor": recall_floor, "fp_per_hour_budget": fp_per_hour_budget,
            "threshold": best["threshold"], "validation": best}


def apply_rule(probs: np.ndarray, thresholds: dict[int, float], class_index: dict[str, int]) -> np.ndarray:
    """probs: (n, C) with columns in class_index order. thresholds: {class index: t} for S and V."""
    n, C = probs.shape
    pred = np.empty(n, np.int64)
    trig_idx = list(thresholds.keys())
    for i in range(n):
        trig = [k for k in trig_idx if probs[i, k] >= thresholds[k]]
        if trig:
            pred[i] = max(trig, key=lambda k: probs[i, k])
        else:
            allowed = [class_index["N"], class_index["F"]]
            pred[i] = max(allowed, key=lambda k: probs[i, k])
    return pred
