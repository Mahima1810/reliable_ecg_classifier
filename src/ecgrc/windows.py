"""Beat windows, causal RR features and causal beat sequences.

Causality: the sequence for beat i contains beats i-L+1..i only. No beat after i is used.
Q beats are removed before sequence construction (they are not context and are never scored).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RR_CAP_S = 3.0


def beat_windows(signal: np.ndarray, samples: np.ndarray, pre: int, post: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (windows (n, pre+post) float32, valid (n,) bool). Beats too close to the record edges are invalid."""
    n = len(samples)
    out = np.zeros((n, pre + post), np.float32)
    valid = np.zeros(n, bool)
    for i, s in enumerate(samples):
        lo, hi = int(s) - pre, int(s) + post
        if lo < 0 or hi > len(signal):
            continue
        out[i] = signal[lo:hi]
        valid[i] = True
    return out, valid


def causal_rr(samples: np.ndarray, fs: float) -> np.ndarray:
    """(n,): RR interval to the previous beat in seconds, capped at RR_CAP_S. The first beat gets 0.

    Only past beats are used, so this feature is causal.
    """
    s = np.asarray(samples, dtype=np.float64)
    prev_rr = np.zeros(len(s))
    if len(s) > 1:
        prev_rr[1:] = np.diff(s) / fs
    return np.minimum(prev_rr, RR_CAP_S).astype(np.float32)


def causal_sequences(windows: np.ndarray, rr_prev: np.ndarray, labels: np.ndarray, L: int):
    """Yield causal sequences: for each beat i, the L most recent beats ending at i.

    Returns arrays: x (N, L, W), rr (N, L, 1), mask (N, L) True where a real beat is present, y (N, L) class index
    for each position (-1 for padding). Only the last position is the scored beat when used for evaluation.
    """
    n, W = windows.shape
    x = np.zeros((n, L, W), np.float32)
    rr = np.zeros((n, L, 1), np.float32)
    mask = np.zeros((n, L), bool)
    y = np.full((n, L), -1, np.int64)
    for i in range(n):
        lo = max(0, i - L + 1)
        k = i - lo + 1
        x[i, L - k:] = windows[lo:i + 1]
        rr[i, L - k:, 0] = rr_prev[lo:i + 1]
        mask[i, L - k:] = True
        y[i, L - k:] = labels[lo:i + 1]
    return x, rr, mask, y


def assemble(beats: pd.DataFrame, class_index: dict[str, int], signals: dict, pre: int, post: int, L: int, fs: float):
    """Build model inputs for one or more records.

    beats: rows with dataset, record, sample, aami, lead, patient_id (Q rows already removed).
    signals: (dataset, record) -> resampled signal at processing rate fs.
    Returns (x, rr, mask, y, metadata) with one row per scored beat.
    """
    xs, rrs, masks, ys, metas = [], [], [], [], []
    for (ds, rec), g in beats.groupby(["dataset", "record"], sort=False):
        g = g.sort_values("sample").reset_index(drop=True)
        sig = signals[(ds, rec)]
        w, valid = beat_windows(sig, g["sample"].to_numpy(), pre, post)
        g = g[valid].reset_index(drop=True)
        w = w[valid]
        if len(g) == 0:
            continue
        rr_prev = causal_rr(g["sample"].to_numpy(), fs=fs)
        lab = np.array([class_index[c] for c in g["aami"]], dtype=np.int64)
        x, rr, m, y = causal_sequences(w, rr_prev, lab, L)
        xs.append(x); rrs.append(rr); masks.append(m); ys.append(y)
        metas.append(g[["dataset", "record", "patient_id", "sample", "aami", "symbol", "lead"]])
    if not xs:
        raise ValueError("no valid beats after windowing")
    return (np.concatenate(xs), np.concatenate(rrs), np.concatenate(masks), np.concatenate(ys),
            pd.concat(metas, ignore_index=True))
