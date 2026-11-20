"""Per-beat class probabilities for one WFDB record, given externally supplied R-peak positions.

This module does NOT detect QRS complexes. The R-peak positions must come from a detector or an annotation file, and
their source is recorded in the output. Causality: each beat's prediction uses that beat's window plus the previous
L-1 beats, and its window reaches `post` samples after the R peak (see ecgrc.lookahead for how often the next beat falls there).
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample_poly

from .checkpoint import load_checkpoint
from .config import dataset_config
from .labels import CLASSES
from .model import CNNGRU
from .records import choose_lead
from .thresholds import apply_rule
from .windows import beat_windows, causal_rr, causal_sequences


def load_signal(base: Path, dataset: str, fs_processing: float, cfg: dict | None = None):
    """Read one lead (preference order from configs/datasets.yaml) and resample to the processing rate."""
    cfg = cfg or dataset_config()
    dcfg = cfg["datasets"][dataset]
    rec = wfdb.rdrecord(str(base))
    if abs(rec.fs - dcfg["native_fs"]) > 1e-6:
        raise ValueError(f"header fs {rec.fs} != configured native_fs {dcfg['native_fs']} for {dataset}")
    li = choose_lead(list(rec.sig_name), dcfg["lead_preference"])
    x = rec.p_signal[:, li].astype(np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite samples in the selected lead")
    r = Fraction(fs_processing / rec.fs).limit_denominator(1000)
    y = resample_poly(x, r.numerator, r.denominator).astype(np.float32)
    return y, rec.sig_name[li], rec.fs


def predict_record(model: CNNGRU, signal: np.ndarray, peaks_processing: np.ndarray, *, pre: int, post: int,
                   seq_len: int, fs: float, device: torch.device, batch_size: int = 256,
                   thresholds: dict[int, float] | None = None, class_index: dict[str, int] | None = None) -> pd.DataFrame:
    """Return one row per beat with a valid window: sample, p_N, p_S, p_V, p_F, decision, lookahead flag."""
    class_index = class_index or {c: i for i, c in enumerate(CLASSES)}
    peaks = np.sort(np.asarray(peaks_processing, dtype=np.int64))
    windows, valid = beat_windows(signal, peaks, pre, post)
    kept = peaks[valid]
    w = windows[valid]
    if len(kept) == 0:
        raise ValueError("no beat has a complete window")
    rr = causal_rr(kept, fs=fs)
    dummy_labels = np.zeros(len(kept), dtype=np.int64)      # labels are not used for prediction
    x, rr_seq, mask, _ = causal_sequences(w, rr, dummy_labels, seq_len)
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[s:s + batch_size], device=device)
            rb = torch.as_tensor(rr_seq[s:s + batch_size], device=device)
            mb = torch.as_tensor(mask[s:s + batch_size], device=device)
            logits = model(xb, rb, mb)[:, -1, :]
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    p = np.concatenate(probs)
    if thresholds:
        decision_idx = apply_rule(p, thresholds, class_index)
    else:
        decision_idx = p.argmax(axis=1)
    # lookahead: does the next peak fall inside this beat's post-window?
    all_next = np.full(len(peaks), np.iinfo(np.int64).max)
    all_next[:-1] = peaks[1:]
    next_of_kept = all_next[np.searchsorted(peaks, kept)]
    looks = (next_of_kept - kept) < post
    return pd.DataFrame({"sample": kept, "p_N": p[:, class_index["N"]], "p_S": p[:, class_index["S"]],
                         "p_V": p[:, class_index["V"]], "p_F": p[:, class_index["F"]],
                         "decision": [CLASSES[i] for i in decision_idx], "next_peak_inside_window": looks})
