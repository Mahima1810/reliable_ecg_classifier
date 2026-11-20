"""Measure how much future signal the beat window contains.

The beat window spans `pre` samples before the R peak and `post` samples after it. A beat's window is causal only if it
ends before the next beat's R peak. Any beat whose next R peak lies inside its post-window "looks ahead" by that amount.
This module reports that fraction; it does not change the window.
"""
from __future__ import annotations

import numpy as np


def lookahead_report(samples: np.ndarray, post: int, fs: float) -> dict:
    """samples: sorted R-peak positions (processing-rate samples) of one record. post: samples after the peak in the window."""
    s = np.sort(np.asarray(samples, dtype=np.int64))
    n = len(s)
    if n < 2:
        return {"beats": n, "beats_with_lookahead": 0, "fraction": float("nan"), "max_lookahead_ms": 0.0,
                "window_post_ms": 1000.0 * post / fs}
    gap = np.diff(s)                       # distance from beat i to beat i+1
    looks = gap < post                     # the next R peak falls inside beat i's post-window
    lookahead_samples = np.where(looks, post - gap, 0)
    return {"beats": int(n - 1), "beats_with_lookahead": int(looks.sum()),
            "fraction": float(looks.mean()) if len(looks) else float("nan"),
            "max_lookahead_ms": float(1000.0 * lookahead_samples.max() / fs),
            "window_post_ms": float(1000.0 * post / fs)}
