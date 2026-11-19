"""Atomic checkpoint save and resume, with RNG state so that a resumed run continues the same random sequence."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, model, optimizer, epoch: int, best: dict, extra: dict | None = None) -> None:
    """Write to a temporary file in the same directory, then replace. A crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
               "best": best, "rng": rng_state(), "extra": extra or {}}
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, optimizer=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "rng" in payload:
        set_rng_state(payload["rng"])
    return payload
