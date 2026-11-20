"""Data assembly and training/prediction primitives. Test data is not touched by anything in this module except
`predict_probs` when called from evaluate.py after thresholds are frozen."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .labels import CLASSES, EXCLUDED, SymbolMap
from .records import list_records, load_record
from .windows import assemble


def build_beat_table(dataset_names: list[str], root, labels: SymbolMap, cfg: dict, processing_fs: float):
    """Load every configured record. Returns (beats, signals, record_info, skipped)."""
    beats_all, signals, info, skipped_all = [], {}, [], {}
    for ds in dataset_names:
        bases, skipped = list_records(ds, root, cfg)
        skipped_all.update({f"{ds}/{k}": v for k, v in skipped.items()})
        for base in bases:
            rec = load_record(ds, base, processing_fs, labels, cfg)
            signals[(ds, rec.name)] = rec.signal
            info.append({"dataset": ds, "record": rec.name, "patient_id": rec.patient_id, "lead": rec.lead,
                         "native_fs": rec.native_fs, "duration_s": rec.duration_s, "beats": len(rec.beats),
                         "nonbeat_symbols": rec.n_nonbeat})
            beats_all.append(rec.beats)
    beats = pd.concat(beats_all, ignore_index=True)
    return beats, signals, pd.DataFrame(info), skipped_all


def split_beats(beats: pd.DataFrame, split_map: dict[str, str]) -> pd.DataFrame:
    out = beats.copy()
    out["split"] = out["patient_id"].map(split_map)
    if out["split"].isna().any():
        raise ValueError("beats without a split assignment")
    return out


def scored_only(beats: pd.DataFrame) -> pd.DataFrame:
    """Drop Q (never scored, never context). Asserts no unmapped classes remain."""
    bad = set(beats["aami"]) - set(CLASSES) - set(EXCLUDED)
    if bad:
        raise ValueError(f"unexpected classes {bad}")
    return beats[beats["aami"].isin(CLASSES)].reset_index(drop=True)


def class_weights(y: np.ndarray, n_classes: int = len(CLASSES), cap: float = 10.0) -> torch.Tensor:
    """Inverse-frequency weights over scored labels only (labels < 0 are unscored and ignored), capped."""
    y = np.asarray(y)
    counts = np.bincount(y[y >= 0].astype(np.int64), minlength=n_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    w = np.minimum(w / w[w > 0].mean(), cap)
    return torch.tensor(w, dtype=torch.float32)


def train_epoch(model, opt, x, rr, mask, y, batch_size: int, weights: torch.Tensor, device, rng: np.random.Generator,
                max_steps: int | None = None, amp: bool = False, max_grad_norm: float = 5.0) -> float:
    """One pass over the training sequences. amp=True uses fp16 autocast with loss scaling (CUDA only).

    fp16 is the required mixed-precision mode on Turing-class GPUs (sm_75): bf16 is slow there.
    """
    model.train()
    use_amp = bool(amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    idx = rng.permutation(len(x))
    total, n = 0.0, 0
    steps = 0
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        xb = torch.as_tensor(x[b], device=device)
        rb = torch.as_tensor(rr[b], device=device)
        mb = torch.as_tensor(mask[b], device=device)
        yb = torch.as_tensor(y[b], device=device)
        target = yb[:, -1]
        valid = target >= 0
        if int(valid.sum()) == 0:
            continue
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(xb, rb, mb)[:, -1, :]        # newest beat only, matching evaluation
            loss = F.cross_entropy(logits[valid].float(), target[valid], weight=weights.to(device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(opt)
        scaler.update()
        total += float(loss) * int(valid.sum())
        n += int(valid.sum())
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return total / max(n, 1)


@torch.no_grad()
def predict_probs(model, x, rr, mask, y, batch_size: int, device):
    """Return (probs for the newest beat of each sequence, labels of the newest beat). Causal: newest position only."""
    model.eval()
    probs = []
    for s in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[s:s + batch_size], device=device)
        rb = torch.as_tensor(rr[s:s + batch_size], device=device)
        mb = torch.as_tensor(mask[s:s + batch_size], device=device)
        logits = model(xb, rb, mb)[:, -1, :]
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs), y[:, -1]
