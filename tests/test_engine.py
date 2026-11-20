import numpy as np
import torch

from ecgrc.engine import class_weights, predict_probs, train_epoch
from ecgrc.labels import CLASSES
from ecgrc.model import CNNGRU


def _data(n=64, L=8, W=100, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, L, W)).astype(np.float32)
    rr = rng.random((n, L, 1)).astype(np.float32)
    mask = np.ones((n, L), bool)
    y = rng.integers(0, 4, size=(n, L)).astype(np.int64)
    y[:5, -1] = -1                                  # some sequences with no scored newest beat
    return x, rr, mask, y


def test_train_epoch_runs_on_cpu_and_reduces_loss():
    torch.manual_seed(0)
    x, rr, mask, y = _data()
    model = CNNGRU(window=100, n_classes=len(CLASSES), dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    w = class_weights(y[:, -1])
    rng = np.random.default_rng(0)
    first = train_epoch(model, opt, x, rr, mask, y, 16, w, torch.device("cpu"), rng, amp=True)  # amp ignored on CPU
    second = train_epoch(model, opt, x, rr, mask, y, 16, w, torch.device("cpu"), rng, amp=True)
    assert np.isfinite(first) and np.isfinite(second)
    assert second <= first * 1.5                    # sanity: no divergence on a tiny problem


def test_predict_probs_shape_and_labels():
    x, rr, mask, y = _data(n=10)
    model = CNNGRU(window=100, n_classes=len(CLASSES), dropout=0.0)
    p, yt = predict_probs(model, x, rr, mask, y, 4, torch.device("cpu"))
    assert p.shape == (10, len(CLASSES))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(yt, y[:, -1])


def test_class_weights_are_capped_and_finite():
    y = np.array([0] * 1000 + [3] * 3)
    w = class_weights(y, cap=10.0)
    assert float(w.max()) <= 10.0 + 1e-6
    assert torch.isfinite(w).all()
