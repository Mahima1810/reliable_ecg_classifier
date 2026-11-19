import numpy as np
import torch

from ecgrc.model import CNNGRU, count_parameters
from ecgrc.windows import beat_windows, causal_rr, causal_sequences


def test_beat_windows_shape_and_edges():
    sig = np.arange(1000, dtype=np.float32)
    w, valid = beat_windows(sig, np.array([5, 500, 995]), pre=40, post=60)
    assert w.shape == (3, 100)
    assert list(valid) == [False, True, False]
    assert np.allclose(w[1], sig[460:560])


def test_causal_rr_uses_only_previous_beats():
    rr = causal_rr(np.array([0, 100, 250, 400]), fs=100.0)
    assert rr[0] == 0.0
    assert np.allclose(rr[1:], [1.0, 1.5, 1.5])
    # changing a later beat must not change earlier RR values
    rr2 = causal_rr(np.array([0, 100, 250, 900]), fs=100.0)
    assert np.allclose(rr[:3], rr2[:3])


def test_causal_sequences_contain_only_past_beats():
    W, n, L = 4, 6, 3
    windows = np.arange(n * W, dtype=np.float32).reshape(n, W)
    rr = np.arange(n, dtype=np.float32)
    labels = np.arange(n) % 4
    x, r, m, y = causal_sequences(windows, rr, labels, L)
    for i in range(n):
        real = m[i].sum()
        assert real == min(i + 1, L)
        # the last position is beat i; earlier real positions are beats i-real+1..i-1, never later beats
        assert np.allclose(x[i, -1], windows[i])
        assert np.allclose(x[i, L - real:], windows[i - real + 1:i + 1])
        assert y[i, -1] == labels[i]


def test_model_output_shape_and_parameter_count():
    model = CNNGRU(window=100, n_classes=4, emb=64, hidden=64, n_layers=2, dropout=0.0)
    B, L, W = 5, 16, 100
    x = torch.randn(B, L, W)
    rr = torch.rand(B, L, 1)
    mask = torch.ones(B, L, dtype=torch.bool)
    out = model(x, rr, mask)
    assert tuple(out.shape) == (B, L, 4)
    assert count_parameters(model) > 0


def test_model_is_causal():
    torch.manual_seed(0)
    model = CNNGRU(window=100, n_classes=4, dropout=0.0).eval()
    B, L, W = 3, 10, 100
    x = torch.randn(B, L, W)
    rr = torch.rand(B, L, 1)
    mask = torch.ones(B, L, dtype=torch.bool)
    with torch.no_grad():
        base = model(x, rr, mask)
        x2 = x.clone()
        x2[:, -1] = torch.randn(B, W) * 10              # change only the newest beat
        pert = model(x2, rr, mask)
    # every position before the newest must be unchanged (no future leakage into earlier outputs)
    assert torch.allclose(base[:, :-1], pert[:, :-1], atol=1e-6)


def test_padding_does_not_affect_real_positions():
    torch.manual_seed(0)
    model = CNNGRU(window=100, n_classes=4, dropout=0.0).eval()
    B, L, W = 2, 8, 100
    x = torch.randn(B, L, W)
    rr = torch.rand(B, L, 1)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, :3] = False                                   # first three positions are padding
    with torch.no_grad():
        a = model(x, rr, mask)
        x2 = x.clone()
        x2[:, :3] = torch.randn(B, 3, W) * 5                 # changing padding must not matter
        b = model(x2, rr, mask)
    assert torch.allclose(a[:, 3:], b[:, 3:], atol=1e-6)


def test_wrong_window_length_is_rejected():
    model = CNNGRU(window=100, n_classes=4)
    try:
        model(torch.randn(1, 4, 99), torch.rand(1, 4, 1), torch.ones(1, 4, dtype=torch.bool))
    except ValueError as e:
        assert "window" in str(e)
    else:
        raise AssertionError("expected ValueError")
