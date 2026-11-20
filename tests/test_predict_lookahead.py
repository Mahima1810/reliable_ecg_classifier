import numpy as np
import pytest
import torch
import wfdb

from ecgrc.checkpoint import save_checkpoint
from ecgrc.labels import CLASSES
from ecgrc.lookahead import lookahead_report
from ecgrc.model import CNNGRU
from ecgrc.predict import load_signal, predict_record


def test_lookahead_report_counts_and_magnitude():
    # beats at 0, 20, 100 samples; post-window 60 samples at 100 Hz
    r = lookahead_report(np.array([0, 20, 100]), post=60, fs=100.0)
    assert r["beats"] == 2
    assert r["beats_with_lookahead"] == 1           # only the beat at 0 has its successor (20) inside 60 samples
    assert r["fraction"] == pytest.approx(0.5)
    assert r["max_lookahead_ms"] == pytest.approx(400.0)   # (60 - 20) samples at 100 Hz
    assert r["window_post_ms"] == pytest.approx(600.0)


def test_lookahead_report_no_beats_inside_window():
    r = lookahead_report(np.array([0, 100, 200]), post=60, fs=100.0)
    assert r["beats_with_lookahead"] == 0
    assert r["max_lookahead_ms"] == 0.0


def test_predict_record_end_to_end(tmp_path):
    fs_native, n = 360, 360 * 20
    rng = np.random.default_rng(0)
    sig = np.stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n)], axis=1)
    wfdb.wrsamp("rec", fs=fs_native, units=["mV", "mV"], sig_name=["MLII", "V5"], p_signal=sig, fmt=["16", "16"],
                write_dir=str(tmp_path))
    # one beat every 0.5 s at 360 Hz, plus an edge beat at sample 30 (8 samples at 100 Hz: no complete pre-window)
    peaks_native = np.concatenate([[30], np.arange(400, n - 400, 180)])
    model = CNNGRU(window=100, n_classes=len(CLASSES), dropout=0.0)
    ck = tmp_path / "ckpt.pt"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(ck, model, opt, epoch=0, best={})
    signal, lead, native = load_signal(tmp_path / "rec", "mitdb", 100.0)
    assert lead == "MLII" and native == 360
    peaks = (peaks_native * 100.0 / native).round().astype(np.int64)
    df = predict_record(model.eval(), signal, peaks, pre=40, post=60, seq_len=16, fs=100.0,
                        device=torch.device("cpu"))
    assert {"sample", "p_N", "p_S", "p_V", "p_F", "decision", "next_peak_inside_window"} <= set(df.columns)
    probs = df[["p_N", "p_S", "p_V", "p_F"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert set(df["decision"]) <= set(CLASSES)
    # beats 0.5 s apart are 50 samples apart at 100 Hz, inside the 60-sample post-window: every beat but the last is flagged
    flags = df["next_peak_inside_window"].to_numpy()
    assert flags[:-1].all() and not flags[-1]
    # edge beats without a complete window are not in the output
    assert len(df) < len(peaks)


def test_predict_record_rejects_mismatched_rate(tmp_path):
    sig = np.zeros((2500, 1))
    wfdb.wrsamp("bad", fs=250, units=["mV"], sig_name=["MLII"], p_signal=sig, fmt=["16"], write_dir=str(tmp_path))
    with pytest.raises(ValueError, match="native_fs"):
        load_signal(tmp_path / "bad", "mitdb", 100.0)
