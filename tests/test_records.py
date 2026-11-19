from pathlib import Path

import numpy as np
import pytest
import wfdb

from ecgrc.config import dataset_config
from ecgrc.labels import SymbolMap, UnknownSymbolError
from ecgrc.records import DataError, choose_lead, load_record


def _write_record(tmp: Path, name: str, fs: int, samples: int, sig_names, symbols):
    t = np.arange(samples) / fs
    sig = np.stack([np.sin(2 * np.pi * 1.2 * t) * (k + 1) for k in range(len(sig_names))], axis=1)
    wfdb.wrsamp(name, fs=fs, units=["mV"] * len(sig_names), sig_name=sig_names, p_signal=sig, fmt=["16"] * len(sig_names),
                write_dir=str(tmp))
    atr_samples = np.array([s for s, _ in symbols], dtype=int)
    atr_syms = [x for _, x in symbols]
    wfdb.wrann(name, "atr", atr_samples, atr_syms, write_dir=str(tmp))
    return tmp / name


def test_choose_lead_follows_preference_order():
    assert choose_lead(["V5", "MLII"], ["MLII", "II", "V5"]) == 1
    assert choose_lead(["V5", "V1"], ["MLII", "V5"]) == 0


def test_choose_lead_raises_when_none_available():
    with pytest.raises(DataError):
        choose_lead(["V5"], ["MLII"])


def test_load_record_resamples_maps_and_keeps_provenance(tmp_path, monkeypatch):
    fs = 360
    samples = fs * 10
    base = _write_record(tmp_path, "100", fs, samples, ["V5", "MLII"],
                         [(400, "N"), (1100, "V"), (1800, "/"), (2500, "+"), (3200, "A")])
    monkeypatch.setattr("ecgrc.records.patient_of", lambda ds, rec: ("mitdb:p100", "documented"))
    rec = load_record("mitdb", base, 100.0, SymbolMap.load(), dataset_config())
    assert rec.lead == "MLII"                      # preferred over V5 for mitdb
    assert abs(len(rec.signal) - 1000) <= 2        # 10 s at 100 Hz
    assert rec.n_nonbeat == 1                      # '+' is not a beat
    b = rec.beats
    assert list(b["symbol"]) == ["N", "V", "/", "A"]
    assert list(b["aami"]) == ["N", "V", "Q", "S"]
    assert set(b["patient_id"]) == {"mitdb:p100"}
    assert set(b["lead"]) == {"MLII"}
    # sample positions are rescaled from 360 Hz to 100 Hz; the native index is kept
    assert list(b["native_sample"]) == [400, 1100, 1800, 3200]
    assert list(b["sample"]) == [round(s * 100 / 360) for s in [400, 1100, 1800, 3200]]


def test_header_rate_mismatch_is_an_error(tmp_path, monkeypatch):
    base = _write_record(tmp_path, "100", 250, 2500, ["MLII"], [(300, "N")])
    monkeypatch.setattr("ecgrc.records.patient_of", lambda ds, rec: ("mitdb:p100", "documented"))
    with pytest.raises(DataError, match="native_fs"):
        load_record("mitdb", base, 100.0, SymbolMap.load(), dataset_config())


def test_unknown_symbol_in_record_raises(tmp_path, monkeypatch):
    # WFDB converts unrecognised symbols to a comment code when writing, so the annotation is injected directly.
    base = _write_record(tmp_path, "100", 360, 3600, ["MLII"], [(400, "N")])
    monkeypatch.setattr("ecgrc.records.patient_of", lambda ds, rec: ("mitdb:p100", "documented"))

    class FakeAnn:
        sample = np.array([400, 800])
        symbol = ["N", "Z"]

    monkeypatch.setattr("ecgrc.records.wfdb.rdann", lambda *a, **k: FakeAnn())
    with pytest.raises(UnknownSymbolError, match="'Z'"):
        load_record("mitdb", base, 100.0, SymbolMap.load(), dataset_config())


def test_missing_patient_mapping_is_an_error(tmp_path):
    base = _write_record(tmp_path, "99999", 360, 3600, ["MLII"], [(400, "N")])
    with pytest.raises(DataError, match="no patient mapping"):
        load_record("mitdb", base, 100.0, SymbolMap.load(), dataset_config())


def test_provisional_symbol_blocked_in_record(tmp_path, monkeypatch):
    base = _write_record(tmp_path, "100", 360, 3600, ["MLII"], [(400, "B")])
    monkeypatch.setattr("ecgrc.records.patient_of", lambda ds, rec: ("mitdb:p100", "documented"))
    with pytest.raises(UnknownSymbolError, match="provisional"):
        load_record("mitdb", base, 100.0, SymbolMap.load(allow_provisional=False), dataset_config())
