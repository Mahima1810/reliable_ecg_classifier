import pandas as pd
import pytest

from ecgrc.splits import SplitIntegrityError, build_splits, check_split_integrity, holdout_leak_check


def _patients(n_per_ds=40):
    out = {}
    for ds in ["mitdb", "svdb", "incartdb", "edb"]:
        for i in range(n_per_ds):
            out[f"{ds}:{i}"] = [ds]
    return out


def test_every_patient_assigned_once():
    pd_map = _patients()
    split = build_splits(pd_map, seed=1)
    assert set(split) == set(pd_map)
    assert set(split.values()) <= {"train", "val", "test"}


def test_split_is_deterministic_for_seed():
    pd_map = _patients()
    assert build_splits(pd_map, seed=7) == build_splits(pd_map, seed=7)
    assert build_splits(pd_map, seed=7) != build_splits(pd_map, seed=8)


def test_fractions_stratified_by_dataset():
    pd_map = _patients(50)
    split = build_splits(pd_map, seed=3, fractions=(0.6, 0.2, 0.2))
    for ds in ["mitdb", "svdb", "incartdb", "edb"]:
        ids = [p for p in pd_map if p.startswith(ds + ":")]
        counts = {s: sum(1 for p in ids if split[p] == s) for s in ["train", "val", "test"]}
        assert counts == {"train": 30, "val": 10, "test": 10}


def test_holdout_dataset_goes_entirely_to_test():
    pd_map = _patients()
    split = build_splits(pd_map, seed=2, holdout_dataset="svdb")
    held = [p for p in pd_map if p.startswith("svdb:")]
    assert all(split[p] == "test" for p in held)
    beats = pd.DataFrame({"patient_id": list(pd_map), "dataset": [pd_map[p][0] for p in pd_map]})
    holdout_leak_check(split, beats, "svdb")


def test_leak_check_detects_held_out_patient_in_train():
    beats = pd.DataFrame({"patient_id": ["a", "b"], "dataset": ["svdb", "mitdb"]})
    split = {"a": "train", "b": "test"}
    with pytest.raises(SplitIntegrityError):
        holdout_leak_check(split, beats, "svdb")


def test_integrity_fails_for_beat_without_split():
    beats = pd.DataFrame({"patient_id": ["a", "zz"], "dataset": ["mitdb", "mitdb"]})
    with pytest.raises(SplitIntegrityError):
        check_split_integrity({"a": "train"}, beats)


def test_integrity_fails_for_unknown_split_label():
    beats = pd.DataFrame({"patient_id": ["a"], "dataset": ["mitdb"]})
    with pytest.raises(SplitIntegrityError):
        check_split_integrity({"a": "dev"}, beats)


def test_integrity_summary_counts():
    beats = pd.DataFrame({"patient_id": ["a", "b"], "dataset": ["mitdb", "svdb"]})
    s = check_split_integrity({"a": "train", "b": "test"}, beats)
    assert s["patients"] == 2
    assert s["patients_per_split"] == {"train": 1, "val": 0, "test": 1}
