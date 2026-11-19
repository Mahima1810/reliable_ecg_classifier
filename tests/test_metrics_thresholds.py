import numpy as np
import pytest

from ecgrc.labels import CLASSES
from ecgrc.metrics import confusion, patient_bootstrap_ci, per_class, precision_stat, recall_stat
from ecgrc.thresholds import ThresholdError, apply_rule, curve, select_threshold

C = {c: i for i, c in enumerate(CLASSES)}


def test_confusion_and_per_class_on_toy_example():
    y = np.array([C["V"], C["V"], C["V"], C["N"], C["N"], C["S"]])
    p = np.array([C["V"], C["N"], C["V"], C["N"], C["V"], C["S"]])
    cm = confusion(y, p)
    stats = per_class(cm, hours=2.0)
    v = stats["V"]
    assert v["support"] == 3 and v["false_negatives"] == 1
    assert v["recall"] == pytest.approx(2 / 3)
    assert v["precision"] == pytest.approx(2 / 3)       # 2 TP, 1 FP (an N predicted as V)
    assert v["fp_per_hour"] == pytest.approx(0.5)
    assert stats["S"]["recall"] == 1.0


def test_patient_bootstrap_is_reproducible_and_brackets_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    p = np.where(rng.random(400) < 0.8, y, rng.integers(0, 4, 400))
    pid = np.repeat(np.arange(20), 20).astype(str)
    a = patient_bootstrap_ci(y, p, pid, recall_stat("V"), B=300, seed=1)
    b = patient_bootstrap_ci(y, p, pid, recall_stat("V"), B=300, seed=1)
    assert a == b
    assert a["lo"] <= a["point"] <= a["hi"]
    assert a["n_patients"] == 20


def test_recall_and_precision_stats_handle_empty_class():
    cm = np.zeros((4, 4), np.int64)
    assert np.isnan(recall_stat("S")(cm))
    assert np.isnan(precision_stat("S")(cm))


def test_threshold_requires_a_recall_floor():
    scores = np.linspace(0, 1, 50)
    is_v = scores > 0.6
    with pytest.raises(ThresholdError, match="recall_floor"):
        select_threshold(scores, is_v, hours=1.0, recall_floor=None)


def test_threshold_maximises_precision_above_recall_floor():
    scores = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2])
    is_v = np.array([1, 1, 0, 1, 0, 0], bool)
    sel = select_threshold(scores, is_v, hours=1.0, recall_floor=0.6)
    assert sel["feasible"]
    assert sel["validation"]["recall"] >= 0.6
    # any other feasible threshold has precision no higher than the chosen one
    for r in curve(scores, is_v, 1.0):
        if r["recall"] >= 0.6:
            assert r["precision"] <= sel["validation"]["precision"] + 1e-12


def test_infeasible_floor_selects_nothing():
    scores = np.array([0.1, 0.2, 0.3])
    is_v = np.array([0, 0, 0], bool)                  # no positives: recall undefined everywhere
    sel = select_threshold(scores, is_v, hours=1.0, recall_floor=0.9)
    assert not sel["feasible"]
    assert sel["threshold"] is None


def test_fp_budget_can_make_selection_infeasible():
    scores = np.array([0.9, 0.85, 0.8, 0.79, 0.78])
    is_v = np.array([0, 0, 0, 0, 1], bool)            # the only positive is ranked last: full recall needs 4 FPs
    sel = select_threshold(scores, is_v, hours=1.0, recall_floor=1.0, fp_per_hour_budget=0.0)
    assert not sel["feasible"]
    sel2 = select_threshold(scores, is_v, hours=1.0, recall_floor=1.0, fp_per_hour_budget=10.0)
    assert sel2["feasible"]


def test_operating_rule_triggers_and_fallback():
    probs = np.array([
        [0.6, 0.1, 0.1, 0.2],     # N likely, no trigger
        [0.1, 0.1, 0.7, 0.1],     # V triggers
        [0.1, 0.8, 0.05, 0.05],   # S triggers
        [0.05, 0.05, 0.05, 0.85], # F, no trigger
    ])
    thr = {C["S"]: 0.5, C["V"]: 0.5}
    pred = apply_rule(probs, thr, C)
    assert list(pred) == [C["N"], C["V"], C["S"], C["F"]]
