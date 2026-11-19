import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ecgrc.checkpoint import load_checkpoint, save_checkpoint
from ecgrc.model import CNNGRU
from ecgrc.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_roundtrip_and_no_temp_file(tmp_path):
    set_seed(5)
    m = CNNGRU(window=100, n_classes=4, dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    p = tmp_path / "ckpt_last.pt"
    save_checkpoint(p, m, opt, epoch=3, best={"epoch": 2, "statistic": 0.5})
    assert p.exists()
    assert not (tmp_path / "ckpt_last.pt.tmp").exists()
    m2 = CNNGRU(window=100, n_classes=4, dropout=0.0)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    payload = load_checkpoint(p, m2, opt2)
    assert payload["epoch"] == 3
    assert payload["best"]["epoch"] == 2
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_resume_restores_random_state(tmp_path):
    set_seed(11)
    m = CNNGRU(window=100, n_classes=4, dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    p = tmp_path / "c.pt"
    save_checkpoint(p, m, opt, epoch=0, best={})
    expected = np.random.random()
    expected_t = torch.rand(1).item()
    # scramble the global RNGs, then restore from the checkpoint
    np.random.random(); torch.rand(10)
    load_checkpoint(p, m, opt)
    assert np.random.random() == expected
    assert torch.rand(1).item() == expected_t


def _env(tmp_path):
    env = dict(os.environ)
    env["ECG_DATA_ROOT"] = str(tmp_path)
    env["ECG_RUN_ROOT"] = str(tmp_path / "runs")
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_train_refuses_while_seeds_and_split_seed_are_null(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config",
                        str(ROOT / "configs" / "experiment_recall_primary.yaml"), "--run-name", "x", "--seed", "1"],
                       env=_env(tmp_path), capture_output=True, text=True)
    assert r.returncode != 0
    assert "seeds is null" in (r.stderr + r.stdout)


def test_evaluate_refuses_without_checkpoint(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--config",
                        str(ROOT / "configs" / "experiment_recall_primary.yaml"), "--run-name", "none",
                        "--phase", "validate", "--device", "cpu"],
                       env=_env(tmp_path), capture_output=True, text=True)
    assert r.returncode != 0
    assert "missing checkpoint" in (r.stderr + r.stdout)


def test_threshold_selection_refuses_while_floors_are_null():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "experiment_recall_primary.yaml").read_text())
    assert cfg["threshold_selection"]["S_recall_floor"] is None
    assert cfg["threshold_selection"]["V_recall_floor"] is None
    assert cfg["threshold_selection"]["fp_per_hour_budget"] is None


def test_validate_data_help_runs(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_data.py"), "--help"],
                       env=_env(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0
    assert "--config" in r.stdout
