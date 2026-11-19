# Reliable ECG classifier (recall-primary)

Recall-primary beat classifier for S and V abnormal beats, trained and evaluated on public beat-labelled datasets
(MIT-BIH, SVDB, INCART, European ST-T). Designed for patient-level evaluation with explicit reporting of false negatives
and false positives per recording-hour.

## Status: NOT ready for training
- Code and CPU tests are in place (44 tests, all passing on the development environment).
- **No training has been run from this repository.** No GPU has been used.
- Blockers before the first training job are listed below. Do not start training until they are resolved and approved.

## Blockers
1. **Turing environment not verified.** Scheduler, partitions, GPU architecture, CUDA driver, storage and quotas are unknown.
   See `docs/TURING_SETUP.md`. The scripts in `turing/` refuse to run until `TURING_VERIFIED=1`.
2. **Clinical recall floors and false-positive budget are not set** (`configs/experiment_recall_primary.yaml`). Threshold selection
   refuses to run while they are null. These are owner and clinician decisions.
3. **Seeds and split seed are not set.** Training refuses to run while `experiment.seeds` or `splits.split_seed` is null.
4. **EDB patient overlap is unresolved.** A signal screen against AFDB, CUDB and VFDB found no matches, which does not prove
   independence. Decide whether EDB is used at all before the first run.
5. **Mixed precision not implemented.** fp16 autocast is required on Turing-class GPUs and is not yet in `scripts/train.py`.
6. **Provisional symbols B, n, r** are blocked by default. Any mapping needs a documented decision.
7. **Not yet run on real data:** `scripts/validate_data.py` (the data loaders), a leave-one-dataset-out run, and the full
   training loop with a GPU. Only the synthetic-data tests have run.

## Layout
```
configs/          datasets, label map (AAMI, explicit provisional and non-beat symbols), patient map, experiment config
docs/             DATASETS.md, PROTOCOL.md, TURING_SETUP.md
environment/      requirements.txt (tested versions), environment notes
scripts/          validate_data.py, train.py, evaluate.py
src/ecgrc/        library: config, labels, records, splits, windows, model, metrics, thresholds, engine, checkpoint, seed
tests/            CPU tests (labels, splits, records, windows and model causality, metrics and thresholds, checkpoints, CLI)
turing/           setup_env.sh, env.sh.example, train.sbatch.template (templates, not verified)
```

## Setup
```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r environment/requirements.txt       # see the torch note inside the file
export ECG_DATA_ROOT=/path/to/data                 # dirs: mitdb, svdb, incartdb, edb (see docs/DATASETS.md)
export ECG_RUN_ROOT=/path/to/runs                  # writable, persistent
```

## Commands
Tests (CPU, no data needed):
```bash
python -m pytest -q tests
```

Label-map check (no data needed):
```bash
PYTHONPATH=src python -c "from ecgrc.labels import check_map_consistency; print(check_map_consistency() or 'OK')"
```

Data validation (reads the dataset folders; writes a report to $ECG_RUN_ROOT/validation):
```bash
python scripts/validate_data.py --config configs/experiment_recall_primary.yaml
```

Training (refuses until seeds and split seed are set; uses CUDA only):
```bash
python scripts/train.py --config configs/experiment_recall_primary.yaml --run-name run01 --seed <approved_seed>
python scripts/train.py --config configs/experiment_recall_primary.yaml --run-name run01 --seed <approved_seed> --resume
```

Evaluation, two phases (thresholds are frozen before the test set is touched):
```bash
python scripts/evaluate.py --config configs/experiment_recall_primary.yaml --run-name run01 --phase validate
python scripts/evaluate.py --config configs/experiment_recall_primary.yaml --run-name run01 --phase test
```

## Principles
- **Recall first.** S and V recall are primary. Aggregate accuracy and F1 are never the selection basis.
- **Never silent.** Unknown annotation symbols, unmapped or provisional labels, sampling-rate mismatches, missing leads and
  missing patient mappings raise errors. Q beats are never mapped to normal.
- **Patient-level.** Each patient is in one split. Leave-one-dataset-out checks fail on leakage.
- **Validation-only selection.** Checkpoints and thresholds use validation data. The test set is read once per run, after
  thresholds are frozen with hashes.
- **Causal.** The model uses only the current and past beats (tested).
- **No restricted data.** MIMIC-derived data and caches are excluded by `.gitignore` and are not part of this repository.
- **Separate from STABILITY2.** This repository does not modify the STABILITY2 pre-registration or its commit.

## Reproducibility record
Each run directory records: `config_used.json`, `records.csv` (dataset, record, lead, native fs, duration, beat counts),
`skipped_records.json`, `split_summary.json`, `split_map.json`, `train_log.csv`, checkpoints, `frozen_thresholds.json` and
`test_report.json`, with config and checkpoint SHA-256 hashes.
