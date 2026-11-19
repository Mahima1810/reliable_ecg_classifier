# Evaluation protocol (recall-primary)

Status: **draft for approval.** Clinical recall floors, the false-positive budget, the split seed and the run seeds are
not set. The code refuses to select a threshold or train on a seed while they are null.

## Primary metrics
- **S recall** and **V recall**, reported separately, each with a patient-cluster bootstrap confidence interval
  (resampling patients with replacement, B from config, fixed seed).
- Aggregate accuracy and macro-F1 are reported but are never the basis for selecting a model or threshold.

## Secondary metrics
- Per-class precision, F1, support, false negatives, false positives, and false positives per recording-hour.
- Full confusion matrices (N, S, V, F). Q beats are counted separately and are never scored.
- Per-dataset and per-lead results.

## Splits
- Patient-level: each patient is in exactly one of train, val or test. Stratified by dataset (60/20/20 by default).
- Leave-one-dataset-out option: the held-out dataset goes entirely to test. The leak check fails if any of its patients
  is in train or validation.
- The split is written to the run directory (`split_map.json`, `split_summary.json`) for audit.

## Training and model selection
- Model: causal CNN encoder over the beat window, RR feature, unidirectional GRU over the last 16 beats.
  Only the newest beat's label is used in the loss.
- Class weights from the training split only (capped).
- Checkpoint selection: highest validation statistic = mean(average precision for S, average precision for V).
  This is threshold-free and uses validation data only.
- Early stopping on the same statistic (patience from config).

## Threshold selection (validation only)
- Operating rule: alarm class c when p_c ≥ t_c for c in {S, V}. If several alarms fire, the highest-probability one is
  reported. If none fire, the output is the higher of N and F.
- For each class separately: among validation thresholds with recall ≥ the recall floor (and FP/hour ≤ budget, if set), choose the
  threshold with the highest precision. Ties go to the higher threshold.
- If no threshold is feasible, nothing is selected and the run is reported as infeasible.
- The recall floors are clinical decisions. They must be set by the project owner with clinical review and recorded in the config
  before any threshold is selected. This repository sets no values.

## Test evaluation
- Thresholds are frozen in `frozen_thresholds.json`, with SHA-256 hashes of the config and checkpoint.
- The test phase refuses to run if the thresholds are missing or stale, or if `test_report.json` already exists.
- The test set is evaluated once per run. Results are not used to change thresholds, checkpoints or the protocol.

## Reporting
- Every report states the dataset versions, the lead per record, the split seed, the run seed, the config hash and the
  checkpoint hash.
- Failed or infeasible selections are reported, not replaced.
- The EDB patient overlap is reported as unresolved for any result that uses EDB.

## Not yet decided
- Recall floors and the false-positive budget (owner and clinician).
- Split seed and run seeds.
- Whether EDB is used for training, validation, external testing or not at all, pending the overlap review.
- Provisional symbols (`B`, `n`, `r`): whether any mapping is adopted.
