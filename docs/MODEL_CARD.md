# Model card: recall-primary CNN-GRU beat classifier

Status: **research code, not trained in this repository, not validated, not a medical device.**

## Intended use
- Research into per-beat classification of N, S, V and F on annotated, public beat-labelled ECG recordings.
- Future use as a screening aid (S and V recall-focused) on a single-lead wearable recording, **only after** the
  validation steps in `docs/PROTOCOL.md` and clinical review have been completed. Not for diagnosis or treatment decisions.

## Not intended for
- Diagnosis, treatment decisions, or unsupervised alarms.
- Q beats (paced or unclassifiable): they are excluded from training, scoring and context. The model never predicts Q.
- Any lead, device, sampling rate or population not represented in the training datasets, until it has been validated.
- Real-time use: no QRS detector or signal filtering is included, and beat windows contain 0.6 s of lookahead.

## Inputs
- One lead per record, chosen by a fixed preference order per dataset (`configs/datasets.yaml`). The lead is recorded.
- Resampled (anti-aliased) to 100 Hz.
- A beat window of 0.4 s before and 0.6 s after an R-peak position (annotated or externally supplied).
- The RR interval to the previous beat, capped at 3 s.

## Architecture
- Per-beat CNN encoder (3 convolution blocks, global pooling), concatenated with the RR feature.
- Unidirectional 2-layer GRU over the last 16 beats (causal). Output: logits for N, S, V, F.
- Parameters: 82,788 for the configured model (computed with `count_parameters`; `scripts/train.py` prints it at start).

## Training data (planned, not yet run here)
- MIT-BIH Arrhythmia, SVDB, INCART (beat-labelled, patient-level splits).
- European ST-T: planned as external test only, because patient identity is inferred and overlap was not fully resolved.
- Class weights from the training split (inverse frequency, capped). Only scored beats are used.

## Evaluation (as designed, not yet run)
- Primary: S recall and V recall, with patient-cluster bootstrap intervals.
- Secondary: precision, F1, confusion matrices, false negatives, false positives per recording-hour.
- Thresholds chosen on validation data only, with recall floors set by the owner and a clinician. Frozen with hashes; the
  test set is evaluated once.

## Known limitations
1. **Labels.** AAMI EC57 mapping is lossy for S and V. In SVDB records 841, 866 and 868, S, V and N are not separable by
   morphology in either lead. F has very few examples. The provisional symbols B, n and r are blocked.
2. **Patient identity.** SVDB and European ST-T identities are largely inferred; patient-level separation depends on that.
3. **Lookahead.** The beat window contains up to 0.6 s of the following signal. Measure the share of beats affected
   with `ecgrc.lookahead.lookahead_report`.
4. **Annotated beats only.** Evaluation uses reference R-peaks. Missed or spurious detections in a real device are not
   counted.
5. **Single lead and dataset-confounded leads.** Lead identity correlates with the source dataset. Cross-lead and
   cross-device performance is not measured.
6. **Calibration.** Probabilities are not validated per class. The output should not be read as calibrated risk.
7. **Small test strata.** Some subgroups have few patients, so confidence intervals will be wide.
8. **Nondeterminism.** GPU training with fp16 is not bit-reproducible; a re-run must be checked against the reported ranges.

## Safety conditions (to be implemented before any use beyond research)
- Abstention or "uncertain" output for poor signal, unsupported leads, and low-confidence beats.
- Signal-quality gating and QRS-detector disagreement checks.
- Recall and false-positive budgets set by the owner and a clinician, validated on held-out data.
- Prospective evaluation under an ethics approval.

## Data protections
- Restricted data (MIMIC-derived material) is never included in this repository or its history.
- Patient-level identifiers in the repository are dataset record names and pseudonymous keys only.

## Contact and version
- Repository: this repository. Version: see git history. Changes to this card must accompany any change to the model, data or protocol.
