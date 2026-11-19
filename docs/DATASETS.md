# Datasets

Beat-labelled research datasets used by this repository. Counts are from the local audit on 2026-11-17 to 2026-11-19 and
should be re-checked by `scripts/validate_data.py` on each machine. Licences were checked from PhysioNet or AWS listings,
not from each file's licence text.

| Dataset | Records | Native fs | Leads | Beat labels (N / S / V / F / Q) | Patients | Licence | Status |
|---|---|---|---|---|---|---|---|
| MIT-BIH Arrhythmia (`mitdb`) | 48; 4 paced and 202 excluded (see config), 43 used in V2 splits | 360 Hz | 2 (MLII preferred) | counted from annotation file: N 90,631; S 2,781; V 7,236; F 803; Q 8,043 (all records, before exclusions) | Documented; 201 and 202 are the same patient | ODC-By 1.0 | Used in V2 tests (exploratory) |
| SVDB (`svdb`) | 78 | 128 Hz | 2 (ECG1 preferred) | N 162,340; S 12,198; V 9,943; F 23; Q 79 | 63 identities assumed; 15 verified as duplicates of AFDB tapes | ODC-By 1.0 | Used in V2 test (audit); overlap with AFDB |
| INCART (`incartdb`) | 75 | 257 Hz | 12 (II preferred) | N 153,709; V 20,013; S 1,960; F 219; Q 6 | 32 patients (header-based) | ODC-By 1.0 | Used in V2 test (exploratory) |
| European ST-T (`edb`) | 90 | 250 Hz | 2 (lead varies by record) | N 784,633 (all beat symbols); V 4,467 in 50 records; S 1,093 in 64; F 354 in 14; Q 11 | Record-level only; age and sex in headers | ODC-By 1.0 (PhysioNet listing) | **Patient overlap with other datasets unresolved** |

## Known limitations
- **Labels.** AAMI EC57 mapping is lossy for S and V. Records 841, 866 and 868 (SVDB) have S, V and N that cannot be separated
  by morphology in either lead. F has very few beats in every source. Q is not scored.
- **Provisional symbols.** `B`, `n` and `r` appear in the V2 label map but not in AAMI EC57. They raise an error unless
  `allow_provisional_labels` is enabled in the experiment config.
- **Patient identity.** SVDB and EDB identities are largely assumed (one record = one patient). Patient-independent splits
  depend on that assumption until it is verified.
- **Overlap.** SVDB and AFDB share 15 records. EDB overlap with MIT-BIH, SVDB, INCART and AFDB has not been resolved:
  an exact-digital-sample screen against AFDB, CUDB and VFDB found no matches, which does not establish independence.
- **Leads.** Lead choice is per record. Reports are stratified by lead; a single-lead model may transfer differently between
  leads.
- **Sampling rate.** Each record is checked against the native rate in `configs/datasets.yaml`. Mismatches stop loading.
  Resampling is anti-aliased polyphase (`scipy.signal.resample_poly`) to the processing rate.
- **Recording duration.** The four datasets differ in duration (30 min for INCART records, 2 h for EDB). FP per hour is
  computed from the recorded duration of the scored records.

## Excluded and restricted sources
- **MIMIC-IV, MIMIC-IV-ECG, MIMIC-IV-WDB and derived projects are not used here.** They are credentialed or restricted and must
  never be copied into this repository. See the access notes in the project documentation kept outside Git.
- No dataset files, caches or checkpoints are stored in Git (see `.gitignore`).

## Patient map
`configs/patient_map.csv` lists (dataset, record, patient_id, evidence) for every record used. It contains dataset record names
and pseudonymous patient keys derived from the existing V2 split manifest. It contains no MIMIC data.
