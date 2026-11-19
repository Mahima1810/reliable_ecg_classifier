# Downloading the datasets (step by step, no assistant needed)

Turing has no Claude or other assistant, so every step below must be run by hand in a shell. All four datasets are **open access**
on PhysioNet: no account, no credentials, no data-use agreement. Everything below uses only `bash`, `curl`, `sha256sum` and `awk`.

Do **not** download MIMIC data with these instructions. MIMIC is restricted and is not part of this project.

## 1. What you need
| Item | Amount |
|---|---|
| Free disk on a persistent (non-scratch) volume | about 1.5 GB for the four datasets (measured on 2026-11-19: mitdb about 91 MB, svdb about 54 MB, incartdb about 779 MB, edb about 436 MB) |
| Tools | `bash`, `curl`, `sha256sum`, `awk` (all standard on Linux) |
| Internet access | from the machine where the download runs (a login node, if compute nodes have no internet) |

Choose a location and set it once:
```bash
export ECG_DATA_ROOT=/persistent/path/to/ecg_data    # any directory on persistent storage
mkdir -p "$ECG_DATA_ROOT"
```

## 2. Preview what will be downloaded (no download)
```bash
bash scripts/download_datasets.sh --list
```
Expected root-level file counts, which are the checksum entries the script will fetch:

| Dataset | Records | Expected files | Note |
|---|---|---|---|
| mitdb | 48 | 145 | 48 × (.hea, .dat, .atr), plus `102-0.atr` (a secondary annotation of record 102, not used) |
| svdb | 78 | 234 | 78 × 3 |
| incartdb | 75 | 225 | 75 × 3 |
| edb | 90 | 270 | 90 × 3 |

## 3. Download and verify all four
```bash
bash scripts/download_datasets.sh
```
This creates `$ECG_DATA_ROOT/mitdb`, `svdb`, `incartdb` and `edb`. Each file is checked against the `SHA256SUMS.txt` published on
PhysioNet. The script exits with status 1 if any file is missing or fails its checksum.

Run only one dataset:
```bash
bash scripts/download_datasets.sh --datasets "edb"
```

## 4. Re-check files already on disk (no download)
```bash
bash scripts/download_datasets.sh --verify-only
```
Use this after an interrupted run, after copying the data to another machine, or to confirm integrity before a training run.

## 5. What the script does
- Downloads the checksum list for each dataset from `https://physionet.org/files/<dataset>/1.0.0/SHA256SUMS.txt`.
- Selects root-level `.hea`, `.dat` and `.atr` files only. Sub-folders, such as `mitdb/x_mitdb/` (alternative signals), are not fetched.
- Fetches each missing file with `curl -L`. PhysioNet replies with HTTP 302 redirects, so `-L` is required. Without it,
  files come back empty and the checksums fail.
- Resumes partial downloads (`curl -C -`). Re-running the script fetches only missing files.
- Verifies every fetched or existing file against its published SHA-256.

## 6. Check the result
```bash
for ds in mitdb svdb incartdb edb; do
  echo "$ds: $(ls "$ECG_DATA_ROOT/$ds"/*.hea | wc -l) headers, $(ls "$ECG_DATA_ROOT/$ds"/*.dat | wc -l) signal files, $(ls "$ECG_DATA_ROOT/$ds"/*.atr | wc -l) annotation files"
done
```
Expected: mitdb 48 / 48 / 49 (49 `.atr` because of `102-0.atr`); svdb 78 / 78 / 78; incartdb 75 / 75 / 75; edb 90 / 90 / 90.

Then validate the data with the repository's own loaders:
```bash
export ECG_RUN_ROOT=/persistent/path/to/runs
python scripts/validate_data.py --config configs/experiment_recall_primary.yaml
```

## 7. Common problems
| Symptom | Cause | Fix |
|---|---|---|
| Files of 0 bytes, checksum failures | The `-L` redirect flag was lost, or the request was cut off | Re-run the script. It replaces empty files and resumes partial ones |
| `cannot fetch SHA256SUMS.txt` | No internet from this node | Run from a login node, or copy the files over |
| `No space left on device` | Volume too small | Free space or choose a larger persistent path, then re-run |
| `missing tool: curl` (or sha256sum, awk) | Tool not installed | Install it with the system package manager, or ask the admin |
| `FAILED` after a successful download | Corruption in transfer | Delete the named file and re-run with `--verify-only` first to find it |

## 8. Not downloaded by this script
- MIMIC-IV, MIMIC-IV-ECG, MIMIC-IV-WDB and derived projects (restricted).
- CUDB, VFDB, AFDB, NSTDB (not used by the training configuration; used only for the overlap screen).
- CPSC 2018, Chapman-Shaoxing / ecg-arrhythmia, PTB-XL (their licences and counts are not yet verified).
