# Environment

## Tested
- Python 3.10
- torch 2.5.1+cu121 (CUDA runtime 12.1), numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, wfdb 4.3.1, PyYAML 6.0.3,
  scikit-learn 1.7.2, pytest 9.2.0
- Tests were run on CPU only. No GPU training has been run from this repository.

## Setup (any machine)
```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install -r environment/requirements.txt  # installs torch CPU or default build; see the note in requirements.txt
```

For a CUDA build of torch, install the matching torch wheel first (see `requirements.txt`), then the rest.

## Not verified
- Turing GPU architecture, driver and CUDA version. The only Turing references found locally describe an RTX 2080 Ti
  (sm_75). fp16 is the mixed-precision choice for that architecture; bf16 is slow there and must not be used.
- Turing scheduler, partition names, storage quotas and job limits. See `docs/TURING_SETUP.md`.
