#!/usr/bin/env bash
# Create the Python environment for this repository. Uses only paths given in turing/env.sh (not committed).
# Status: template. Run on a login node after filling in turing/env.sh and checking GPU/CUDA with docs/TURING_SETUP.md.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -f "$REPO_DIR/turing/env.sh" ]; then
  echo "turing/env.sh is missing. Copy turing/env.sh.example to turing/env.sh and fill it in." >&2
  exit 2
fi
# shellcheck disable=SC1091
. "$REPO_DIR/turing/env.sh"

: "${TURING_VERIFIED:?set TURING_VERIFIED=1 in env.sh only after the Turing details in docs/TURING_SETUP.md are confirmed}"
: "${TURING_PYTHON:?set the Python 3.10 interpreter path in env.sh}"
: "${ECG_VENV:?set the venv location in env.sh (persistent storage)}"

"$TURING_PYTHON" -m venv "$ECG_VENV"
# shellcheck disable=SC1091
. "$ECG_VENV/bin/activate"
python -m pip install --upgrade pip
# torch: install the wheel that matches the cluster's CUDA driver first (see environment/requirements.txt)
if [ -n "${TORCH_SPEC:-}" ]; then
  python -m pip install ${TORCH_SPEC}
fi
python -m pip install -r "$REPO_DIR/environment/requirements.txt"
python -c "import torch, numpy, wfdb, sklearn; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'cuda available', torch.cuda.is_available())"
echo "environment ready: $ECG_VENV"
