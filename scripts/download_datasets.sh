#!/usr/bin/env bash
# Download the four open beat-labelled datasets from PhysioNet into $ECG_DATA_ROOT and verify every file against
# the SHA256SUMS.txt published with each dataset. No credentials are needed (all four are open-access).
#
# Usage:
#   export ECG_DATA_ROOT=/persistent/path/data          # needs about 1.5 GB free
#   bash scripts/download_datasets.sh                   # download + verify all four
#   bash scripts/download_datasets.sh --list            # show which files would be fetched; no download
#   bash scripts/download_datasets.sh --verify-only     # check files already present; no download
#   bash scripts/download_datasets.sh --datasets "edb"  # one dataset
#
# What is fetched (root-level files only, the ones the loader reads):
#   <record>.hea, <record>.dat, <record>.atr
# Sub-folders such as mitdb/x_mitdb/ (alternative signals) and other annotation types are NOT fetched.
#
# Requirements: bash, curl, sha256sum, awk. PhysioNet answers with HTTP 302 redirects, so curl must follow them (-L).
set -euo pipefail

BASE="https://physionet.org/files"
VERSION="1.0.0"
# dataset name on PhysioNet -> local subdirectory name used by configs/datasets.yaml
declare -A SUBDIR=( [mitdb]=mitdb [svdb]=svdb [incartdb]=incartdb [edb]=edb )
ALL="mitdb svdb incartdb edb"

MODE="download"
DATASETS="$ALL"
while [ $# -gt 0 ]; do
  case "$1" in
    --list) MODE="list" ;;
    --verify-only) MODE="verify" ;;
    --datasets) shift; DATASETS="$1" ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$MODE" != "list" ]; then
  : "${ECG_DATA_ROOT:?set ECG_DATA_ROOT to a persistent directory with about 1.5 GB free}"
  mkdir -p "$ECG_DATA_ROOT"
fi
for tool in curl sha256sum awk; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 3; }
done

fail=0
for ds in $DATASETS; do
  [ -n "${SUBDIR[$ds]:-}" ] || { echo "unknown dataset: $ds" >&2; exit 2; }
  url="$BASE/$ds/$VERSION"
  if [ "$MODE" = "list" ]; then
    dest="(not created)"
  else
    dest="$ECG_DATA_ROOT/${SUBDIR[$ds]}"
    mkdir -p "$dest"
  fi
  echo "== $ds ($url) -> $dest"

  # 1. checksum list from PhysioNet (the reference for every file)
  sums_tmp="$(mktemp)"
  if ! curl -sSL --retry 3 -o "$sums_tmp" "$url/SHA256SUMS.txt"; then
    echo "  cannot fetch SHA256SUMS.txt for $ds" >&2; rm -f "$sums_tmp"; fail=1; continue
  fi
  # keep root-level .hea/.dat/.atr only (no '/' in the name)
  want="$(mktemp)"
  awk '{ f=$2; sub(/^\*/, "", f); if (f !~ /\// && f ~ /\.(hea|dat|atr)$/) print $1, f }' "$sums_tmp" > "$want"
  n=$(wc -l < "$want")
  echo "  root-level files to fetch: $n"
  if [ "$MODE" = "list" ]; then
    awk '{print "  " $2}' "$want" | head -5
    echo "  ..."
    rm -f "$sums_tmp" "$want"; continue
  fi

  # 2. download missing files (resume partial downloads)
  if [ "$MODE" = "download" ]; then
    while read -r _ f; do
      if [ ! -s "$dest/$f" ]; then
        tmp="$dest/$f.part"
        if ! curl -sSL --retry 3 -C - -o "$tmp" "$url/$f"; then
          echo "  download failed: $f" >&2; fail=1; continue
        fi
        mv "$tmp" "$dest/$f"
      fi
    done < "$want"
  fi

  # 3. verify every expected file against the published checksum
  missing=0
  while read -r _ f; do [ -s "$dest/$f" ] || missing=$((missing+1)); done < "$want"
  if [ "$missing" -gt 0 ]; then
    echo "  $missing file(s) missing or empty in $dest" >&2; fail=1
  fi
  check="$(mktemp)"
  (cd "$dest" && awk '{print $1"  "$2}' "$want" | sha256sum -c --quiet --strict 2>&1) > "$check" || true
  bad=$(grep -c -v -E '^OK|: OK$|^$' "$check" || true)
  if [ "$bad" -gt 0 ]; then
    echo "  CHECKSUM FAILURES in $ds:" >&2; grep -v -E ': OK$' "$check" | head -10 >&2; fail=1
  else
    echo "  checksums OK for $n files"
  fi
  rm -f "$sums_tmp" "$want" "$check"
done

if [ "$fail" -ne 0 ]; then
  echo "FAILED: see messages above" >&2
  exit 1
fi
echo "done"
