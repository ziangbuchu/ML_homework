#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEX_FILE="$ROOT_DIR/reports/ML_homework_report.tex"
PDF_FILE="$ROOT_DIR/reports/ML_homework_report.pdf"
LOCAL_TECTONIC="$ROOT_DIR/.tools/tectonic/tectonic"

if [[ -n "${TECTONIC_BIN:-}" ]]; then
  TECTONIC="$TECTONIC_BIN"
elif [[ -x "$LOCAL_TECTONIC" ]]; then
  TECTONIC="$LOCAL_TECTONIC"
elif command -v tectonic >/dev/null 2>&1; then
  TECTONIC="$(command -v tectonic)"
else
  cat >&2 <<'MSG'
No tectonic executable found.

Install it in user space, then rerun this script. On this Linux host the
static musl release is preferred:

  mkdir -p .tools/tectonic
  curl -L https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.16.9/tectonic-0.16.9-x86_64-unknown-linux-musl.tar.gz \
    | tar -xz -C .tools/tectonic tectonic
  chmod +x .tools/tectonic/tectonic

You can also set TECTONIC_BIN=/path/to/tectonic.
MSG
  exit 127
fi

if [[ ! -f "$TEX_FILE" ]]; then
  echo "Missing LaTeX source: $TEX_FILE" >&2
  exit 1
fi

"$TECTONIC" --keep-logs --outdir "$ROOT_DIR/reports" "$TEX_FILE"

if [[ ! -s "$PDF_FILE" ]]; then
  echo "Expected PDF was not created: $PDF_FILE" >&2
  exit 1
fi

if command -v pdfinfo >/dev/null 2>&1; then
  pdfinfo "$PDF_FILE" >/dev/null
fi

echo "Built $PDF_FILE"
