#!/usr/bin/env bash
# Usage: bash scripts/evaluate_g1.sh plan|check|smoke|run|summarize --config FILE [OPTIONS]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

exec "${EVAL_PYTHON:-python}" -B -m robocasa_eval "$@"
