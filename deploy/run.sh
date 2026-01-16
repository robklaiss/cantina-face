#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
source "$REPO_DIR/venv/bin/activate"
python app.py
