#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
VENVDIR="$REPO_DIR/venv"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python3 no está instalado. Instálalo antes de continuar." >&2
    exit 1
fi

if [ ! -d "$VENVDIR" ]; then
    python3 -m venv "$VENVDIR"
fi

source "$VENVDIR/bin/activate"
pip install --upgrade pip
pip install -r "$REPO_DIR/requirements.txt"

echo "Instalación completada. Usa 'source venv/bin/activate' y luego 'python app.py' para iniciar." 
