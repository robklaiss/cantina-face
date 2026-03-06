#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "No existe el entorno virtual en $VENV_DIR. Ejecuta ./deploy/install.sh primero." >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

# Allow BIND override for backwards compatibility
if [ -n "${BIND:-}" ]; then
    HOST="$BIND"
fi

UVICORN_ARGS=("--host" "$HOST" "--port" "$PORT")

if [ "${RELOAD:-0}" = "1" ]; then
    UVICORN_ARGS+=("--reload")
fi

export FACE_MAX_EMB_PER_SEC="${FACE_MAX_EMB_PER_SEC:-2}"
export FACE_CACHE_MS="${FACE_CACHE_MS:-500}"
export ORT_INTRA_THREADS="${ORT_INTRA_THREADS:-1}"
export ORT_INTER_THREADS="${ORT_INTER_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null; then
        echo "Aviso: el puerto $PORT ya está en uso. Cambia PORT o detén el proceso anterior." >&2
    fi
fi

echo "Iniciando Cantina Face en http://$HOST:$PORT"

exec uvicorn app:app "${UVICORN_ARGS[@]}"
