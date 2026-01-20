#!/bin/bash

set -euo pipefail

if [ ! -d "venv" ]; then
    echo "❌ No se encontró el entorno virtual. Ejecuta ./setup.sh primero." >&2
    exit 1
fi

echo "🔧 Activando entorno virtual..."
# shellcheck disable=SC1091
source venv/bin/activate

echo "🚀 Iniciando Cantina Face en http://localhost:8000/static/index.html"
echo "🛑 Presiona Ctrl+C para detener el servidor"

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
