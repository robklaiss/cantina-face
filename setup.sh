#!/bin/bash

set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 no está instalado. Instálalo antes de continuar." >&2
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
else
    echo "El entorno virtual ya existe."
fi

echo "Activando entorno virtual..."
# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Listo ✅"
echo "Ejecuta ./run.sh para iniciar el servidor."
