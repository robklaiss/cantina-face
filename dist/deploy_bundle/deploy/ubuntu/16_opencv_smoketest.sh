#!/usr/bin/env bash
# 16_opencv_smoketest.sh — Ejecuta smoketest_face.py dentro de la venv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

log "=== 16_opencv_smoketest.sh ==="
require_root

VENV_DIR="$CANTINA_DIR/.venv"
SMOKE_SCRIPT="$CANTINA_DIR/deploy/ubuntu/smoketest_face.py"
VIDEO_NODE="${VIDEO_DEVICE:-/dev/video0}"

if [ ! -d "$VENV_DIR" ]; then
    die "No existe la venv en $VENV_DIR. Ejecutá 10_install_app.sh primero."
fi

if [ ! -f "$SMOKE_SCRIPT" ]; then
    die "No se encuentra $SMOKE_SCRIPT (asegurate de copiar deploy/ubuntu al destino)."
fi

if ! wait_for_camera_device; then
    if camera_is_optional; then
        warn "Cámara no disponible, CAMERA_OPTIONAL=1 → omitiendo smoketest."
        exit 0
    fi
    die "No se detectó ${VIDEO_NODE} tras ${CAMERA_WAIT_SECONDS}s."
fi

SMOKE_SECONDS="${SMOKE_SECONDS:-3}"

log "📸 Ejecutando smoketest_face.py (device=${VIDEO_DEVICE}, seconds=${SMOKE_SECONDS})..."
sudo -u "$SILOE_USER" bash -c "
    source '$VENV_DIR/bin/activate'
    python3 '$SMOKE_SCRIPT' --device '${VIDEO_DEVICE}' --seconds '${SMOKE_SECONDS}'
"

log "=== 16_opencv_smoketest.sh completado ==="
