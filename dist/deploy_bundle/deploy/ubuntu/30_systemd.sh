#!/usr/bin/env bash
# 30_systemd.sh — Crear/activar servicio systemd para cantina-face
# Debe ejecutarse como root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

log "=== 30_systemd.sh ==="
require_root

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
DEFAULTS_FILE="/etc/default/cantina-face"
SILOE_GROUP="$(id -gn "$SILOE_USER")"

# ─── Crear /etc/default/cantina-face (EnvironmentFile) ───────────────────────
log "Escribiendo $DEFAULTS_FILE..."
cat > "$DEFAULTS_FILE" <<EOF
# Configuración de entorno para cantina-face (generado por 30_systemd.sh)
# Editá este archivo para cambiar puerto, timezone, etc.
CANTINA_PORT=${CANTINA_PORT}
CANTINA_DIR=${CANTINA_DIR}
SILOE_USER=${SILOE_USER}
VIDEO_DEVICE=${VIDEO_DEVICE}
CAMERA_WAIT_SECONDS=${CAMERA_WAIT_SECONDS}
CAMERA_OPTIONAL=${CAMERA_OPTIONAL}
MODEL_URL=${MODEL_URL}
LOCAL_TIMEZONE=America/Asuncion
# SECRET_KEY=cambia-esta-clave-super-secreta
# ADMIN_EMAIL=admin@siloe.com.py
# ADMIN_PASSWORD=admin321
# LOG_LEVEL=INFO
EOF
chmod 644 "$DEFAULTS_FILE"

# ─── Generar unit file con variables resueltas ────────────────────────────────
log "Generando $SERVICE_FILE..."
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Cantina Face Recognition System (FastAPI + Uvicorn)
After=network-online.target gdm.service
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=3

[Service]
Type=simple
User=${SILOE_USER}
Group=${SILOE_GROUP}
WorkingDirectory=${CANTINA_DIR}
EnvironmentFile=-/etc/default/cantina-face
Environment=PYTHONUNBUFFERED=1
Environment=ORT_INTRA_THREADS=1
Environment=ORT_INTER_THREADS=1
Environment=OMP_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=NUMEXPR_NUM_THREADS=1

ExecStartPre=/usr/bin/bash ${CANTINA_DIR}/deploy/ubuntu/preflight.sh
ExecStart=${CANTINA_DIR}/.venv/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port ${CANTINA_PORT}

Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cantina-face

[Install]
WantedBy=graphical.target
EOF

chmod 644 "$SERVICE_FILE"

# ─── Recargar, habilitar e iniciar ──────────────────────────────────────────
log "Recargando systemd y habilitando servicio..."
systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}.service"

# Esperar un momento y verificar
sleep 3
if systemctl is-active --quiet "${SERVICE_NAME}.service" 2>/dev/null; then
    log "✅ Servicio ${SERVICE_NAME} activo y corriendo"
else
    warn "El servicio no arrancó correctamente. Revisá con: sudo journalctl -u ${SERVICE_NAME} -n 200 --no-pager"
fi

log "=== 30_systemd.sh completado ==="
