#!/usr/bin/env bash
# 99_reboot.sh — Reiniciar el equipo para que el servicio arranque al boot
# Desactivar con: NO_REBOOT=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

log "=== 99_reboot.sh ==="
require_root

if [ "${NO_REBOOT:-0}" = "1" ]; then
    log "NO_REBOOT=1 → no se reinicia. Para reiniciar manualmente: sudo reboot"
    exit 0
fi

log ""
log "╔══════════════════════════════════════════════════════════════╗"
log "║  El sistema se reiniciará en 5 segundos.                    ║"
log "║  Tras el reboot, el servicio cantina-face arrancará solo.   ║"
log "║                                                             ║"
log "║  Verificar con:                                             ║"
log "║    systemctl status ${SERVICE_NAME}                      ║"
log "║    journalctl -u ${SERVICE_NAME} -f                      ║"
log "║    curl http://localhost:${CANTINA_PORT}/docs                   ║"
log "╚══════════════════════════════════════════════════════════════╝"
log ""

sleep 5
reboot
