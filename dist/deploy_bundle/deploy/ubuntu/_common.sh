#!/usr/bin/env bash
# _common.sh — Variables y funciones compartidas por los scripts de deploy/ubuntu
# Este archivo se "sourcea" desde los demás scripts.

# ─── Parámetros configurables ─────────────────────────────────────────────────
export SILOE_USER="${SILOE_USER:-$(logname 2>/dev/null || echo "${SUDO_USER:-$(whoami)}")}"
export CANTINA_PORT="${CANTINA_PORT:-8000}"
export CANTINA_DIR="${CANTINA_DIR:-/opt/cantina-face}"
export SERVICE_NAME="${SERVICE_NAME:-cantina-face}"
export VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}"
export CAMERA_WAIT_SECONDS="${CAMERA_WAIT_SECONDS:-45}"
export CAMERA_OPTIONAL="${CAMERA_OPTIONAL:-0}"
export MODEL_URL="${MODEL_URL:-}"

# Directorio del repo (donde está este script → deploy/ubuntu → repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
export REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ─── Logging ──────────────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $*" >&2
}

die() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
    exit 1
}

# ─── Validación: no ejecutar en modo Live ─────────────────────────────────────
check_not_live() {
    if [ -d "/cow" ]; then
        die "Estás en modo Live (se detectó /cow). Reiniciá sin el USB e instalá Ubuntu primero."
    fi

    local current_user current_host
    current_user="$(whoami)"
    current_host="$(hostname)"

    if [ "$current_user" = "ubuntu" ] && [ "$current_host" = "ubuntu" ]; then
        die "Estás en modo Live (usuario=ubuntu, hostname=ubuntu). Reiniciá sin el USB e instalá Ubuntu primero."
    fi
}

# ─── Verificar root ──────────────────────────────────────────────────────────
require_root() {
    if [ "$(id -u)" -ne 0 ]; then
        die "Este script debe ejecutarse como root. Usá: sudo $0"
    fi
}

# ─── Utilidades de cámara ──────────────────────────────────────────────────────
camera_is_optional() {
    [[ "${CAMERA_OPTIONAL:-0}" = "1" ]]
}

wait_for_path() {
    local path="$1"
    local timeout="${2:-30}"
    local waited=0

    while [ "$waited" -lt "$timeout" ]; do
        if [ -e "$path" ]; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    return 1
}

wait_for_camera_device() {
    local device="${VIDEO_DEVICE:-/dev/video0}"
    local timeout="${CAMERA_WAIT_SECONDS:-45}"

    if wait_for_path "$device" "$timeout"; then
        return 0
    fi

    return 1
}
