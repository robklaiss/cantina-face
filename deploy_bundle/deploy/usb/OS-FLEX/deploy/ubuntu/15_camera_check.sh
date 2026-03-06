#!/usr/bin/env bash
# 15_camera_check.sh — Verifica cámara, permisos y captura de smoke test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

log "=== 15_camera_check.sh ==="
require_root

CAPTURE_FILE="/tmp/cantina_cam_test.jpg"
VIDEO_NODE="${VIDEO_DEVICE:-/dev/video0}"

log "ℹ️  Usando VIDEO_DEVICE=$VIDEO_NODE"

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

if ! ls /dev/video* >/dev/null 2>&1; then
    warn "No se detectaron dispositivos /dev/video* inicialmente."
fi

log "⌛ Esperando hasta ${CAMERA_WAIT_SECONDS}s por ${VIDEO_NODE}..."
if ! wait_for_camera_device; then
    if camera_is_optional; then
        warn "Cámara no disponible pero CAMERA_OPTIONAL=1, se continúa sin pruebas."
        exit 0
    fi
    die "No se detectó ${VIDEO_NODE} tras ${CAMERA_WAIT_SECONDS}s."
fi

log "📷 Dispositivos de video encontrados:"
ls -l /dev/video* || true

if have_cmd v4l2-ctl; then
    log "📋 Salida de v4l2-ctl --list-devices:"
    v4l2-ctl --list-devices || true
else
    warn "v4l2-ctl no está instalado (paquete v4l-utils)."
fi

if [ ! -e "$VIDEO_NODE" ]; then
    if camera_is_optional; then
        warn "${VIDEO_NODE} no existe pero la cámara es opcional."
        exit 0
    fi
    die "No se encontró el dispositivo configurado VIDEO_DEVICE=$VIDEO_NODE"
fi

if ! have_cmd ffmpeg; then
    die "ffmpeg no está instalado. Reinstalá ejecutando 00_bootstrap_system.sh"
fi

if ! id "$SILOE_USER" >/dev/null 2>&1; then
    die "El usuario $SILOE_USER no existe. Ajustá SILOE_USER antes de continuar."
fi

log "🔐 Verificando grupo 'video'..."
if have_cmd getent; then
    log "📋 getent group video:"
    getent group video || warn "El grupo video no existe"
fi
log "👤 Grupos actuales de $SILOE_USER:"
id -nG "$SILOE_USER"

if id -nG "$SILOE_USER" | tr ' ' '\n' | grep -qx video; then
    log "El usuario $SILOE_USER ya pertenece al grupo video."
else
    log "Agregando $SILOE_USER al grupo video..."
    usermod -aG video "$SILOE_USER"
    log "✅ $SILOE_USER agregado al grupo video (se aplicará tras el próximo reinicio)."
fi

if camera_is_optional; then
    log "CAMERA_OPTIONAL=1 → omitiendo captura con ffmpeg"
else
    log "🎞️  Capturando frame desde $VIDEO_NODE con ffmpeg..."
    rm -f "$CAPTURE_FILE"
    if ! timeout 10 ffmpeg -loglevel error -y -f video4linux2 -i "$VIDEO_NODE" -frames:v 1 "$CAPTURE_FILE"; then
        die "ffmpeg no pudo capturar desde $VIDEO_NODE. Revisá la cámara o cambia VIDEO_DEVICE."
    fi

    if [ ! -s "$CAPTURE_FILE" ]; then
        die "El archivo $CAPTURE_FILE no se generó correctamente."
    fi

    log "✅ Captura OK: $CAPTURE_FILE ($(du -h "$CAPTURE_FILE" | cut -f1))"
fi
log "=== 15_camera_check.sh completado ==="
