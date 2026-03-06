#!/usr/bin/env bash
# 20_models.sh — Resolver modelo: mobile_face.onnx -> arcface_r50.onnx (symlink)
# Idempotente: siempre recrea el symlink y puede re-hidratar el modelo desde el USB/ZIP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

log "=== 20_models.sh ==="
require_root

MODELS_DIR="$CANTINA_DIR/models"
ARCFACE_MODEL="$MODELS_DIR/arcface_r50.onnx"
MOBILE_FACE="$MODELS_DIR/mobile_face.onnx"
DEFAULT_USB_ROOT="/media/${SILOE_USER}/OS-FLEX"
USB_ROOT="${USB_ROOT:-$DEFAULT_USB_ROOT}"
DEFAULT_USB_MODEL_SOURCE="${USB_ROOT}/models/arcface_r50.onnx"
USB_MODEL_SOURCE="${USB_MODEL_SOURCE:-$DEFAULT_USB_MODEL_SOURCE}"
USB_PROJECT_ZIP="${USB_ROOT}/deploy/project.zip"

mkdir -p "$MODELS_DIR"

extract_model_from_zip() {
    local zip_path="$1"
    local destination="$2"
    [[ -f "$zip_path" ]] || return 1
    if ! command -v unzip >/dev/null 2>&1; then
        warn "unzip no está disponible; no puedo extraer arcface_r50.onnx desde $zip_path"
        return 1
    fi

    local member
    member="$(unzip -Z1 "$zip_path" | grep -E '(^|/)arcface_r50\.onnx$' | head -n 1 || true)"
    if [[ -z "$member" ]]; then
        return 1
    fi

    log "Extrayendo arcface_r50.onnx desde ${zip_path} (${member})"
    unzip -p "$zip_path" "$member" > "$destination"
    chmod 644 "$destination"
    return 0
}

ensure_model_present() {
    if [[ -s "$ARCFACE_MODEL" ]]; then
        log "Modelo arcface_r50.onnx ya existe en destino"
        return
    fi

    if [[ -n "$USB_MODEL_SOURCE" && -s "$USB_MODEL_SOURCE" ]]; then
        log "Copiando arcface_r50.onnx desde $USB_MODEL_SOURCE"
        install -m 644 "$USB_MODEL_SOURCE" "$ARCFACE_MODEL"
        return
    fi

    if [[ -n "$USB_ROOT" && -s "${USB_ROOT}/project/models/arcface_r50.onnx" ]]; then
        log "Copiando modelo desde ${USB_ROOT}/project/models/arcface_r50.onnx"
        install -m 644 "${USB_ROOT}/project/models/arcface_r50.onnx" "$ARCFACE_MODEL"
        return
    fi

    if extract_model_from_zip "$USB_PROJECT_ZIP" "$ARCFACE_MODEL"; then
        return
    fi

    die "No se pudo provisionar arcface_r50.onnx. Copiá el modelo a ${USB_ROOT}/models/arcface_r50.onnx y reintentá."
}

ensure_model_present

# ─── Crear symlink mobile_face.onnx -> arcface_r50.onnx (siempre) ────────────
log "Creando symlink: mobile_face.onnx -> arcface_r50.onnx"
rm -f "$MOBILE_FACE"
ln -s arcface_r50.onnx "$MOBILE_FACE"

# Asegurar permisos
chown -h "$SILOE_USER":"$(id -gn "$SILOE_USER")" "$MOBILE_FACE"
chown "$SILOE_USER":"$(id -gn "$SILOE_USER")" "$ARCFACE_MODEL"

log "Modelo listo: $(ls -la "$MOBILE_FACE")"
log "=== 20_models.sh completado ==="
