#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_update.sh - Runner para actualización desde USB (noexec-safe)
# ============================================================================
# Copia TODO el bundle a disco local antes de ejecutar para evitar:
# - Problemas con USB montado noexec
# - Problemas con symlinks en exFAT/FAT
# - Problemas de permisos
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="/opt/cantina-face-deploy"
TARGET_APP_DIR="${TARGET_APP_DIR:-/opt/cantina-face}"

echo "============================================"
echo "Cantina Face - Actualización desde USB"
echo "============================================"
echo ""
echo "Origen:         $SCRIPT_DIR"
echo "Copia local:    $LOCAL_DIR"
echo "App destino:    $TARGET_APP_DIR"
echo ""

# Verificar que estamos en el bundle correcto
if [ ! -f "$SCRIPT_DIR/project.zip" ]; then
    echo "ERROR: No se encontró project.zip en $SCRIPT_DIR" >&2
    echo "Verifica que estés ejecutando desde deploy_bundle/" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/deploy/update.sh" ]; then
    echo "ERROR: No se encontró deploy/update.sh en $SCRIPT_DIR" >&2
    exit 1
fi

# Copiar a disco local (requiere sudo)
echo "[1/3] Copiando bundle a disco local..."
if [ "$EUID" -ne 0 ]; then
    echo "Se requiere sudo para copiar a $LOCAL_DIR"
    sudo mkdir -p "$LOCAL_DIR"
    sudo rsync -a --delete "$SCRIPT_DIR/" "$LOCAL_DIR/"
    sudo chown -R "$USER:$(id -gn)" "$LOCAL_DIR"
else
    mkdir -p "$LOCAL_DIR"
    rsync -a --delete "$SCRIPT_DIR/" "$LOCAL_DIR/"
fi

echo "[2/3] Ejecutando update.sh desde disco local..."
cd "$LOCAL_DIR"

# Ejecutar update.sh con TARGET_APP_DIR
sudo env TARGET_APP_DIR="$TARGET_APP_DIR" bash "$LOCAL_DIR/deploy/update.sh" "$LOCAL_DIR/project.zip"

echo ""
echo "[3/3] Actualización completada"
echo ""
echo "El bundle local está en: $LOCAL_DIR"
echo "La app está en: $TARGET_APP_DIR"
echo "Puedes desconectar el USB de forma segura."
