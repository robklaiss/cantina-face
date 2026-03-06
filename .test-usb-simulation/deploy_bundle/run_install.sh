#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_install.sh - Runner para instalación inicial desde USB (noexec-safe)
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
echo "Cantina Face - Instalación inicial desde USB"
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

if [ ! -f "$SCRIPT_DIR/deploy/install.sh" ]; then
    echo "ERROR: No se encontró deploy/install.sh en $SCRIPT_DIR" >&2
    exit 1
fi

# Copiar a disco local (requiere sudo)
echo "[1/4] Copiando bundle a disco local..."
if [ "$EUID" -ne 0 ]; then
    echo "Se requiere sudo para copiar a $LOCAL_DIR"
    sudo mkdir -p "$LOCAL_DIR"
    sudo rsync -a --delete "$SCRIPT_DIR/" "$LOCAL_DIR/"
    sudo chown -R "$USER:$(id -gn)" "$LOCAL_DIR"
else
    mkdir -p "$LOCAL_DIR"
    rsync -a --delete "$SCRIPT_DIR/" "$LOCAL_DIR/"
fi

echo "[2/4] Descomprimiendo project.zip..."
cd "$LOCAL_DIR"
unzip -q "$LOCAL_DIR/project.zip" -d "$LOCAL_DIR"

# Limpiar basura de macOS del descomprimido
find "$LOCAL_DIR" -name '__MACOSX' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$LOCAL_DIR" -name '.DS_Store' -delete 2>/dev/null || true
find "$LOCAL_DIR" -name '._*' -delete 2>/dev/null || true

echo "[3/4] Ejecutando install.sh desde disco local..."

# Ejecutar install.sh con TARGET_APP_DIR
sudo env TARGET_APP_DIR="$TARGET_APP_DIR" bash "$LOCAL_DIR/deploy/install.sh"

echo ""
echo "[4/4] Instalación completada"
echo ""
echo "El sistema está instalado en: $TARGET_APP_DIR"
echo "Puedes desconectar el USB de forma segura."
echo ""
echo "Para iniciar el servidor: bash $TARGET_APP_DIR/deploy/run.sh"
