#!/usr/bin/env bash
set -euo pipefail

# Script para instalación automática de Cloudflare Tunnel (modo no interactivo)
# Usado por el sistema de actualización remota

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$HOME/.cloudflared"
TUNNEL_NAME="${TUNNEL_NAME:-cantina-caja}"
TUNNEL_CONFIG="$CONFIG_DIR/config.yml"

echo "[cloudflare] Verificando instalación de cloudflared..."

# Verificar si ya está instalado
if command -v cloudflared &> /dev/null; then
    echo "[cloudflare] ✅ cloudflared ya está instalado ($(cloudflared --version))"
    exit 0
fi

echo "[cloudflare] Instalando cloudflared..."

# Detectar arquitectura
ARCH=$(uname -m)
case $ARCH in
    x86_64)
        PACKAGE="cloudflared-linux-amd64.deb"
        ;;
    aarch64|arm64)
        PACKAGE="cloudflared-linux-arm64.deb"
        ;;
    armv7l)
        PACKAGE="cloudflared-linux-arm.deb"
        ;;
    *)
        echo "[cloudflare] ⚠️  Arquitectura no soportada: $ARCH. Saltando instalación."
        exit 0
        ;;
esac

# Descargar e instalar
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "[cloudflare] Descargando $PACKAGE..."
if wget -q "https://github.com/cloudflare/cloudflared/releases/latest/download/$PACKAGE"; then
    echo "[cloudflare] Instalando paquete..."
    if sudo dpkg -i "$PACKAGE" 2>/dev/null || sudo apt-get install -f -y; then
        echo "[cloudflare] ✅ cloudflared instalado correctamente"
    else
        echo "[cloudflare] ⚠️  Error al instalar cloudflared"
        cd - > /dev/null
        rm -rf "$TEMP_DIR"
        exit 0
    fi
else
    echo "[cloudflare] ⚠️  Error al descargar cloudflared"
fi

cd - > /dev/null
rm -rf "$TEMP_DIR"

echo "[cloudflare] Instalación completada. Ejecuta 'deploy/setup_cloudflare_tunnel.sh' para configurar el túnel."
