#!/usr/bin/env bash
set -euo pipefail

# Script para subir deploy bundle por SSH a la máquina caja
# Uso: ./tools/deploy_ssh.sh [usuario@host]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUNDLE_DIR="$ROOT_DIR/dist/deploy_bundle"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Uso: $0 [usuario@host]

Sube el deploy bundle a la máquina caja por SSH y ejecuta la actualización.

Ejemplos:
  $0 cantina@192.168.1.100
  $0 ubuntu@caja.local
  $0 user@10.0.0.50

Variables de entorno opcionales:
  SSH_PORT=22              Puerto SSH (default: 22)
  REMOTE_DIR=/tmp/deploy   Directorio temporal remoto (default: /tmp/cantina-deploy)
  AUTO_UPDATE=1            Ejecutar actualización automáticamente (default: 0)
  SSH_KEY=/path/to/key     Usar clave SSH específica

EOF
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

SSH_TARGET="$1"
SSH_PORT="${SSH_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/cantina-deploy}"
AUTO_UPDATE="${AUTO_UPDATE:-0}"
SSH_KEY="${SSH_KEY:-}"

# Construir comando SSH
SSH_CMD="ssh -p $SSH_PORT"
if [ -n "$SSH_KEY" ]; then
    SSH_CMD="$SSH_CMD -i $SSH_KEY"
fi

echo -e "${GREEN}=== Deploy Bundle por SSH ===${NC}"
echo "Target: $SSH_TARGET"
echo "Puerto: $SSH_PORT"
echo "Directorio remoto: $REMOTE_DIR"
echo ""

# Verificar que el bundle existe
if [ ! -d "$BUNDLE_DIR" ]; then
    echo -e "${RED}Error: Bundle no encontrado en $BUNDLE_DIR${NC}"
    echo "Ejecuta primero: make deploy-bundle"
    exit 1
fi

# Verificar archivos críticos
CRITICAL_FILES=(
    "run_update.sh"
    "run_install.sh"
    "project.zip"
    "deploy/update.sh"
    "deploy/install.sh"
    "models/arcface_r50.onnx"
)

echo -e "${YELLOW}[1/5] Verificando bundle...${NC}"
for file in "${CRITICAL_FILES[@]}"; do
    if [ ! -e "$BUNDLE_DIR/$file" ]; then
        echo -e "${RED}Error: Archivo crítico faltante: $file${NC}"
        exit 1
    fi
done
echo "✅ Bundle válido"

# Probar conexión SSH
echo -e "${YELLOW}[2/5] Probando conexión SSH...${NC}"
if ! $SSH_CMD "$SSH_TARGET" "echo 'Conexión exitosa'" > /dev/null 2>&1; then
    echo -e "${RED}Error: No se puede conectar a $SSH_TARGET${NC}"
    echo "Verifica:"
    echo "  - Usuario y host correctos"
    echo "  - Puerto SSH ($SSH_PORT)"
    echo "  - Clave SSH (si es necesaria)"
    exit 1
fi
echo "✅ Conexión SSH exitosa"

# Crear directorio remoto
echo -e "${YELLOW}[3/5] Creando directorio remoto...${NC}"
$SSH_CMD "$SSH_TARGET" "mkdir -p $REMOTE_DIR"
echo "✅ Directorio creado: $REMOTE_DIR"

# Subir bundle
echo -e "${YELLOW}[4/5] Subiendo bundle (esto puede tomar varios minutos)...${NC}"
RSYNC_CMD="rsync -avz --progress -e 'ssh -p $SSH_PORT"
if [ -n "$SSH_KEY" ]; then
    RSYNC_CMD="$RSYNC_CMD -i $SSH_KEY"
fi
RSYNC_CMD="$RSYNC_CMD' --delete --exclude '.DS_Store' --exclude '._*' --exclude '__MACOSX'"

eval "$RSYNC_CMD $BUNDLE_DIR/ $SSH_TARGET:$REMOTE_DIR/"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Bundle subido exitosamente${NC}"
else
    echo -e "${RED}Error al subir bundle${NC}"
    exit 1
fi

# Verificar archivos remotos
echo -e "${YELLOW}[5/5] Verificando archivos remotos...${NC}"
$SSH_CMD "$SSH_TARGET" "ls -lh $REMOTE_DIR/project.zip $REMOTE_DIR/run_update.sh"
echo "✅ Archivos verificados"

echo ""
echo -e "${GREEN}=== Bundle subido exitosamente ===${NC}"
echo ""

# Ejecutar actualización si AUTO_UPDATE=1
if [ "$AUTO_UPDATE" = "1" ]; then
    echo -e "${YELLOW}Ejecutando actualización automática...${NC}"
    echo ""
    $SSH_CMD "$SSH_TARGET" "bash $REMOTE_DIR/run_update.sh"
    echo ""
    echo -e "${GREEN}✅ Actualización completada${NC}"
else
    echo "Para ejecutar la actualización, conéctate por SSH y ejecuta:"
    echo ""
    echo -e "${YELLOW}  ssh $SSH_TARGET${NC}"
    echo -e "${YELLOW}  bash $REMOTE_DIR/run_update.sh${NC}"
    echo ""
    echo "O ejecuta este script con AUTO_UPDATE=1:"
    echo ""
    echo -e "${YELLOW}  AUTO_UPDATE=1 $0 $SSH_TARGET${NC}"
fi

echo ""
echo "Comandos útiles:"
echo "  Ver logs: ssh $SSH_TARGET 'sudo journalctl -u cantina-face -f'"
echo "  Reiniciar: ssh $SSH_TARGET 'sudo systemctl restart cantina-face'"
echo "  Estado: ssh $SSH_TARGET 'sudo systemctl status cantina-face'"
