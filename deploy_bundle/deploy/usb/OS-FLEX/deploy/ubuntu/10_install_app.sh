#!/usr/bin/env bash
# 10_install_app.sh — Copiar app a CANTINA_DIR, crear venv, instalar deps
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

log "=== 10_install_app.sh ==="

# ─── Crear directorio destino ─────────────────────────────────────────────────
if [ ! -d "$CANTINA_DIR" ]; then
    log "Creando $CANTINA_DIR..."
    mkdir -p "$CANTINA_DIR"
fi

# ─── Sincronizar archivos del repo ────────────────────────────────────────────
log "Sincronizando archivos desde $REPO_DIR a $CANTINA_DIR..."

rsync -a --delete \
    --exclude='.git' \
    --exclude='venv/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    "$REPO_DIR/" "$CANTINA_DIR/"

# Asegurar permisos ejecutables para scripts dentro de deploy/ubuntu
if [ -d "$CANTINA_DIR/deploy/ubuntu" ]; then
    chmod +x "$CANTINA_DIR"/deploy/ubuntu/*.sh 2>/dev/null || true
    chmod +x "$CANTINA_DIR/deploy/ubuntu/smoketest_face.py" 2>/dev/null || true
fi

# ─── Asegurar permisos del usuario target ─────────────────────────────────────
log "Asignando propiedad de $CANTINA_DIR a $SILOE_USER..."
chown -R "$SILOE_USER":"$(id -gn "$SILOE_USER")" "$CANTINA_DIR"

# ─── Crear venv si no existe ──────────────────────────────────────────────────
VENV_DIR="$CANTINA_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    log "Creando entorno virtual en $VENV_DIR..."
    sudo -u "$SILOE_USER" python3 -m venv "$VENV_DIR"
else
    log "Entorno virtual ya existe en $VENV_DIR"
fi

# ─── Instalar dependencias ───────────────────────────────────────────────────
log "Instalando dependencias Python..."
sudo -u "$SILOE_USER" bash -c "
    source '$VENV_DIR/bin/activate'
    python -m pip install --upgrade pip -q
    pip install --no-deps 'bcrypt==3.2.2' 'passlib[bcrypt]==1.7.4' -q
    pip install -r '$CANTINA_DIR/requirements.txt' -q
"

# ─── Crear directorios de datos ──────────────────────────────────────────────
log "Asegurando directorios de datos..."
sudo -u "$SILOE_USER" mkdir -p "$CANTINA_DIR/data/faces"
sudo -u "$SILOE_USER" mkdir -p "$CANTINA_DIR/models"

# ─── Copiar .env-claves si existe en el repo y no en destino ─────────────────
if [ -f "$REPO_DIR/.env-claves" ] && [ ! -f "$CANTINA_DIR/.env-claves" ]; then
    log "Copiando .env-claves al destino..."
    cp "$REPO_DIR/.env-claves" "$CANTINA_DIR/.env-claves"
    chown "$SILOE_USER":"$(id -gn "$SILOE_USER")" "$CANTINA_DIR/.env-claves"
    chmod 600 "$CANTINA_DIR/.env-claves"
elif [ ! -f "$CANTINA_DIR/.env-claves" ] && [ -f "$REPO_DIR/.env-claves.example" ]; then
    log "Copiando .env-claves.example como .env-claves (editá los valores)..."
    cp "$REPO_DIR/.env-claves.example" "$CANTINA_DIR/.env-claves"
    chown "$SILOE_USER":"$(id -gn "$SILOE_USER")" "$CANTINA_DIR/.env-claves"
    chmod 600 "$CANTINA_DIR/.env-claves"
fi

log "=== 10_install_app.sh completado ==="
