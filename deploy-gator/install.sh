#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Cantina Face – AWS EC2 (Ubuntu 22.04/24.04) bootstrap script
#
# Uso:
#   scp project.zip ubuntu@<IP>:~/
#   ssh ubuntu@<IP>
#   bash install.sh                          # primera instalación
#   bash install.sh --domain cantina.ejemplo.com   # con SSL (Let's Encrypt)
#   bash install.sh --update                 # actualizar código existente
#
# Variables de entorno opcionales:
#   APP_USER        usuario del sistema que correrá la app  (default: cantina)
#   APP_DIR         directorio de instalación               (default: /opt/cantina-face)
#   APP_PORT        puerto interno de uvicorn               (default: 8000)
#   PYTHON_BIN      binario de Python                       (default: python3)
#   ZIP_PATH        ruta al project.zip                     (default: junto al script)
#   SKIP_NGINX      1 para omitir configuración de nginx    (default: 0)
#   SKIP_SSL        1 para omitir certbot/SSL               (default: 0)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
APP_USER="${APP_USER:-cantina}"
APP_DIR="${APP_DIR:-/opt/cantina-face}"
APP_PORT="${APP_PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
zip_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_PATH="${ZIP_PATH:-$zip_script_dir/project.zip}"
SKIP_NGINX="${SKIP_NGINX:-0}"
SKIP_SSL="${SKIP_SSL:-0}"
DOMAIN=""
UPDATE_ONLY=0
SERVICE_NAME="cantina-face"
STATIC_DIR="/var/www/cantina-static"
GUARDRAIL_REL_PATH="deploy/guardrails/check_python_hardcode.sh"
LOCAL_GUARDRAIL="$zip_script_dir/../$GUARDRAIL_REL_PATH"

# ── Parse args ───────────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
Uso:
  bash install.sh [opciones]

Opciones:
  --domain DOMINIO    Configura nginx + SSL con Let's Encrypt para ese dominio
  --update            Solo actualiza código y dependencias (no reinstala sistema)
  --skip-nginx        Omite configuración de nginx
  --skip-ssl          Omite certbot (útil si ya tienes SSL o quieres HTTP)
  -h, --help          Muestra esta ayuda
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)
            DOMAIN="${2:-}"
            [[ -z "$DOMAIN" ]] && { echo "Error: --domain requiere un valor" >&2; exit 1; }
            shift 2 ;;
        --update)
            UPDATE_ONLY=1; shift ;;
        --skip-nginx)
            SKIP_NGINX=1; shift ;;
        --skip-ssl)
            SKIP_SSL=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Opción desconocida: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -f "$LOCAL_GUARDRAIL" ]]; then
    if [[ ! -x "$LOCAL_GUARDRAIL" ]]; then
        chmod +x "$LOCAL_GUARDRAIL" 2>/dev/null || true
    fi
    bash "$LOCAL_GUARDRAIL"
else
    warn "Guardrail local no encontrado en $LOCAL_GUARDRAIL; se validará al desplegar el ZIP."
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*" >&2; }
fail()  { echo -e "\033[1;31m[FAIL]\033[0m  $*" >&2; exit 1; }

need_root() {
    [[ "$EUID" -eq 0 ]] || fail "Este script debe ejecutarse como root (usa sudo bash install.sh ...)"
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

need_root

# ── 1. Paquetes del sistema ──────────────────────────────────────────────────
install_system_packages() {
    info "Actualizando índice de paquetes..."
    apt-get update -qq

    local pkgs=(
        software-properties-common
        build-essential g++
        python3 python3-venv python3-dev python3-pip
        rsync unzip sqlite3 curl
        nginx certbot python3-certbot-nginx
        ufw
    )

    info "Instalando dependencias del sistema..."
    apt-get install -y -qq "${pkgs[@]}"

    ok "Paquetes del sistema instalados"
}

# ── 2. Usuario del sistema ───────────────────────────────────────────────────
ensure_app_user() {
    if id "$APP_USER" &>/dev/null; then
        info "Usuario $APP_USER ya existe"
    else
        info "Creando usuario $APP_USER..."
        useradd --system --shell /usr/sbin/nologin --home-dir "$APP_DIR" "$APP_USER"
        ok "Usuario $APP_USER creado"
    fi
}

run_guardrail() {
    local guardrail="$1/$GUARDRAIL_REL_PATH"
    if [[ ! -f "$guardrail" ]]; then
        fail "Guardrail obligatorio no encontrado en $guardrail"
    fi
    if [[ ! -x "$guardrail" ]]; then
        chmod +x "$guardrail" 2>/dev/null || true
    fi
    bash "$guardrail"
}

# ── 3. Desplegar código ─────────────────────────────────────────────────────
deploy_code() {
    if [[ ! -f "$ZIP_PATH" ]]; then
        fail "No se encontró $ZIP_PATH. Sube project.zip al servidor primero."
    fi

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    trap "rm -rf $tmp_dir" RETURN

    info "Descomprimiendo $ZIP_PATH..."
    unzip -q "$ZIP_PATH" -d "$tmp_dir"

    # Detectar directorio raíz dentro del zip
    local src_dir="$tmp_dir"
    if [[ -d "$tmp_dir/cantina-face" ]]; then
        src_dir="$tmp_dir/cantina-face"
    else
        local candidate
        candidate="$(find "$tmp_dir" -maxdepth 1 -mindepth 1 -type d -name 'cantina-face*' | head -n 1 || true)"
        [[ -n "$candidate" ]] && src_dir="$candidate"
    fi

    # Limpiar artefactos macOS
    find "$src_dir" -name '__MACOSX' -type d -prune -exec rm -rf {} + 2>/dev/null || true
    find "$src_dir" -name '.DS_Store' -delete 2>/dev/null || true
    find "$src_dir" -name '._*' -delete 2>/dev/null || true

    # Crear directorio destino
    mkdir -p "$APP_DIR"

    # Preservar data/ si existe
    local data_backup=""
    if [[ -d "$APP_DIR/data" ]]; then
        data_backup="$(mktemp -d)"
        info "Preservando data/..."
        mv "$APP_DIR/data" "$data_backup/data"
    fi

    # Preservar .env-claves si existe
    local env_backup=""
    if [[ -f "$APP_DIR/.env-claves" ]]; then
        env_backup="$(mktemp)"
        cp "$APP_DIR/.env-claves" "$env_backup"
    fi

    run_guardrail "$src_dir"

    # Sync código
    rsync -a --delete \
        --exclude='venv' \
        --exclude='data' \
        --exclude='.env-claves' \
        --exclude='deploy/backups' \
        --exclude='.DS_Store' \
        --exclude='._*' \
        "$src_dir/" "$APP_DIR/"

    # Restaurar data/
    if [[ -n "$data_backup" && -d "$data_backup/data" ]]; then
        mv "$data_backup/data" "$APP_DIR/data"
        rm -rf "$data_backup"
    fi
    mkdir -p "$APP_DIR/data"

    # Restaurar .env-claves
    if [[ -n "$env_backup" && -f "$env_backup" ]]; then
        mv "$env_backup" "$APP_DIR/.env-claves"
    fi

    chown -R "$APP_USER:$APP_USER" "$APP_DIR"
    ok "Código desplegado en $APP_DIR"
}

# ── 4. Entorno virtual + dependencias Python ────────────────────────────────
setup_venv() {
    local venv_dir="$APP_DIR/venv"

    if [[ ! -d "$venv_dir" ]]; then
        info "Creando entorno virtual..."
        "$PYTHON_BIN" -m venv "$venv_dir"
    fi

    info "Instalando dependencias Python..."
    source "$venv_dir/bin/activate"
    python -m ensurepip --upgrade >/dev/null 2>&1 || true
    python -m pip install --upgrade pip -q
    python -m pip install --no-deps "bcrypt==3.2.2" "passlib[bcrypt]==1.7.4" -q
    python -m pip install -r "$APP_DIR/requirements.txt" -q
    deactivate

    chown -R "$APP_USER:$APP_USER" "$venv_dir"
    ok "Dependencias Python instaladas"
}

# ── 5. Archivo .env-claves ──────────────────────────────────────────────────
setup_env_file() {
    local env_file="$APP_DIR/.env-claves"

    if [[ -f "$env_file" ]]; then
        info ".env-claves ya existe, no se sobreescribe"
        return
    fi

    # Generar SECRET_KEY aleatorio
    local secret_key
    secret_key="$(openssl rand -hex 32)"

    cat > "$env_file" <<EOF
SECRET_KEY="$secret_key"
ADMIN_EMAIL="admin@siloe.com.py"
ADMIN_PASSWORD="admin321"
ACCESS_TOKEN_EXPIRE_MINUTES="120"
LOCAL_TIMEZONE="America/Asuncion"
EOF

    chown "$APP_USER:$APP_USER" "$env_file"
    chmod 600 "$env_file"

    warn "Se creó $env_file con valores por defecto."
    warn "¡CAMBIA ADMIN_PASSWORD y revisa los valores antes de usar en producción!"
    ok ".env-claves configurado"
}

# ── 6. Servicio systemd ─────────────────────────────────────────────────────
setup_systemd() {
    local service_file="/etc/systemd/system/${SERVICE_NAME}.service"

    info "Configurando servicio systemd..."
    cat > "$service_file" <<EOF
[Unit]
Description=Cantina Face (FastAPI)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
Environment=PYTHONUNBUFFERED=1
Environment=HOST=127.0.0.1
Environment=PORT=$APP_PORT
ExecStart=$APP_DIR/venv/bin/uvicorn app:app --host 127.0.0.1 --port $APP_PORT
Restart=always
RestartSec=5

# Seguridad
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$APP_DIR/data $APP_DIR
ProtectHome=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
    systemctl restart "$SERVICE_NAME"
    ok "Servicio $SERVICE_NAME activo"
}

# ── 7. Desplegar frontends estáticos (backend/, padres/, updates/) ──────────
deploy_static_sites() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    info "Desplegando frontends estáticos en $STATIC_DIR..."
    mkdir -p "$STATIC_DIR"

    for subdir in backend padres updates; do
        local src="$script_dir/$subdir"
        if [[ -d "$src" ]]; then
            rsync -a --delete "$src/" "$STATIC_DIR/$subdir/"
            ok "$subdir/ desplegado"
        else
            warn "$src no encontrado, omitiendo"
        fi
    done

    chown -R www-data:www-data "$STATIC_DIR"
    ok "Frontends estáticos desplegados en $STATIC_DIR"
}

# ── 8. Nginx reverse proxy ──────────────────────────────────────────────────
setup_nginx() {
    if [[ "$SKIP_NGINX" == "1" ]]; then
        info "Omitiendo configuración de nginx (SKIP_NGINX=1)"
        return
    fi

    local server_name="${DOMAIN:-_}"
    local conf_file="/etc/nginx/sites-available/$SERVICE_NAME"

    info "Configurando nginx..."
    cat > "$conf_file" <<EOF
server {
    listen 80;
    server_name $server_name;

    client_max_body_size 10M;

    # Frontends estáticos
    location /backend {
        alias $STATIC_DIR/backend;
        index index.html;
        try_files \$uri \$uri/ /backend/index.html;
    }

    location /padres {
        alias $STATIC_DIR/padres;
        index index.html;
        try_files \$uri \$uri/ /padres/index.html;
    }

    location /updates {
        alias $STATIC_DIR/updates;
        autoindex off;
    }

    # Todo lo demás va al API (FastAPI)
    location / {
        proxy_pass http://127.0.0.1:$APP_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # WebSocket support (si se necesita en el futuro)
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

    # Habilitar sitio
    ln -sf "$conf_file" /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default

    nginx -t || fail "Configuración de nginx inválida"
    systemctl reload nginx
    ok "Nginx configurado (server_name=$server_name)"
}

# ── 9. SSL con Let's Encrypt ────────────────────────────────────────────────
setup_ssl() {
    if [[ "$SKIP_SSL" == "1" || -z "$DOMAIN" ]]; then
        [[ -z "$DOMAIN" ]] && info "Sin --domain, omitiendo SSL"
        [[ "$SKIP_SSL" == "1" ]] && info "Omitiendo SSL (SKIP_SSL=1)"
        return
    fi

    info "Solicitando certificado SSL para $DOMAIN..."
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos \
        --redirect --email "admin@${DOMAIN}" \
        || warn "certbot falló. Puedes ejecutarlo manualmente después: certbot --nginx -d $DOMAIN"

    ok "SSL configurado para $DOMAIN"
}

# ── 10. Firewall ─────────────────────────────────────────────────────────────
setup_firewall() {
    info "Configurando firewall (ufw)..."
    ufw --force reset >/dev/null 2>&1 || true
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow ssh
    ufw allow 'Nginx Full'
    ufw --force enable
    ok "Firewall configurado (SSH + HTTP/HTTPS)"
}

# ── 11. Descargar modelo ONNX si no existe ──────────────────────────────────
ensure_model() {
    local model_path="$APP_DIR/models/mobile_face.onnx"
    local model_url="https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/mobilefacenet-arcface.onnx"

    if [[ -f "$model_path" ]]; then
        info "Modelo ONNX ya existe"
        return
    fi

    info "Descargando modelo ONNX..."
    mkdir -p "$APP_DIR/models"
    curl -fSL "$model_url" -o "$model_path" || warn "No se pudo descargar el modelo. La app lo descargará al iniciar."
    chown -R "$APP_USER:$APP_USER" "$APP_DIR/models"
    ok "Modelo descargado"
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$UPDATE_ONLY" == "1" ]]; then
    info "═══ Modo actualización ═══"
    deploy_code
    setup_venv
    deploy_static_sites
    setup_systemd
    ok "Actualización completada. Servicio reiniciado."
    echo ""
    echo "  Estado: sudo systemctl status $SERVICE_NAME"
    echo "  Logs:   sudo journalctl -u $SERVICE_NAME -f"
    exit 0
fi

info "═══ Instalación completa de Cantina Face en EC2 ═══"
echo ""

install_system_packages
ensure_app_user
deploy_code
setup_venv
setup_env_file
ensure_model
setup_systemd
deploy_static_sites
setup_nginx
setup_ssl
setup_firewall

echo ""
ok "═══ Instalación completada ═══"
echo ""
echo "  App:     http://${DOMAIN:-<IP_PUBLICA>}"
[[ -n "$DOMAIN" ]] && echo "  SSL:     https://$DOMAIN"
echo "  Padres:  https://${DOMAIN:-<IP_PUBLICA>}/padres"
echo "  Backend: https://${DOMAIN:-<IP_PUBLICA>}/backend"
echo "  Estado:  sudo systemctl status $SERVICE_NAME"
echo "  Logs:    sudo journalctl -u $SERVICE_NAME -f"
echo "  Config:  $APP_DIR/.env-claves"
echo ""
warn "Recuerda editar $APP_DIR/.env-claves con credenciales seguras para producción."
echo ""
