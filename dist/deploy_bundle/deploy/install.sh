#!/usr/bin/env bash
set -euo pipefail

# Deterministic path calculation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DEPLOY_DIR="$SCRIPT_DIR"
TARGET_APP_DIR="${TARGET_APP_DIR:-/opt/cantina-face}"
VENV_DIR="$ROOT_DIR/venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIRED_APT_PACKAGES=(python3 python3-venv python3-dev python3-pip build-essential g++ rsync sqlite3)
GUARDRAIL_SCRIPT="$DEPLOY_DIR/guardrails/check_python_hardcode.sh"
MODELS_SRC="$ROOT_DIR/models"
MODELS_DST="$TARGET_APP_DIR/models"
OPTIMIZE_DEST=""
SETUP_AUTOSTART=0
FORCE_VENV="${FORCE_VENV:-0}"

usage() {
    cat <<'EOF'
Uso:
  deploy/install.sh              # instala dependencias y venv locales
  deploy/install.sh --optimize-facial DESTINO
  deploy/install.sh --setup-autostart [opciones]

La opción --optimize-facial sincroniza únicamente config.py y static/app.js
al DESTINO (local o remoto user@host:/ruta) y finaliza.
--setup-autostart (combinable con la instalación) crea el servicio systemd,
el autostart gráfico y el acceso directo de escritorio. Variables opcionales:
  SERVICE_NAME, LOGIN_URL, TARGET_USER, AUTOSTART_DISPLAY, ICON_PATH_OVERRIDE
EOF
}

run_face_backup() {
    local backup_script="$DEPLOY_DIR/backup_faces.sh"

    if [ "${SKIP_FACE_BACKUP:-0}" = "1" ]; then
        echo "[install] SKIP_FACE_BACKUP=1, omitiendo respaldo facial"
        return
    fi

    if [ ! -f "$backup_script" ]; then
        echo "[install] No se encontró $backup_script; omitiendo respaldo facial" >&2
        return
    fi

    echo "[install] Generando copia de seguridad facial (db/index/faces)..."
    if ! bash "$backup_script"; then
        echo "[install] Advertencia: backup_faces.sh retornó un error" >&2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --optimize-facial)
            OPTIMIZE_DEST="${2:-}"
            if [[ -z "$OPTIMIZE_DEST" ]]; then
                echo "Error: --optimize-facial requiere un destino" >&2
                usage
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --setup-autostart)
            SETUP_AUTOSTART=1
            shift
            ;;
        *)
            echo "Opción desconocida: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [ ! -f "$GUARDRAIL_SCRIPT" ]; then
    echo "[install] Guardrail obligatorio no encontrado en $GUARDRAIL_SCRIPT" >&2
    exit 1
fi
bash "$GUARDRAIL_SCRIPT"

sync_optimize_facial() {
    if ! command -v rsync >/dev/null 2>&1; then
        echo "Error: se requiere 'rsync' para --optimize-facial" >&2
        exit 1
    fi

    local files=(
        "config.py"
        "static/app.js"
    )

    local sources=()
    local relpath
    for relpath in "${files[@]}"; do
        local src="$ROOT_DIR/$relpath"
        if [[ ! -f "$src" ]]; then
            echo "Error: no se encontró $src" >&2
            exit 1
        fi
        sources+=("$ROOT_DIR/./$relpath")
    done

    if [[ "$OPTIMIZE_DEST" != *:* ]]; then
        mkdir -p "$OPTIMIZE_DEST"
    fi

    rsync -aR --info=stats1 "${sources[@]}" "$OPTIMIZE_DEST/"
    echo "Archivos sincronizados en $OPTIMIZE_DEST. Reinicia servicios si aplica."
}

if [[ -n "$OPTIMIZE_DEST" ]]; then
    sync_optimize_facial
    exit 0
fi

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

as_root() {
    if [ "$EUID" -eq 0 ]; then
        "$@"
    else
        if ! have_cmd sudo; then
            echo "Se requiere 'sudo' para ejecutar esta acción." >&2
            exit 1
        fi
        sudo "$@"
    fi
}

install_file_for_user() {
    local src="$1"
    local dest="$2"
    local mode="$3"
    local owner="$4"
    local group="$5"

    if [ "$EUID" -eq 0 ]; then
        install -D -o "$owner" -g "$group" -m "$mode" "$src" "$dest"
    else
        install -D -m "$mode" "$src" "$dest"
    fi
}

install_system_packages() {
    if ! have_cmd apt; then
        echo "[install] 'apt' no está disponible. Instala manualmente: ${REQUIRED_APT_PACKAGES[*]}" >&2
        return
    fi

    local sudo_cmd=""
    if have_cmd sudo; then
        sudo_cmd="sudo -E"
    elif [ "$(id -u)" -ne 0 ]; then
        echo "Necesitas privilegios de administrador para instalar dependencias. Ejecuta nuevamente con sudo." >&2
        exit 1
    fi

    echo "Actualizando índice de paquetes..."
    $sudo_cmd apt-get update
    echo "Instalando dependencias del sistema..."
    $sudo_cmd apt-get install -y "${REQUIRED_APT_PACKAGES[@]}"
}

is_crostini() {
    # Detect ChromeOS Linux container (Crostini)
    [ -f /dev/.cros_milestone ] || grep -qsi cros /proc/version 2>/dev/null || [ -d /opt/google/cros-containers ] 2>/dev/null
}

setup_autostart() {
    local service_name="${SERVICE_NAME:-cantina-face}"
    local login_url="${LOGIN_URL:-http://localhost:8000/login.html}"
    local display_value="${AUTOSTART_DISPLAY:-:0}"
    local icon_path="${ICON_PATH_OVERRIDE:-$ROOT_DIR/siloe-logo-blanco.png}"
    local run_script="$DEPLOY_DIR/run.sh"

    if [ ! -f "$run_script" ]; then
        echo "No se encontró $run_script" >&2
        exit 1
    fi
    chmod +x "$run_script"

    if ! command -v systemctl >/dev/null 2>&1; then
        echo "systemctl no está disponible. Se requiere systemd para --setup-autostart." >&2
        exit 1
    fi

    local target_user
    if [ -n "${TARGET_USER:-}" ]; then
        target_user="$TARGET_USER"
    elif [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
        target_user="$SUDO_USER"
    elif [ "$EUID" -ne 0 ]; then
        target_user="$USER"
    else
        echo "Define TARGET_USER=<usuario> cuando ejecutes como root." >&2
        exit 1
    fi

    if [ "$target_user" = "root" ]; then
        echo "El autostart debe pertenecer a un usuario normal. Usa TARGET_USER=<usuario>." >&2
        exit 1
    fi

    if ! id "$target_user" >/dev/null 2>&1; then
        echo "El usuario $target_user no existe." >&2
        exit 1
    fi

    local target_group
    target_group="$(id -gn "$target_user")"

    local target_home=""
    if [ "$target_user" = "$USER" ] && [ "$EUID" -ne 0 ] && [ -n "${HOME:-}" ]; then
        target_home="$HOME"
    else
        target_home="$(getent passwd "$target_user" | cut -d: -f6 || true)"
        if [ -z "$target_home" ]; then
            target_home="$(eval echo "~$target_user")"
        fi
    fi

    if [ -z "$target_home" ] || [ ! -d "$target_home" ]; then
        echo "No se pudo determinar el HOME del usuario $target_user." >&2
        exit 1
    fi

    local autostart_entry="$target_home/.config/autostart/cantina-face-login.desktop"
    local desktop_entry="$target_home/Desktop/CantinaFace.desktop"
    local service_file="/etc/systemd/system/${service_name}.service"

    # --- ChromeOS Flex / Crostini path ---
    if is_crostini; then
        echo "[autostart] Detectado entorno ChromeOS (Crostini)"

        # In Crostini, use user-level systemd for the backend service
        local user_service_dir="$target_home/.config/systemd/user"
        mkdir -p "$user_service_dir"

        cat >"$user_service_dir/${service_name}.service" <<EOF
[Unit]
Description=Cantina Face (FastAPI + reconocimiento facial)
After=default.target

[Service]
Type=simple
WorkingDirectory=$ROOT_DIR
ExecStart=/bin/bash $run_script
Restart=on-failure
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

        # Enable user-level service (runs when Linux container starts)
        systemctl --user daemon-reload
        systemctl --user enable "$service_name"
        systemctl --user restart "$service_name"
        # Ensure user services start on boot (linger)
        if have_cmd loginctl; then
            loginctl enable-linger "$target_user" 2>/dev/null || true
        fi

        # .desktop in ~/.local/share/applications is exported to ChromeOS launcher
        local cros_app_dir="$target_home/.local/share/applications"
        mkdir -p "$cros_app_dir"

        # Startup script that ensures server is running then opens browser
        local starter_script="$DEPLOY_DIR/cantina-start-chromeos.sh"
        cat >"$starter_script" <<SCRIPT
#!/bin/bash
# Ensure the backend service is running
if command -v systemctl >/dev/null 2>&1; then
    systemctl --user start ${service_name} 2>/dev/null || true
fi
# Wait for the server to be ready (max 15s)
for i in \$(seq 1 30); do
    if curl -sf http://localhost:8000/docs >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done
# Open browser (garcon-url-handler works in Crostini, fallback to xdg-open)
if command -v garcon-url-handler >/dev/null 2>&1; then
    garcon-url-handler "$login_url"
else
    xdg-open "$login_url"
fi
SCRIPT
        chmod +x "$starter_script"

        # App launcher entry (visible in ChromeOS app drawer)
        cat >"$cros_app_dir/cantina-face.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Cantina Face
Comment=Sistema de reconocimiento facial - Cantina
Exec=/bin/bash $starter_script
Icon=$icon_path
Terminal=false
StartupNotify=true
Categories=Utility;
EOF

        # Autostart entry (runs when Linux container starts)
        mkdir -p "$target_home/.config/autostart"
        cat >"$autostart_entry" <<EOF
[Desktop Entry]
Type=Application
Name=Cantina Face Login
Comment=Abrir el login de Cantina Face al iniciar sesión
Exec=/bin/bash $starter_script
Icon=$icon_path
X-GNOME-Autostart-enabled=true
Terminal=false
EOF

        cat <<EOM

✅ Autostart configurado para ChromeOS Flex
- Servicio systemd (usuario): ~/.config/systemd/user/${service_name}.service
- App en launcher de ChromeOS: $cros_app_dir/cantina-face.desktop
- Autostart al abrir Linux: $autostart_entry
- Script de inicio: $starter_script

📌 IMPORTANTE para inicio automático al encender:
   1. Abre chrome://flags en el navegador de ChromeOS
   2. Busca "Crostini" y activa "Start Linux on login" (#crostini-reset-lxd-db
      o #crostini-multi-container según la versión)
   3. O bien: ancla "Cantina Face" desde el launcher al estante (shelf)
      para iniciar con un solo clic.
EOM
        return
    fi

    # --- Standard Linux path (non-ChromeOS) ---
    local tmp_service tmp_autostart tmp_desktop
    tmp_service="$(mktemp)"
    cat >"$tmp_service" <<EOF
[Unit]
Description=Cantina Face (FastAPI + reconocimiento facial)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$ROOT_DIR
ExecStart=/bin/bash $run_script
Restart=on-failure
User=$target_user
Group=$target_group
Environment=DISPLAY=$display_value
Environment=PYTHONUNBUFFERED=1
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    as_root install -m 644 "$tmp_service" "$service_file"
    rm -f "$tmp_service"

    as_root systemctl daemon-reload
    as_root systemctl enable "$service_name"
    as_root systemctl restart "$service_name"

    tmp_autostart="$(mktemp)"
    cat >"$tmp_autostart" <<EOF
[Desktop Entry]
Type=Application
Name=Cantina Face Login
Comment=Abrir el login de Cantina Face al iniciar sesión
Exec=xdg-open $login_url
Icon=$icon_path
X-GNOME-Autostart-enabled=true
EOF
    install_file_for_user "$tmp_autostart" "$autostart_entry" 755 "$target_user" "$target_group"
    rm -f "$tmp_autostart"

    tmp_desktop="$(mktemp)"
    cat >"$tmp_desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Cantina Face
Comment=Iniciar Cantina Face y abrir el login
Exec=/bin/bash -c "\"$run_script\" & sleep 3 && xdg-open $login_url"
Icon=$icon_path
Terminal=false
EOF
    install_file_for_user "$tmp_desktop" "$desktop_entry" 755 "$target_user" "$target_group"
    rm -f "$tmp_desktop"

    cat <<EOM
✅ Autostart configurado
- Servicio systemd: $service_file
- Autostart: $autostart_entry
- Acceso directo: $desktop_entry
EOM
}

ensure_python() {
    if have_cmd "$PYTHON_BIN"; then
        return
    fi
    if have_cmd python3; then
        echo "Advertencia: no se encontró $PYTHON_BIN, usando python3 por defecto." >&2
        PYTHON_BIN="python3"
    else
        echo "Python3 no está instalado. Instálalo antes de continuar." >&2
        exit 1
    fi
}

install_system_packages
ensure_python

if [ "$FORCE_VENV" = "1" ] && [ -d "$VENV_DIR" ]; then
    echo "[install] FORCE_VENV=1 → eliminando entorno existente en $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install --upgrade pip
python -m pip install --no-deps "bcrypt==3.2.2" "passlib[bcrypt]==1.7.4"

# Buscar requirements.txt (primero en ROOT_DIR, luego extraer del ZIP si es necesario)
REQ_FILE=""
if [ -f "$ROOT_DIR/requirements.txt" ]; then
    REQ_FILE="$ROOT_DIR/requirements.txt"
elif [ -f "$ROOT_DIR/project.zip" ]; then
    echo "[install] Extrayendo requirements.txt desde project.zip..." >&2
    TMP_REQ="$(mktemp)"
    unzip -p "$ROOT_DIR/project.zip" '*/requirements.txt' 2>/dev/null > "$TMP_REQ" || \
    unzip -p "$ROOT_DIR/project.zip" 'requirements.txt' 2>/dev/null > "$TMP_REQ" || true
    if [ -s "$TMP_REQ" ]; then
        REQ_FILE="$TMP_REQ"
    else
        rm -f "$TMP_REQ"
        echo "[install] ERROR: No se pudo extraer requirements.txt del ZIP" >&2
        exit 1
    fi
else
    echo "[install] ERROR: No se encontró requirements.txt ni project.zip" >&2
    exit 1
fi

python -m pip install -r "$REQ_FILE"
[ -n "${TMP_REQ:-}" ] && rm -f "$TMP_REQ" || true

verify_bcrypt() {
    python - <<'PY'
import sys
import importlib

try:
    bcrypt = importlib.import_module("bcrypt")
except Exception as exc:  # pragma: no cover
    sys.exit(f"bcrypt import failed: {exc}")

if not hasattr(bcrypt, "__about__"):
    sys.exit("bcrypt.__about__ missing")
PY
}

if ! verify_bcrypt; then
    echo "[install] Reinstalando bcrypt==3.2.2 por compatibilidad con passlib" >&2
    python -m pip install --no-cache-dir --force-reinstall "bcrypt==3.2.2"
    verify_bcrypt
fi

# ─── Instalar modelos en TARGET_APP_DIR ──────────────────────────────────────
echo "[install] Instalando modelos en $MODELS_DST ..."
as_root mkdir -p "$MODELS_DST"
if [ -f "$MODELS_SRC/arcface_r50.onnx" ]; then
    as_root install -m 0644 "$MODELS_SRC/arcface_r50.onnx" "$MODELS_DST/arcface_r50.onnx"
    echo "[install] OK: arcface_r50.onnx instalado en $MODELS_DST"
    # Recrear symlink mobile_face.onnx -> arcface_r50.onnx
    as_root rm -f "$MODELS_DST/mobile_face.onnx"
    as_root ln -s arcface_r50.onnx "$MODELS_DST/mobile_face.onnx"
else
    echo "[install] FAIL: No se encontró arcface_r50.onnx en $MODELS_SRC" >&2
    echo "[install] El modelo es obligatorio. Verifica que el bundle incluya models/arcface_r50.onnx" >&2
    exit 1
fi

run_face_backup

if [ "$SETUP_AUTOSTART" -eq 1 ]; then
    setup_autostart
fi

# Instalar cloudflared si no está presente (para túnel remoto)
CLOUDFLARE_INSTALLER="$DEPLOY_DIR/install_cloudflare_auto.sh"
if [ -f "$CLOUDFLARE_INSTALLER" ]; then
    echo "[install] Verificando cloudflared para acceso remoto..."
    bash "$CLOUDFLARE_INSTALLER" || echo "[install] Advertencia: No se pudo instalar cloudflared"
fi

echo "Instalación completada. Puedes iniciar con deploy/run.sh"
