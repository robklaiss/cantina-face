#!/usr/bin/env bash
# 00_bootstrap_system.sh — Paquetes del sistema, Chrome, energía
# Debe ejecutarse como root (o con sudo).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

log "=== 00_bootstrap_system.sh ==="

# ─── Paquetes base ────────────────────────────────────────────────────────────
REQUIRED_PACKAGES=(
    python3 python3-venv python3-dev python3-pip
    build-essential g++
    sqlite3
    rsync curl wget gnupg ca-certificates
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
    v4l-utils ffmpeg
)

log "Actualizando índice de paquetes..."
apt-get update -qq

log "Instalando paquetes base..."
apt-get install -y --no-install-recommends "${REQUIRED_PACKAGES[@]}"

# ─── Entorno gráfico (GDM + desktop) ──────────────────────────────────────────
ensure_graphical_stack() {
    local desktop_packages=(ubuntu-desktop-minimal gdm3)

    log "Instalando entorno gráfico (ubuntu-desktop-minimal + gdm3)..."
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${desktop_packages[@]}"

    log "Habilitando gdm3 y graphical.target..."
    systemctl enable gdm3.service
    systemctl set-default graphical.target
}

ensure_graphical_stack

# ─── Google Chrome (repo oficial) ─────────────────────────────────────────────
install_chrome() {
    if command -v google-chrome-stable >/dev/null 2>&1; then
        log "Google Chrome ya está instalado: $(google-chrome-stable --version 2>/dev/null || true)"
    else
        log "Instalando Google Chrome desde repo oficial..."
        # Agregar keyring
        wget -qO- https://dl.google.com/linux/linux_signing_key.pub \
            | gpg --dearmor -o /usr/share/keyrings/google-chrome-keyring.gpg 2>/dev/null || true

        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome-keyring.gpg] \
http://dl.google.com/linux/chrome/deb/ stable main" \
            > /etc/apt/sources.list.d/google-chrome.list

        apt-get update -qq
        apt-get install -y google-chrome-stable
    fi

    # Fix sandbox permissions
    local sandbox="/opt/google/chrome/chrome-sandbox"
    if [ -f "$sandbox" ]; then
        log "Aplicando fix de sandbox a Chrome..."
        chown root:root "$sandbox"
        chmod 4755 "$sandbox"
    fi
}

install_chrome

# ─── Energía: desactivar suspensión / blank screen ───────────────────────────
configure_power() {
    log "Configurando energía (sin suspensión, sin blank screen)..."

    # GNOME Desktop (si hay sesión gráfica)
    if command -v gsettings >/dev/null 2>&1; then
        # Intentar con el usuario target; si falla, no es crítico
        local target_user="${SILOE_USER:-${SUDO_USER:-}}"
        if [ -n "$target_user" ] && id "$target_user" >/dev/null 2>&1; then
            local target_home
            target_home="$(getent passwd "$target_user" | cut -d: -f6 || true)"
            if [ -n "$target_home" ]; then
                log "  Aplicando gsettings para usuario $target_user..."
                su - "$target_user" -c "
                    export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/\$(id -u)/bus 2>/dev/null || true
                    gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true
                    gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' 2>/dev/null || true
                    gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null || true
                    gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null || true
                " 2>/dev/null || log "  (gsettings no disponible para $target_user, continuando)"
            fi
        fi
    fi

    # logind: desactivar idle action (funciona sin GUI)
    local logind_conf="/etc/systemd/logind.conf"
    if [ -f "$logind_conf" ]; then
        # IdleAction=ignore
        if grep -q "^IdleAction=" "$logind_conf"; then
            sed -i 's/^IdleAction=.*/IdleAction=ignore/' "$logind_conf"
        elif grep -q "^#IdleAction=" "$logind_conf"; then
            sed -i 's/^#IdleAction=.*/IdleAction=ignore/' "$logind_conf"
        else
            echo "IdleAction=ignore" >> "$logind_conf"
        fi

        # HandleLidSwitch=ignore (por si acaso, no daña en mini-PC)
        if grep -q "^HandleLidSwitch=" "$logind_conf"; then
            sed -i 's/^HandleLidSwitch=.*/HandleLidSwitch=ignore/' "$logind_conf"
        elif grep -q "^#HandleLidSwitch=" "$logind_conf"; then
            sed -i 's/^#HandleLidSwitch=.*/HandleLidSwitch=ignore/' "$logind_conf"
        else
            echo "HandleLidSwitch=ignore" >> "$logind_conf"
        fi

        systemctl restart systemd-logind 2>/dev/null || true
    fi

    # Desactivar suspend via systemd targets
    systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target 2>/dev/null || true
}

configure_power

log "=== 00_bootstrap_system.sh completado ==="
