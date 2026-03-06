#!/usr/bin/env bash
# check_update.sh — Pull updater for Cantina Face
# Consults the remote manifest, downloads the zip if newer, validates sha256,
# and optionally runs deploy/update.sh + restarts the service.
#
# Usage:
#   deploy/check_update.sh                     # check + download + update
#   deploy/check_update.sh --check-only        # just print if update available
#   deploy/check_update.sh --no-restart        # update but skip service restart
#
# Environment overrides:
#   MANIFEST_URL   (default https://siloe.com.py/updates/manifest.json)
#   DOWNLOAD_DIR   (default $REPO_DIR/deploy/downloads)
#   SERVICE_NAME   (default cantina-face)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST_URL="${MANIFEST_URL:-https://siloe.com.py/updates/manifest.json}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$REPO_DIR/deploy/downloads}"
SERVICE_NAME="${SERVICE_NAME:-cantina-face}"
CHECK_ONLY=0
NO_RESTART=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only) CHECK_ONLY=1; shift ;;
        --no-restart) NO_RESTART=1; shift ;;
        -h|--help)
            sed -n '2,14p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Opción desconocida: $1" >&2; exit 1 ;;
    esac
done

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# Require curl or wget
if have_cmd curl; then
    http_get() { curl -fsSL "$1"; }
    http_download() { curl -fsSL -o "$2" "$1"; }
elif have_cmd wget; then
    http_get() { wget -qO- "$1"; }
    http_download() { wget -qO "$2" "$1"; }
else
    echo "Se requiere curl o wget." >&2
    exit 1
fi

# Require sha256sum or shasum
if have_cmd sha256sum; then
    sha256_of() { sha256sum "$1" | awk '{print $1}'; }
elif have_cmd shasum; then
    sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    echo "Se requiere sha256sum o shasum." >&2
    exit 1
fi

# Parse JSON value (minimal, no jq dependency)
json_val() {
    # Usage: json_val "key" < json_string
    local key="$1"
    local json="$2"
    echo "$json" | sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" | head -1
}

echo "[check_update] Consultando $MANIFEST_URL ..."
MANIFEST_JSON="$(http_get "$MANIFEST_URL")" || {
    echo "No se pudo descargar el manifest." >&2
    exit 1
}

REMOTE_VERSION="$(json_val "version" "$MANIFEST_JSON")"
REMOTE_SHA256="$(json_val "sha256" "$MANIFEST_JSON")"
REMOTE_ZIP_URL="$(json_val "zip_url" "$MANIFEST_JSON")"
REMOTE_NOTES="$(json_val "notes" "$MANIFEST_JSON")"

if [[ -z "$REMOTE_VERSION" || -z "$REMOTE_SHA256" || -z "$REMOTE_ZIP_URL" ]]; then
    echo "Manifest incompleto (faltan version, sha256 o zip_url)." >&2
    exit 1
fi

# Check local version
LOCAL_VERSION_FILE="$REPO_DIR/deploy/.current_version"
LOCAL_VERSION=""
if [[ -f "$LOCAL_VERSION_FILE" ]]; then
    LOCAL_VERSION="$(cat "$LOCAL_VERSION_FILE" | tr -d '[:space:]')"
fi

echo "[check_update] Versión local:  ${LOCAL_VERSION:-desconocida}"
echo "[check_update] Versión remota: $REMOTE_VERSION"
if [[ -n "$REMOTE_NOTES" ]]; then
    echo "[check_update] Notas: $REMOTE_NOTES"
fi

if [[ "$LOCAL_VERSION" == "$REMOTE_VERSION" ]]; then
    echo "[check_update] Ya estás en la última versión ($REMOTE_VERSION). Nada que hacer."
    exit 0
fi

echo "[check_update] Nueva versión disponible: $REMOTE_VERSION"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "[check_update] (--check-only) No se descarga ni actualiza."
    exit 0
fi

# Download
mkdir -p "$DOWNLOAD_DIR"
ZIP_FILE="$DOWNLOAD_DIR/project-${REMOTE_VERSION}.zip"

echo "[check_update] Descargando $REMOTE_ZIP_URL ..."
http_download "$REMOTE_ZIP_URL" "$ZIP_FILE"

# Validate SHA256
ACTUAL_SHA256="$(sha256_of "$ZIP_FILE")"
if [[ "$ACTUAL_SHA256" != "$REMOTE_SHA256" ]]; then
    echo "SHA256 no coincide." >&2
    echo "  Esperado: $REMOTE_SHA256" >&2
    echo "  Obtenido: $ACTUAL_SHA256" >&2
    rm -f "$ZIP_FILE"
    exit 1
fi
echo "[check_update] SHA256 verificado OK."

# Run update
UPDATE_SCRIPT="$REPO_DIR/deploy/update.sh"
if [[ ! -f "$UPDATE_SCRIPT" ]]; then
    echo "No se encontró $UPDATE_SCRIPT" >&2
    exit 1
fi

echo "[check_update] Ejecutando deploy/update.sh con $ZIP_FILE ..."
bash "$UPDATE_SCRIPT" "$ZIP_FILE"

# Save version
echo "$REMOTE_VERSION" > "$LOCAL_VERSION_FILE"
echo "[check_update] Versión actualizada a $REMOTE_VERSION"

# Restart service
if [[ "$NO_RESTART" -eq 1 ]]; then
    echo "[check_update] (--no-restart) Omitiendo reinicio del servicio."
    echo "[check_update] Para reiniciar manualmente:"
    echo "  sudo systemctl restart $SERVICE_NAME"
    echo "  # o bien: deploy/run.sh"
    exit 0
fi

if have_cmd systemctl && systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    echo "[check_update] Reiniciando servicio $SERVICE_NAME ..."
    if have_cmd sudo; then
        sudo systemctl restart "$SERVICE_NAME"
    else
        systemctl restart "$SERVICE_NAME"
    fi
    echo "[check_update] Servicio reiniciado."
else
    echo "[check_update] El servicio $SERVICE_NAME no está activo vía systemd."
    echo "[check_update] Reiniciá manualmente con: deploy/run.sh"
fi

echo "[check_update] Actualización completada."
