#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${LOG_FILE:-/var/log/cantina-face-install.log}"
CURRENT_STEP="inicio"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

warn() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $*" >&2
}

die() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
  exit 1
}

step() {
  CURRENT_STEP="$1"
  log "--- $1 ---"
}

on_error() {
  local exit_code=$?
  warn "El instalador falló en el paso '${CURRENT_STEP:-desconocido}' (línea $1). Revisá ${LOG_FILE} y corregí antes de reintentar."
  exit "$exit_code"
}

trap 'on_error $LINENO' ERR

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  exec sudo -E LOG_FILE="${LOG_FILE}" bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USB_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
APP_DIR="/opt/cantina-face"
SERVICE_NAME="cantina-face"
export USB_ROOT

mkdir -p "$(dirname "${LOG_FILE}")"
touch "${LOG_FILE}"
chmod 644 "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

APP_USER="${SUDO_USER:-}"
if [[ -z "${APP_USER}" ]]; then
  APP_USER="$(logname 2>/dev/null || true)"
fi
[[ -n "${APP_USER}" ]] || die "No pude determinar el usuario real (SUDO_USER/logname)."
APP_GROUP="$(id -gn "${APP_USER}")"

USB_PROJECT_DIR="${USB_ROOT}/project"
USB_DEPLOY_DIR="${USB_ROOT}/deploy"
USB_MODELS_DIR="${USB_ROOT}/models"
PROJECT_ZIP="${USB_DEPLOY_DIR}/project.zip"
USB_MODEL="${USB_MODELS_DIR}/arcface_r50.onnx"

cleanup_macos_trash() {
  local target="$1"
  [[ -d "$target" ]] || return 0
  find "$target" -name '._*' -type f -delete 2>/dev/null || true
  find "$target" -name '.DS_Store' -type f -delete 2>/dev/null || true
  find "$target" -type d -name '__MACOSX' -exec rm -rf {} + 2>/dev/null || true
  rm -rf "$target/.Spotlight-V100" "$target/.Trashes" "$target/.fseventsd"
}

ensure_project_from_zip() {
  [[ -f "$PROJECT_ZIP" ]] || die "Falta ${PROJECT_ZIP}. Copiá deploy/project.zip al USB y reintentá."
  local tmp_dir
  tmp_dir="$(mktemp -d /tmp/osflex_extract.XXXXXX)"
  log "Extrayendo ${PROJECT_ZIP} para generar project/…"
  unzip -q "$PROJECT_ZIP" -d "$tmp_dir"
  local extracted_root="$tmp_dir"
  local candidate
  candidate="$(find "$tmp_dir" -maxdepth 3 -type f -name app.py | head -n 1 || true)"
  if [[ -n "$candidate" ]]; then
    extracted_root="$(dirname "$candidate")"
  elif [[ -d "$tmp_dir/project" && -f "$tmp_dir/project/app.py" ]]; then
    extracted_root="$tmp_dir/project"
  fi
  mkdir -p "$USB_PROJECT_DIR"
  rsync -a --delete \
    --exclude 'deploy/' \
    --exclude 'models/' \
    --exclude '.DS_Store' \
    --exclude '._*' \
    "$extracted_root"/ "$USB_PROJECT_DIR"/
  cleanup_macos_trash "$USB_PROJECT_DIR"
  rm -rf "$tmp_dir"
}

ensure_usb_model() {
  mkdir -p "$USB_MODELS_DIR"
  if [[ -s "$USB_MODEL" ]]; then
    log "Modelo ya presente en ${USB_MODEL}"
    return
  fi

  if [[ -s "${USB_PROJECT_DIR}/models/arcface_r50.onnx" ]]; then
    log "Copiando modelo desde project/models…"
    install -m 644 "${USB_PROJECT_DIR}/models/arcface_r50.onnx" "$USB_MODEL"
    return
  fi

  if [[ -f "$PROJECT_ZIP" ]]; then
    local member
    member="$(unzip -Z1 "$PROJECT_ZIP" | grep -E '(^|/)arcface_r50\.onnx$' | head -n 1 || true)"
    if [[ -n "$member" ]]; then
      log "Extrayendo modelo desde ${PROJECT_ZIP} (${member})…"
      unzip -p "$PROJECT_ZIP" "$member" > "$USB_MODEL"
      chmod 644 "$USB_MODEL"
      return
    fi
  fi

  die "No se encontró arcface_r50.onnx. Copiá el modelo a ${USB_MODELS_DIR}/arcface_r50.onnx y reintentá."
}

log "=== Cantina Face Ubuntu installer (OFFLINE) ==="
log "Log completo: ${LOG_FILE}"
log "USB_ROOT=${USB_ROOT}"
log "APP_DIR=${APP_DIR}"
log "APP_USER=${APP_USER}:${APP_GROUP}"

[[ "${USB_ROOT}" == /media/*/OS-FLEX ]] || die "Este instalador sólo puede ejecutarse desde /media/<usuario>/OS-FLEX (actual: ${USB_ROOT})."
[[ -d "${USB_DEPLOY_DIR}/ubuntu" ]] || die "Falta deploy/ubuntu en el USB. Ejecutá make_usb_offline.sh nuevamente."

step "Verificando dependencias base"
REQUIRED_CMDS=(python3 python3-venv python3-pip rsync sqlite3 ffmpeg v4l2-ctl unzip)
MISSING_CMDS=()
for cmd in "${REQUIRED_CMDS[@]}"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    MISSING_CMDS+=("$cmd")
  fi
done

if ((${#MISSING_CMDS[@]})); then
  log "Faltan dependencias: ${MISSING_CMDS[*]}"
  export DEBIAN_FRONTEND=noninteractive
  if ! apt-get update -y >/dev/null 2>&1; then
    warn "'apt-get update' falló (posible modo offline). Intentaré instalar desde la caché local."
  fi
  if ! apt-get install -y --no-install-recommends python3 python3-venv python3-pip rsync sqlite3 ffmpeg v4l-utils unzip >/dev/null; then
    die "No se pudieron instalar dependencias (${MISSING_CMDS[*]}). Instalalas manualmente y reintentá."
  fi
else
  log "Dependencias base ya presentes; omitiendo apt-get install."
fi

step "Preparando layout del USB"
if [[ ! -d "${USB_PROJECT_DIR}" ]]; then
  ensure_project_from_zip
fi
[[ -f "${USB_PROJECT_DIR}/app.py" ]] || die "La carpeta project/ no parece contener app.py."
cleanup_macos_trash "${USB_PROJECT_DIR}"
cleanup_macos_trash "${USB_DEPLOY_DIR}"
ensure_usb_model
cleanup_macos_trash "${USB_MODELS_DIR}"

step "Deteniendo servicio previo"
if systemctl list-unit-files | grep -q "^${SERVICE_NAME}\.service"; then
  systemctl stop "${SERVICE_NAME}" || true
  systemctl disable "${SERVICE_NAME}" >/dev/null 2>&1 || true
  systemctl reset-failed "${SERVICE_NAME}" || true
fi

step "Copiando aplicación a ${APP_DIR}"
mkdir -p "${APP_DIR}"
rsync -a --delete \
  --exclude 'data/' \
  --exclude 'logs/' \
  --exclude 'models/' \
  --exclude '.env-claves' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  --exclude '._*' \
  "${USB_PROJECT_DIR}/" "${APP_DIR}/"

if [[ ! -d "${APP_DIR}/data" || -z "$(ls -A "${APP_DIR}/data" 2>/dev/null)" ]]; then
  if [[ -d "${USB_PROJECT_DIR}/data" ]]; then
    log "Sembrando data/ inicial desde el USB…"
    rsync -a "${USB_PROJECT_DIR}/data/" "${APP_DIR}/data/"
  else
    log "USB no trae data/ inicial; creando carpeta vacía en destino"
    mkdir -p "${APP_DIR}/data"
  fi
fi

mkdir -p "${APP_DIR}/logs"

step "Sincronizando deploy/"
rsync -a --delete "${USB_DEPLOY_DIR}/" "${APP_DIR}/deploy/"
chmod +x "${APP_DIR}/deploy/ubuntu"/*.sh

step "Copiando modelo offline"
mkdir -p "${APP_DIR}/models"
install -m 644 "${USB_MODEL}" "${APP_DIR}/models/arcface_r50.onnx"

if [[ ! -f "${APP_DIR}/.env-claves" ]]; then
  if [[ -f "${USB_PROJECT_DIR}/.env-claves" ]]; then
    log "Copiando .env-claves desde el USB…"
    install -m 600 "${USB_PROJECT_DIR}/.env-claves" "${APP_DIR}/.env-claves"
  elif [[ -f "${USB_PROJECT_DIR}/.env-claves.example" ]]; then
    log "Generando .env-claves desde el ejemplo (editá los valores)…"
    install -m 600 "${USB_PROJECT_DIR}/.env-claves.example" "${APP_DIR}/.env-claves"
  fi
fi

step "Permisos y entorno virtual"
chown -R "${APP_USER}:${APP_GROUP}" "${APP_DIR}"
sudo -u "${APP_USER}" -H bash -lc "
  set -euo pipefail
  cd '${APP_DIR}'
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip wheel >/dev/null
  if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt >/dev/null
  fi
"

step "Provisionando modelos"
USB_MODEL_SOURCE="${USB_MODEL}" bash "${APP_DIR}/deploy/ubuntu/20_models.sh"

step "Preflight offline"
if ! bash "${APP_DIR}/deploy/ubuntu/preflight.sh"; then
  die "preflight.sh falló. Corregí el problema y reejecutá: sudo bash ${APP_DIR}/deploy/ubuntu/preflight.sh"
fi

step "Instalando systemd"
bash "${APP_DIR}/deploy/ubuntu/30_systemd.sh"

log "Estado actual del servicio:"
systemctl --no-pager --full status "${SERVICE_NAME}.service" || true

log "=== Instalación completa ==="
log "Comando de diagnóstico: sudo journalctl -u ${SERVICE_NAME} -n 200 --no-pager"
log "Si necesitás reiniciar el servicio: sudo systemctl restart ${SERVICE_NAME}"
