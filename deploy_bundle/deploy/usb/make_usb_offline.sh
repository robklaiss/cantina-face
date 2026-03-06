#!/usr/bin/env bash
# make_usb_offline.sh — Genera el USB sellado (OS-FLEX) listo para uso offline.
set -euo pipefail

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
  warn "El sellado falló en el paso '${CURRENT_STEP:-desconocido}' (línea $1)."
  exit "$exit_code"
}

trap 'on_error $LINENO' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
USB_ROOT="$SCRIPT_DIR/OS-FLEX"
USB_DEPLOY_DIR="$USB_ROOT/deploy"
USB_PROJECT_DIR="$USB_ROOT/project"
USB_MODELS_DIR="$USB_ROOT/models"
DEPLOY_UBUNTU_SRC="$REPO_DIR/deploy/ubuntu"
PROJECT_SRC="$REPO_DIR/project"
MODEL_SOURCE="$REPO_DIR/models/arcface_r50.onnx"
PROJECT_ZIP="$USB_DEPLOY_DIR/project.zip"

REQUIRED_CMDS=(rsync zip)
for cmd in "${REQUIRED_CMDS[@]}"; do
  command -v "$cmd" >/dev/null 2>&1 || die "Comando requerido no disponible: $cmd"
done

cleanup_macos_trash() {
  local target="$1"
  [[ -d "$target" ]] || return 0
  find "$target" -name '._*' -type f -delete 2>/dev/null || true
  find "$target" -name '.DS_Store' -type f -delete 2>/dev/null || true
  find "$target" -type d -name '__MACOSX' -exec rm -rf {} + 2>/dev/null || true
  rm -rf "$target/.Spotlight-V100" "$target/.Trashes" "$target/.fseventsd"
}

copy_deploy_scripts() {
  step "Copiando deploy/ubuntu"
  mkdir -p "$USB_DEPLOY_DIR"
  rsync -a --delete "$DEPLOY_UBUNTU_SRC/" "$USB_DEPLOY_DIR/ubuntu/"
  chmod +x "$USB_DEPLOY_DIR/ubuntu"/*.sh
}

copy_project_tree() {
  step "Copiando project/ base"
  mkdir -p "$USB_PROJECT_DIR"
  local -a excludes=(
    '--exclude=.git'
    '--exclude=.venv/'
    '--exclude=venv/'
    '--exclude=__pycache__/'
    '--exclude=.mypy_cache/'
    '--exclude=.pytest_cache/'
    '--exclude=.DS_Store'
    '--exclude=._*'
    '--exclude=deploy/'
    '--exclude=models/'
  )
  rsync -a --delete "${excludes[@]}" "$PROJECT_SRC/" "$USB_PROJECT_DIR/"
  cleanup_macos_trash "$USB_PROJECT_DIR"
}

copy_model() {
  step "Copiando modelo arcface_r50.onnx"
  mkdir -p "$USB_MODELS_DIR"
  if [[ ! -s "$MODEL_SOURCE" ]]; then
    die "Falta $MODEL_SOURCE. Descargá/copialo localmente antes de sellar el USB."
  fi
  install -m 644 "$MODEL_SOURCE" "$USB_MODELS_DIR/arcface_r50.onnx"
}

build_project_zip() {
  step "Construyendo deploy/project.zip"
  mkdir -p "$USB_DEPLOY_DIR"
  rm -f "$PROJECT_ZIP"
  local stage_dir
  stage_dir="$(mktemp -d /tmp/osflex_zip.XXXXXX)"
  rsync -a --delete "$USB_PROJECT_DIR/" "$stage_dir/"
  if [[ -s "$MODEL_SOURCE" ]]; then
    mkdir -p "$stage_dir/models"
    install -m 644 "$MODEL_SOURCE" "$stage_dir/models/arcface_r50.onnx"
  fi
  pushd "$stage_dir" >/dev/null
  zip -r -q "$PROJECT_ZIP" . -x '**/.DS_Store' '**/._*' '**/__pycache__/*'
  popd >/dev/null
  rm -rf "$stage_dir"
}

write_usb_readme() {
  step "Escribiendo README_USB.md"
  cat > "$USB_ROOT/README_USB.md" <<'EOF'
# Cantina Face — USB OFFLINE (OS-FLEX)

1. Montá el USB. Debe quedar en `/media/$USER/OS-FLEX`.
2. Confirmá que existen estas rutas:
   - `project/`
   - `models/arcface_r50.onnx`
   - `deploy/project.zip`
   - `deploy/ubuntu/install.sh`
3. Ejecutá el instalador en la caja objetivo:

```bash
sudo bash /media/$USER/OS-FLEX/deploy/ubuntu/install.sh
```

4. Diagnóstico del servicio luego de instalar:

```bash
sudo journalctl -u cantina-face -n 200 --no-pager
```

> Re-sellá el USB corriendo `bash deploy/usb/make_usb_offline.sh` dentro del repo.
EOF
}

main() {
  step "Preparando árbol OS-FLEX"
  rm -rf "$USB_ROOT"
  mkdir -p "$USB_ROOT"

  copy_deploy_scripts
  copy_project_tree
  copy_model
  build_project_zip
  write_usb_readme

  cleanup_macos_trash "$USB_ROOT"
  cleanup_macos_trash "$USB_DEPLOY_DIR"
  cleanup_macos_trash "$USB_PROJECT_DIR"
  cleanup_macos_trash "$USB_MODELS_DIR"
  log "USB sellado en $USB_ROOT"
}

main "$@"
