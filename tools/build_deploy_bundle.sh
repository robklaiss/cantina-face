#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# build_deploy_bundle.sh - Construye el bundle de deploy para USB/noexec
# ============================================================================
# Genera dist/deploy_bundle/ con la estructura correcta para copiar al pen.
# Elimina basura de macOS, valida archivos críticos, y genera README.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$REPO_ROOT/dist"
BUNDLE_DIR="$DIST_DIR/deploy_bundle"
PROJECT_DIR="$REPO_ROOT/project"
DEPLOY_SRC="$REPO_ROOT/deploy_bundle/deploy"

MODELS_SRC="$REPO_ROOT/models"

echo "============================================"
echo "Building deploy_bundle for USB deployment"
echo "============================================"

# Limpiar y crear directorio de salida
if [ -d "$BUNDLE_DIR" ]; then
    echo "[1/7] Limpiando bundle anterior..."
    rm -rf "$BUNDLE_DIR"
fi

mkdir -p "$BUNDLE_DIR"

# Copiar estructura deploy/ (excluyendo legacy usb/)
echo "[2/7] Copiando scripts de deploy..."
if [ ! -d "$DEPLOY_SRC" ]; then
    echo "ERROR: No se encontró $DEPLOY_SRC" >&2
    exit 1
fi

rsync -a "$DEPLOY_SRC/" "$BUNDLE_DIR/deploy/" --exclude='usb/'

# Eliminar rutas legacy que pudieran haber quedado
rm -rf "$BUNDLE_DIR/deploy/usb" 2>/dev/null || true
rm -rf "$BUNDLE_DIR/project" 2>/dev/null || true

# Copiar modelos
echo "[3/7] Copiando modelos..."
mkdir -p "$BUNDLE_DIR/models"
if [ -f "$MODELS_SRC/arcface_r50.onnx" ]; then
    cp "$MODELS_SRC/arcface_r50.onnx" "$BUNDLE_DIR/models/arcface_r50.onnx"
    echo "  arcface_r50.onnx: $(du -h "$BUNDLE_DIR/models/arcface_r50.onnx" | cut -f1)"
else
    echo "ERROR: No se encontró arcface_r50.onnx en $MODELS_SRC" >&2
    echo "El modelo es obligatorio para el bundle." >&2
    exit 1
fi

# Crear project.zip
echo "[4/7] Creando project.zip..."
cd "$PROJECT_DIR"

# Crear zip temporal sin basura de macOS
TEMP_ZIP="$BUNDLE_DIR/project.zip.tmp"
zip -r "$TEMP_ZIP" . \
    -x "venv/*" \
    -x "data/*" \
    -x ".DS_Store" \
    -x "._*" \
    -x "__pycache__/*" \
    -x "*.pyc" \
    -x ".git/*" \
    -q

mv "$TEMP_ZIP" "$BUNDLE_DIR/project.zip"

# Crear runners
echo "[5/7] Creando runners (run_update.sh, run_install.sh)..."

cat > "$BUNDLE_DIR/run_update.sh" <<'RUNNER_UPDATE'
#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_update.sh - Runner para actualización desde USB (noexec-safe)
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
echo "Cantina Face - Actualización desde USB"
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

if [ ! -f "$SCRIPT_DIR/deploy/update.sh" ]; then
    echo "ERROR: No se encontró deploy/update.sh en $SCRIPT_DIR" >&2
    exit 1
fi

# Copiar a disco local (requiere sudo)
echo "[1/3] Copiando bundle a disco local..."
if [ "$EUID" -ne 0 ]; then
    echo "Se requiere sudo para copiar a $LOCAL_DIR"
    sudo mkdir -p "$LOCAL_DIR"
    sudo rsync -a --delete "$SCRIPT_DIR/" "$LOCAL_DIR/"
    sudo chown -R "$USER:$(id -gn)" "$LOCAL_DIR"
else
    mkdir -p "$LOCAL_DIR"
    rsync -a --delete "$SCRIPT_DIR/" "$LOCAL_DIR/"
fi

echo "[2/3] Ejecutando update.sh desde disco local..."
cd "$LOCAL_DIR"

# Ejecutar update.sh con TARGET_APP_DIR
sudo env TARGET_APP_DIR="$TARGET_APP_DIR" bash "$LOCAL_DIR/deploy/update.sh" "$LOCAL_DIR/project.zip"

echo ""
echo "[3/3] Actualización completada"
echo ""
echo "El bundle local está en: $LOCAL_DIR"
echo "La app está en: $TARGET_APP_DIR"
echo "Puedes desconectar el USB de forma segura."
RUNNER_UPDATE

cat > "$BUNDLE_DIR/run_install.sh" <<'RUNNER_INSTALL'
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
RUNNER_INSTALL

chmod +x "$BUNDLE_DIR/run_update.sh"
chmod +x "$BUNDLE_DIR/run_install.sh"

# Limpiar basura de macOS del bundle
echo "[6/7] Limpiando archivos de macOS..."
find "$BUNDLE_DIR" -name '.DS_Store' -delete 2>/dev/null || true
find "$BUNDLE_DIR" -name '._*' -delete 2>/dev/null || true
find "$BUNDLE_DIR" -name '__MACOSX' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# Crear README
echo "[7/7] Generando README_DEPLOY.md..."
cat > "$BUNDLE_DIR/README_DEPLOY.md" <<'README'
# Deploy Bundle - Cantina Face

Bundle de despliegue autocontenido para Ubuntu 24.04. Funciona **incluso con USB montado en noexec** o sistemas de archivos exFAT/FAT sin soporte de symlinks.

## 🚀 Quick Start (Comando Único)

### En la caja Ubuntu:

```bash
# Actualización (recomendado):
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh

# Instalación inicial (primera vez):
bash /media/$USER/OS-FLEX/deploy_bundle/run_install.sh
```

**Eso es todo.** El script copia el bundle a disco local y ejecuta automáticamente.

---

## 📁 Estructura del Bundle

```
deploy_bundle/
├── run_update.sh          # Runner principal para actualización
├── run_install.sh         # Runner principal para instalación inicial
├── project.zip            # Código fuente de la aplicación (app.py, static/, etc.)
├── README_DEPLOY.md       # Este archivo
├── models/                # Modelos de IA (obligatorio)
│   └── arcface_r50.onnx   # Modelo de reconocimiento facial (~166MB)
└── deploy/                # Scripts de deploy
    ├── update.sh          # Script de actualización (llamado por run_update.sh)
    ├── install.sh         # Script de instalación (llamado por run_install.sh)
    ├── run.sh             # Script para iniciar servidor
    ├── backup_faces.sh    # Backup de datos faciales
    ├── guardrails/        # Validaciones pre-deploy
    │   └── check_python_hardcode.sh
    └── ubuntu/            # Scripts específicos de Ubuntu
        └── *.sh, *.service
```

**IMPORTANTE:** El bundle NO contiene `app.py`, `venv/`, ni `static/` en la raíz. Esos archivos están dentro de `project.zip`. Los modelos van SOLO en `models/`.

---

## 🔧 Cómo Funciona (Noexec-Safe)

Los runners (`run_update.sh` y `run_install.sh`) son **noexec-safe**:

1. **Copian TODO el bundle** desde el USB a `/opt/cantina-face-deploy`
2. **Ejecutan los scripts** desde el disco local (evitando noexec)
3. **Crean el venv** en disco local (evitando problemas de symlinks en exFAT)
4. **Permiten desconectar** el USB de forma segura después

**Nunca** intentan ejecutar Python o crear venvs directamente en el USB.

---

## 📋 Preparación del USB (en Mac/Linux de desarrollo)

### Opción 1: Makefile (recomendado)

```bash
# En el repo base:
make deploy-bundle-usb USB=/Volumes/OS-FLEX
```

Esto construye, valida y copia el bundle a `OS-FLEX/deploy_bundle/`.

### Opción 2: Manual

```bash
# Construir bundle
make deploy-bundle

# Validar
make deploy-bundle-validate

# Copiar a USB
rsync -av --delete \
  --exclude '.DS_Store' \
  --exclude '._*' \
  dist/deploy_bundle/ /Volumes/OS-FLEX/deploy_bundle/
```

---

## 🖥️ En la Máquina Ubuntu

### Instalación Inicial (primera vez)

```bash
bash /media/$USER/OS-FLEX/deploy_bundle/run_install.sh
```

Esto:
- Copia el bundle a `/opt/cantina-face-deploy`
- Descomprime `project.zip`
- Crea el venv
- Instala dependencias
- Configura el sistema

### Actualización (versiones posteriores)

```bash
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
```

Esto:
- Copia el bundle a `/opt/cantina-face-deploy`
- Hace backup de la versión actual
- Descomprime el nuevo `project.zip`
- Instala modelos en `/opt/cantina-face/models/`
- Preserva `data/` (DB, índices faciales)
- Reinstala dependencias

---

## ✅ Requisitos

- **Ubuntu 24.04** (o compatible)
- **Python 3** (se instala automáticamente si falta)
- **rsync** (se instala automáticamente si falta)
- **Conexión a internet** (para descargar dependencias Python)

---

## 🐛 Solución de Problemas

### "Permission denied" al ejecutar

```bash
# Ejecutar con bash explícitamente (no requiere +x):
bash /media/$USER/OS-FLEX/deploy_bundle/run_update.sh
```

### USB montado con noexec

✅ **No hay problema.** Los runners detectan esto automáticamente y copian el bundle a `/opt/cantina-face-deploy` antes de ejecutar.

### Sistema de archivos exFAT/FAT

✅ **No hay problema.** El venv se crea en disco local, no en el USB.

### "run_update.sh no existe"

Verifica la estructura en el USB:
```bash
ls -la /media/$USER/OS-FLEX/deploy_bundle/
```

Debe mostrar:
- `run_update.sh`
- `run_install.sh`
- `project.zip`
- `deploy/`

**NO** debe estar anidado como `OS-FLEX/deploy_bundle/deploy_bundle/`.

### "No se encontró project.zip"

```bash
# Verificar que el ZIP existe y no está vacío:
ls -lh /media/$USER/OS-FLEX/deploy_bundle/project.zip
```

Debe tener al menos 100KB.

---

## 🎯 Después de la Instalación

### Iniciar el servidor manualmente:

```bash
bash /opt/cantina-face-deploy/deploy/run.sh
```

### Configurar autostart (opcional):

```bash
bash /opt/cantina-face-deploy/deploy/update.sh --setup-autostart
```

### Ver logs del servicio:

```bash
sudo journalctl -u cantina-face -f
```

---

## 💾 Backups

Los backups se generan automáticamente en `/opt/cantina-face-deploy/deploy/backups/` con formato:

```
cantina-face-20260210-143022.tgz
```

Excluyen: `venv/`, `data/`, archivos temporales de macOS.

---

## 📝 Notas Importantes

- ✅ El bundle local queda en `/opt/cantina-face-deploy`
- ✅ Los datos faciales (DB, índices) se preservan entre actualizaciones
- ✅ El USB puede desconectarse después de la instalación/actualización
- ✅ Los runners siempre usan `bash` explícitamente (no dependen de shebang)
- ✅ No se crea basura de macOS (`.DS_Store`, `._*`, `__MACOSX`)

---

## 🆘 Soporte

Para más detalles, consulta:
- `deploy/README.md`: documentación del paquete deploy
- `deploy/ubuntu/README.md`: instalación específica Ubuntu
- `README.md`: documentación principal del proyecto

---

**Generado por:** `tools/build_deploy_bundle.sh`  
**Validado por:** `tools/validate_deploy_bundle.sh`
README

# Validar que existen todos los archivos críticos
echo ""
echo "Validando estructura del bundle..."

REQUIRED_FILES=(
    "$BUNDLE_DIR/run_update.sh"
    "$BUNDLE_DIR/run_install.sh"
    "$BUNDLE_DIR/project.zip"
    "$BUNDLE_DIR/README_DEPLOY.md"
    "$BUNDLE_DIR/deploy/update.sh"
    "$BUNDLE_DIR/deploy/install.sh"
    "$BUNDLE_DIR/deploy/run.sh"
    "$BUNDLE_DIR/deploy/backup_faces.sh"
    "$BUNDLE_DIR/deploy/guardrails/check_python_hardcode.sh"
    "$BUNDLE_DIR/models/arcface_r50.onnx"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo ""
    echo "❌ ERROR: Faltan archivos críticos en el bundle:" >&2
    printf '   - %s\n' "${MISSING_FILES[@]}" >&2
    exit 1
fi

# Mostrar resumen
echo ""
echo "✅ Bundle creado exitosamente en: $BUNDLE_DIR"
echo ""
echo "Estructura generada:"
echo "-------------------"
cd "$BUNDLE_DIR"
find . -type f -o -type d | head -30 | sort
echo ""
echo "Tamaño del project.zip: $(du -h project.zip | cut -f1)"
echo "Tamaño del modelo: $(du -h models/arcface_r50.onnx | cut -f1)"
echo ""
echo "Siguiente paso:"
echo "  Copiar $BUNDLE_DIR/ al USB en OS-FLEX/deploy_bundle/"
echo ""
