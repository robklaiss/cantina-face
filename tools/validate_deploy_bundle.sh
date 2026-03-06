#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# validate_deploy_bundle.sh - Valida la estructura del deploy bundle
# ============================================================================
# Verifica que el bundle en dist/deploy_bundle/ tenga todos los archivos
# necesarios y NO contenga archivos que no deberían estar ahí.
# Exit 1 si falla alguna validación.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUNDLE_DIR="$REPO_ROOT/dist/deploy_bundle"

echo "============================================"
echo "Validating deploy_bundle structure"
echo "============================================"
echo ""
echo "Bundle path: $BUNDLE_DIR"
echo ""

# Verificar que el bundle existe
if [ ! -d "$BUNDLE_DIR" ]; then
    echo "❌ ERROR: Bundle directory does not exist: $BUNDLE_DIR" >&2
    echo "Run 'make deploy-bundle' first to create the bundle." >&2
    exit 1
fi

# Archivos que DEBEN existir
REQUIRED_FILES=(
    "run_update.sh"
    "run_install.sh"
    "project.zip"
    "README_DEPLOY.md"
    "deploy/update.sh"
    "deploy/install.sh"
    "deploy/run.sh"
    "deploy/backup_faces.sh"
    "deploy/guardrails/check_python_hardcode.sh"
    "models/arcface_r50.onnx"
)

# Archivos/directorios que NO deben existir (contaminación)
FORBIDDEN_ITEMS=(
    "app.py"
    "config.py"
    "face_engine.py"
    "venv"
    "static"
    "scripts"
    "data"
    "__pycache__"
    "deploy/usb"
    "project"
)

# Patrones de basura de macOS que NO deben existir
MACOS_JUNK_PATTERNS=(
    "._*"
    ".DS_Store"
    "__MACOSX"
)

ERRORS=0

# Validar archivos requeridos
echo "[1/5] Checking required files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$BUNDLE_DIR/$file" ]; then
        echo "  ❌ Missing: $file" >&2
        ERRORS=$((ERRORS + 1))
    else
        echo "  ✅ Found: $file"
    fi
done

# Validar que NO existan archivos prohibidos
echo ""
echo "[2/5] Checking for forbidden items (contamination)..."
for item in "${FORBIDDEN_ITEMS[@]}"; do
    if [ -e "$BUNDLE_DIR/$item" ]; then
        echo "  ❌ Found forbidden item: $item (should NOT be in bundle root)" >&2
        ERRORS=$((ERRORS + 1))
    else
        echo "  ✅ Clean: $item not found"
    fi
done

# Validar que NO exista basura de macOS
echo ""
echo "[3/5] Checking for macOS junk files..."
JUNK_FOUND=0
for pattern in "${MACOS_JUNK_PATTERNS[@]}"; do
    if find "$BUNDLE_DIR" -name "$pattern" -print -quit 2>/dev/null | grep -q .; then
        echo "  ❌ Found macOS junk: $pattern" >&2
        find "$BUNDLE_DIR" -name "$pattern" -print | head -5 | sed 's/^/     /' >&2
        ERRORS=$((ERRORS + 1))
        JUNK_FOUND=1
    fi
done

if [ $JUNK_FOUND -eq 0 ]; then
    echo "  ✅ No macOS junk files found"
fi

# Validar tamaño de arcface_r50.onnx (debe ser > 1MB)
echo ""
echo "[4/5] Checking model size..."
MODEL_FILE="$BUNDLE_DIR/models/arcface_r50.onnx"
if [ -f "$MODEL_FILE" ]; then
    MODEL_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE" 2>/dev/null || echo "0")
    if [ "$MODEL_SIZE" -lt 1048576 ]; then
        echo "  ❌ arcface_r50.onnx is too small ($MODEL_SIZE bytes) - must be > 1MB" >&2
        ERRORS=$((ERRORS + 1))
    else
        echo "  ✅ arcface_r50.onnx size: $(numfmt --to=iec-i --suffix=B $MODEL_SIZE 2>/dev/null || echo "$MODEL_SIZE bytes")"
    fi
else
    echo "  ❌ arcface_r50.onnx not found (already counted in required files)" >&2
fi

# Validar que project.zip no esté vacío
echo ""
echo "[5/5] Checking project.zip size..."
ZIP_SIZE=$(stat -f%z "$BUNDLE_DIR/project.zip" 2>/dev/null || stat -c%s "$BUNDLE_DIR/project.zip" 2>/dev/null || echo "0")
if [ "$ZIP_SIZE" -lt 10000 ]; then
    echo "  ❌ project.zip is too small ($ZIP_SIZE bytes) - might be empty or corrupt" >&2
    ERRORS=$((ERRORS + 1))
else
    echo "  ✅ project.zip size: $(numfmt --to=iec-i --suffix=B $ZIP_SIZE 2>/dev/null || echo "$ZIP_SIZE bytes")"
fi

# Validar que los runners sean ejecutables
echo ""
echo "[Bonus 1] Checking runner permissions..."
for runner in "run_update.sh" "run_install.sh"; do
    if [ ! -x "$BUNDLE_DIR/$runner" ]; then
        echo "  ⚠️  Warning: $runner is not executable (will fix with chmod +x)" >&2
    else
        echo "  ✅ $runner is executable"
    fi
done

# Resultado final
echo ""
echo "============================================"
if [ $ERRORS -eq 0 ]; then
    echo "✅ Validation PASSED"
    echo "============================================"
    echo ""
    echo "Bundle is ready for USB deployment."
    echo "Next step: make deploy-bundle-usb USB=/Volumes/OS-FLEX"
    exit 0
else
    echo "❌ Validation FAILED with $ERRORS error(s)"
    echo "============================================"
    echo ""
    echo "Fix the errors above and run 'make deploy-bundle' again."
    exit 1
fi
