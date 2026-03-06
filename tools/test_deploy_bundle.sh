#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# test_deploy_bundle.sh - Test completo del bundle de deploy
# ============================================================================
# Simula el flujo completo: build -> validate -> simulated USB copy
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_USB_DIR="$REPO_ROOT/.test-usb-simulation"

echo "============================================"
echo "Testing Deploy Bundle Workflow"
echo "============================================"
echo ""

# Limpiar test anterior
if [ -d "$TEST_USB_DIR" ]; then
    echo "[Cleanup] Removing previous test USB simulation..."
    rm -rf "$TEST_USB_DIR"
fi

# Step 1: Build
echo "[Step 1/6] Building deploy bundle..."
bash "$SCRIPT_DIR/build_deploy_bundle.sh"
echo ""

# Step 2: Validate
echo "[Step 2/6] Validating bundle structure..."
bash "$SCRIPT_DIR/validate_deploy_bundle.sh"
echo ""

# Step 3: Simulate USB copy
echo "[Step 3/6] Simulating USB copy..."
mkdir -p "$TEST_USB_DIR/deploy_bundle"
rsync -av --delete \
    --exclude '.DS_Store' \
    --exclude '._*' \
    --exclude '__MACOSX' \
    "$REPO_ROOT/dist/deploy_bundle/" "$TEST_USB_DIR/deploy_bundle/"

echo ""
echo "Simulated USB structure:"
find "$TEST_USB_DIR" -type f -o -type d | sort
echo ""

# Step 4: Verify USB structure
echo "[Step 4/6] Verifying simulated USB structure..."

REQUIRED_USB_FILES=(
    "$TEST_USB_DIR/deploy_bundle/run_update.sh"
    "$TEST_USB_DIR/deploy_bundle/run_install.sh"
    "$TEST_USB_DIR/deploy_bundle/project.zip"
    "$TEST_USB_DIR/deploy_bundle/README_DEPLOY.md"
    "$TEST_USB_DIR/deploy_bundle/deploy/update.sh"
    "$TEST_USB_DIR/deploy_bundle/models/arcface_r50.onnx"
)

ERRORS=0
for file in "${REQUIRED_USB_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ❌ Missing in USB: $file" >&2
        ERRORS=$((ERRORS + 1))
    else
        echo "  ✅ Found in USB: $(basename $file)"
    fi
done

# Verify model size > 1MB
MODEL_IN_USB="$TEST_USB_DIR/deploy_bundle/models/arcface_r50.onnx"
if [ -f "$MODEL_IN_USB" ]; then
    MODEL_SIZE=$(stat -f%z "$MODEL_IN_USB" 2>/dev/null || stat -c%s "$MODEL_IN_USB" 2>/dev/null || echo "0")
    if [ "$MODEL_SIZE" -lt 1048576 ]; then
        echo "  ❌ arcface_r50.onnx too small ($MODEL_SIZE bytes)" >&2
        ERRORS=$((ERRORS + 1))
    else
        echo "  ✅ arcface_r50.onnx size OK ($MODEL_SIZE bytes)"
    fi
fi

# Verify no contamination
FORBIDDEN_USB_ITEMS=(
    "$TEST_USB_DIR/deploy_bundle/app.py"
    "$TEST_USB_DIR/deploy_bundle/venv"
    "$TEST_USB_DIR/deploy_bundle/static"
    "$TEST_USB_DIR/deploy_bundle/deploy/usb"
    "$TEST_USB_DIR/deploy_bundle/project"
)

for item in "${FORBIDDEN_USB_ITEMS[@]}"; do
    if [ -e "$item" ]; then
        echo "  ❌ Found forbidden item in USB: $item" >&2
        ERRORS=$((ERRORS + 1))
    else
        echo "  ✅ Clean: $(basename $item) not in bundle"
    fi
done

# Check for macOS junk
if find "$TEST_USB_DIR" -name '.DS_Store' -o -name '._*' -o -name '__MACOSX' | grep -q .; then
    echo "  ❌ Found macOS junk files in USB simulation" >&2
    ERRORS=$((ERRORS + 1))
else
    echo "  ✅ No macOS junk in USB simulation"
fi

# Step 5: Verify update.sh recognizes TARGET_APP_DIR and models
echo ""
echo "[Step 5/6] Verifying update.sh supports TARGET_APP_DIR and models..."
UPDATE_SH="$TEST_USB_DIR/deploy_bundle/deploy/update.sh"
if [ -f "$UPDATE_SH" ]; then
    if grep -q 'TARGET_APP_DIR' "$UPDATE_SH"; then
        echo "  ✅ update.sh references TARGET_APP_DIR"
    else
        echo "  ❌ update.sh does NOT reference TARGET_APP_DIR" >&2
        ERRORS=$((ERRORS + 1))
    fi
    if grep -q 'MODELS_SRC' "$UPDATE_SH" && grep -q 'MODELS_DST' "$UPDATE_SH"; then
        echo "  ✅ update.sh has MODELS_SRC/MODELS_DST variables"
    else
        echo "  ❌ update.sh missing MODELS_SRC/MODELS_DST" >&2
        ERRORS=$((ERRORS + 1))
    fi
    if grep -q 'arcface_r50.onnx' "$UPDATE_SH"; then
        echo "  ✅ update.sh installs arcface_r50.onnx"
    else
        echo "  ❌ update.sh does NOT install arcface_r50.onnx" >&2
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  ❌ update.sh not found in USB simulation" >&2
    ERRORS=$((ERRORS + 1))
fi

# Step 6: Verify run_update.sh passes TARGET_APP_DIR
echo ""
echo "[Step 6/6] Verifying runners pass TARGET_APP_DIR..."
for runner in run_update.sh run_install.sh; do
    RUNNER_FILE="$TEST_USB_DIR/deploy_bundle/$runner"
    if [ -f "$RUNNER_FILE" ]; then
        if grep -q 'TARGET_APP_DIR' "$RUNNER_FILE"; then
            echo "  ✅ $runner passes TARGET_APP_DIR"
        else
            echo "  ❌ $runner does NOT pass TARGET_APP_DIR" >&2
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo "  ❌ $runner not found" >&2
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
echo "============================================"
if [ $ERRORS -eq 0 ]; then
    echo "✅ ALL TESTS PASSED"
    echo "============================================"
    echo ""
    echo "Bundle is ready for real USB deployment."
    echo ""
    echo "Next steps:"
    echo "  1. Insert USB (e.g., /Volumes/OS-FLEX)"
    echo "  2. Run: make deploy-bundle-usb USB=/Volumes/OS-FLEX"
    echo "  3. In Ubuntu: bash /media/\$USER/OS-FLEX/deploy_bundle/run_update.sh"
    echo ""
    echo "Simulated USB at: $TEST_USB_DIR"
    exit 0
else
    echo "❌ TESTS FAILED with $ERRORS error(s)"
    echo "============================================"
    exit 1
fi
