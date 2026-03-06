#!/bin/bash
set -euo pipefail

# Build script for HostGator deployment package
# Creates deploy-gator-dist/ with production-ready structure

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${REPO_ROOT}/deploy-gator-dist"
SOURCE_PHP="${REPO_ROOT}/server_php"
SOURCE_STATIC="${REPO_ROOT}/deploy-gator"

echo "=== Building HostGator Deployment Package ==="
echo ""

# Clean and create dist directory
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

echo "[1/5] Copying API endpoints..."
cp -r "${SOURCE_PHP}/api" "${DIST_DIR}/api"

echo "[2/5] Copying App core..."
cp -r "${SOURCE_PHP}/app" "${DIST_DIR}/app"
for schema_file in schema.sql db_schema.php; do
    if [ -f "${SOURCE_PHP}/app/${schema_file}" ]; then
        cp "${SOURCE_PHP}/app/${schema_file}" "${DIST_DIR}/app/${schema_file}"
    fi
done

# Remove config.php if it exists (user must create their own)
# Keep config.example.php as fallback/default
if [ -f "${DIST_DIR}/app/config.php" ]; then
    echo "      (removing existing config.php - user must create their own from config.example.php)"
    rm "${DIST_DIR}/app/config.php"
fi

# Ensure config.example.php exists (required for fallback)
if [ ! -f "${DIST_DIR}/app/config.example.php" ]; then
    echo "      (copying config.example.php as fallback)"
    cp "${SOURCE_PHP}/app/config.example.php" "${DIST_DIR}/app/config.example.php"
fi

echo "[3/5] Copying static sites..."
cp -r "${SOURCE_STATIC}/padres" "${DIST_DIR}/padres"
cp -r "${SOURCE_STATIC}/backend" "${DIST_DIR}/backend"
cp -r "${SOURCE_STATIC}/updates" "${DIST_DIR}/updates"
cp "${SOURCE_STATIC}/.htaccess" "${DIST_DIR}/.htaccess"

echo "[3.5/5] Copying PHP CLI tools and seed data..."
cp -r "${SOURCE_PHP}/tools" "${DIST_DIR}/tools"
if [ -f "${SOURCE_PHP}/data/db.sqlite" ]; then
    mkdir -p "${DIST_DIR}/data"
    cp "${SOURCE_PHP}/data/db.sqlite" "${DIST_DIR}/data/db.sqlite"
fi

echo "[4/5] Creating required directories..."
mkdir -p "${DIST_DIR}/data"
mkdir -p "${DIST_DIR}/storage/logs"

echo "[5/5] Verifying structure..."
echo ""
echo "Final structure:"
ls -la "${DIST_DIR}"
echo ""

# Verification
echo "=== Verification ==="

ERRORS=0

# Check required directories
dirs=("api" "app" "padres" "backend" "updates")
for dir in "${dirs[@]}"; do
    if [ -d "${DIST_DIR}/${dir}" ]; then
        echo "✓ Directory exists: ${dir}/"
    else
        echo "✗ MISSING directory: ${dir}/"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check service-worker.js in padres folder
if [ -f "${DIST_DIR}/padres/service-worker.js" ]; then
    echo "✓ File exists: padres/service-worker.js"
else
    echo "✗ MISSING file: padres/service-worker.js (required for PWA)"
    ERRORS=$((ERRORS + 1))
fi

# Check app.js has correct relative SW registration (not hardcoded /parents/)
echo ""
echo "Checking SW registration in padres/app.js..."
if [ -f "${DIST_DIR}/padres/app.js" ]; then
    if grep -q "detectPortalBasePath" "${DIST_DIR}/padres/app.js"; then
        echo "  ✓ app.js uses detectPortalBasePath() for relative SW registration"
    else
        echo "  ✗ app.js missing detectPortalBasePath() - SW registration may be hardcoded"
        ERRORS=$((ERRORS + 1))
    fi
    if grep -q 'register.*service-worker.js' "${DIST_DIR}/padres/app.js"; then
        echo "  ✓ app.js has service worker registration"
    else
        echo "  ✗ app.js missing service worker registration"
        ERRORS=$((ERRORS + 1))
    fi
    # Verify NO hardcoded /parents/ or /padres/ in register call
    if grep -q "register.*/parents/service-worker" "${DIST_DIR}/padres/app.js" || \
       grep -q "register.*/padres/service-worker" "${DIST_DIR}/padres/app.js"; then
        echo "  ✗ app.js has hardcoded path in SW registration - use detectPortalBasePath()"
        ERRORS=$((ERRORS + 1))
    else
        echo "  ✓ No hardcoded paths in SW registration"
    fi
else
    echo "✗ padres/app.js missing"
    ERRORS=$((ERRORS + 1))
fi

# Check .htaccess
if [ -f "${DIST_DIR}/.htaccess" ]; then
    echo "✓ File exists: .htaccess"
else
    echo "✗ MISSING file: .htaccess"
    ERRORS=$((ERRORS + 1))
fi

# Check all API files use correct bootstrap path
# NOTE: health.php uses bootstrap_core.php, others use bootstrap.php

check_bootstrap_core_path() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "  ✗ $(basename "$file") missing"
        ERRORS=$((ERRORS + 1))
        return
    fi

    if grep -qE "require_once\\s+__DIR__\\s*\\.\\s*'/\\.\\./app/bootstrap_core\\.php';" "$file"; then
        echo "  ✓ $(basename "$file") has correct bootstrap_core.php path"
    else
        echo "  ✗ $(basename "$file") missing correct bootstrap_core.php path"
        ERRORS=$((ERRORS + 1))
    fi
}

echo ""
echo "Checking API bootstrap paths..."
check_bootstrap_core_path "${DIST_DIR}/api/dev_create_user.php"
check_bootstrap_core_path "${DIST_DIR}/api/dev_dump_users.php"
check_bootstrap_core_path "${DIST_DIR}/api/dev_create_student.php"
check_bootstrap_core_path "${DIST_DIR}/api/dev_link_parent_student.php"

for api_file in "${DIST_DIR}/api/"*.php; do
    if [ -f "$api_file" ]; then
        fname=$(basename "$api_file")
        case "$fname" in
            dev_create_user.php|dev_dump_users.php)
                continue
                ;;
            health.php|dev_*.php)
                if grep -q "require_once __DIR__ . '/../app/bootstrap_core.php';" "$api_file"; then
                    echo "  ✓ ${fname} has correct bootstrap_core.php path"
                else
                    echo "  ✗ ${fname} missing correct bootstrap_core.php path"
                    ERRORS=$((ERRORS + 1))
                fi
                ;;
            *)
                if grep -q "require_once __DIR__ . '/../app/bootstrap.php';" "$api_file"; then
                    echo "  ✓ ${fname} has correct bootstrap path"
                else
                    echo "  ✗ ${fname} missing correct bootstrap path"
                    ERRORS=$((ERRORS + 1))
                fi
                ;;
        esac
    fi
done

# Check bootstrap_core.php exists and has required components
echo ""
echo "Checking bootstrap_core.php components..."
if [ -f "${DIST_DIR}/app/bootstrap_core.php" ]; then
    if grep -q "config.example.php" "${DIST_DIR}/app/bootstrap_core.php"; then
        echo "  ✓ bootstrap_core.php supports config.example.php fallback"
    else
        echo "  ✗ bootstrap_core.php missing config.example.php fallback support"
        ERRORS=$((ERRORS + 1))
    fi
    if grep -q "bootstrap_core_json_error" "${DIST_DIR}/app/bootstrap_core.php"; then
        echo "  ✓ bootstrap_core.php has JSON error handler"
    else
        echo "  ✗ bootstrap_core.php missing JSON error handler"
        ERRORS=$((ERRORS + 1))
    fi
    if grep -q "function db()" "${DIST_DIR}/app/bootstrap_core.php"; then
        echo "  ✓ bootstrap_core.php has db() helper"
    else
        echo "  ✗ bootstrap_core.php missing db() helper"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "✗ bootstrap_core.php missing - REQUIRED for health.php"
    ERRORS=$((ERRORS + 1))
fi

# Check bootstrap.php exists and includes bootstrap_core.php
if [ -f "${DIST_DIR}/app/bootstrap.php" ]; then
    echo ""
    echo "Checking bootstrap.php..."
    if grep -q "bootstrap_core.php" "${DIST_DIR}/app/bootstrap.php"; then
        echo "  ✓ bootstrap.php includes bootstrap_core.php"
    else
        echo "  ✗ bootstrap.php missing bootstrap_core.php include"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "✗ bootstrap.php missing"
    ERRORS=$((ERRORS + 1))
fi

# Check response.php exists
if [ -f "${DIST_DIR}/app/response.php" ]; then
    echo ""
    echo "Checking response.php..."
    if grep -q "function json_response" "${DIST_DIR}/app/response.php"; then
        echo "  ✓ Has json_response()"
    else
        echo "  ✗ Missing json_response()"
        ERRORS=$((ERRORS + 1))
    fi
    if grep -q "function json_error" "${DIST_DIR}/app/response.php"; then
        echo "  ✓ Has json_error()"
    else
        echo "  ✗ Missing json_error()"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "✗ response.php missing"
    ERRORS=$((ERRORS + 1))
fi

# Check health.php exists
if [ -f "${DIST_DIR}/api/health.php" ]; then
    echo ""
    echo "Checking health.php..."
    if grep -q "'ok' => true" "${DIST_DIR}/api/health.php"; then
        echo "  ✓ Has ok field"
    else
        echo "  ✗ Missing ok field"
        ERRORS=$((ERRORS + 1))
    fi
    if grep -q "'db_ok' =>" "${DIST_DIR}/api/health.php"; then
        echo "  ✓ Has db_ok field"
    else
        echo "  ✗ Missing db_ok field"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "✗ health.php missing"
    ERRORS=$((ERRORS + 1))
fi

# Check config.example.php exists (required for fallback)
if [ -f "${DIST_DIR}/app/config.example.php" ]; then
    echo ""
    echo "Checking config.example.php..."
    echo "  ✓ config.example.php present (fallback config)"
else
    echo ""
    echo "✗ config.example.php missing - required for fallback"
    ERRORS=$((ERRORS + 1))
fi

# DB schema checks
echo ""
echo "Checking DB schema files..."
schema_files=("schema.sql" "db_schema.php")
for schema_file in "${schema_files[@]}"; do
    target="${DIST_DIR}/app/${schema_file}"
    if [ -f "$target" ]; then
        echo "  ✓ ${schema_file} present"
    else
        echo "  ✗ ${schema_file} missing (required for database provisioning)"
        ERRORS=$((ERRORS + 1))
    fi
done

# Remove unwanted files
rm -f "${DIST_DIR}/app/db.php" 2>/dev/null || true

# Final verification output
echo ""
echo "=== Final Verification ==="
echo ""
echo "Service worker file:"
ls -la "${DIST_DIR}/padres/" | grep service-worker.js || echo "  ✗ service-worker.js not found"
echo ""
echo "SW registration in app.js:"
grep -n "register.*service-worker" "${DIST_DIR}/padres/app.js" || echo "  ✗ No register call found"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo "=== BUILD SUCCESS ==="
    echo ""
    echo "To create the ZIP file for upload, run:"
    echo "  cd ${DIST_DIR} && zip -r ../deploy-gator-hostgator.zip ."
    echo ""
    exit 0
else
    echo "=== BUILD FAILED: ${ERRORS} ERRORS ==="
    exit 1
fi
