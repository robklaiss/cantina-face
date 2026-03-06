#!/bin/bash
# Sync students from caja to backend
# Run this periodically (e.g., via cron every hour)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$APP_DIR/.env" ]; then
    set -a
    source "$APP_DIR/.env"
    set +a
fi

INTERNAL_TOKEN="${INTERNAL_TOKEN:-cantina-update-secret-2026}"
BACKEND_SYNC_URL="${BACKEND_SYNC_URL:-https://sistema.siloe.com.py}"

echo "[$(date)] Starting student sync to backend..."

# Call local FastAPI endpoint to trigger sync
response=$(curl -s -w "\n%{http_code}" -X POST \
    "http://localhost:8000/api/admin/sync-students" \
    -H "X-Internal-Token: $INTERNAL_TOKEN" \
    -H "Content-Type: application/json" \
    2>&1) || true

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" = "200" ]; then
    echo "[$(date)] ✅ Sync successful: $body"
    exit 0
else
    echo "[$(date)] ❌ Sync failed (HTTP $http_code): $body"
    exit 1
fi
