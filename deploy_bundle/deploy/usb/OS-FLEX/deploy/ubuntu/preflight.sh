#!/usr/bin/env bash
# preflight.sh — Validaciones y migraciones antes de iniciar el servicio
# Ejecutado como ExecStartPre por systemd.
# Idempotente: puede ejecutarse múltiples veces sin romper nada.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

log "[preflight] Iniciando verificaciones..."

# ─── 1. Verificar cámara ────────────────────────────────────────────────────
VIDEO_NODE="${VIDEO_DEVICE:-/dev/video0}"

# No reiniciar GDM ni bloquear la pantalla: sólo verificaciones suaves
if ! wait_for_camera_device; then
    if camera_is_optional; then
        warn "[preflight] Cámara no disponible pero CAMERA_OPTIONAL=1. Continuando."
    else
        warn "[preflight] No se encontró $VIDEO_NODE tras ${CAMERA_WAIT_SECONDS}s"
        exit 1
    fi
elif [ ! -r "$VIDEO_NODE" ]; then
    if camera_is_optional; then
        warn "[preflight] Sin permisos de lectura sobre $VIDEO_NODE, pero cámara opcional."
    else
        warn "[preflight] No hay permisos para leer $VIDEO_NODE. Agregá el usuario al grupo video."
        exit 1
    fi
fi

# ─── 2. Verificar modelos ───────────────────────────────────────────────────
MODEL_ARC="$CANTINA_DIR/models/arcface_r50.onnx"
MODEL_LINK="$CANTINA_DIR/models/mobile_face.onnx"

if [ ! -f "$MODEL_ARC" ]; then
    warn "[preflight] Falta el modelo arcface_r50.onnx en $MODEL_ARC"
    exit 1
fi

# Recrear symlink si falta (auto-heal)
if [ ! -e "$MODEL_LINK" ]; then
    log "[preflight] Recreando symlink mobile_face.onnx -> arcface_r50.onnx"
    rm -f "$MODEL_LINK"
    ln -s arcface_r50.onnx "$MODEL_LINK"
fi

# ─── 3. Migración DB: "transaction" → "transactions" ────────────────────────
DB_FILE="$CANTINA_DIR/data/db.sqlite"

if [ -f "$DB_FILE" ]; then
    log "[preflight] Verificando migración de base de datos..."

    # Comprobar si existe la tabla "transaction" (nombre viejo, palabra reservada SQLite)
    OLD_TABLE_EXISTS=$(sqlite3 "$DB_FILE" \
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='transaction';" 2>/dev/null || echo "0")

    if [ "$OLD_TABLE_EXISTS" = "1" ]; then
        # Backup con timestamp
        BACKUP_FILE="$CANTINA_DIR/data/db.sqlite.bak_$(date '+%Y%m%d_%H%M%S')"
        log "[preflight] Tabla 'transaction' encontrada (palabra reservada). Migrando..."
        log "[preflight] Backup: $BACKUP_FILE"
        cp "$DB_FILE" "$BACKUP_FILE"

        # Renombrar tabla
        sqlite3 "$DB_FILE" 'ALTER TABLE "transaction" RENAME TO transactions;'
        log "[preflight] Tabla renombrada: transaction → transactions"
    fi

    # Crear índice sobre transactions (idempotente)
    NEW_TABLE_EXISTS=$(sqlite3 "$DB_FILE" \
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='transactions';" 2>/dev/null || echo "0")

    if [ "$NEW_TABLE_EXISTS" = "1" ]; then
        sqlite3 "$DB_FILE" "CREATE INDEX IF NOT EXISTS idx_transactions_student ON transactions (student_id);"
        sqlite3 "$DB_FILE" "CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions (created_at);"
        sqlite3 "$DB_FILE" "CREATE INDEX IF NOT EXISTS idx_transactions_pos ON transactions (point_of_sale_id);"
        log "[preflight] Índices de transactions verificados/creados"
    fi
else
    log "[preflight] DB no existe aún ($DB_FILE), se creará al iniciar la app"
fi

log "[preflight] OK"
