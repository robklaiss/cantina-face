#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
DB_PATH="$DATA_DIR/db.sqlite"
INDEX_PATH="$DATA_DIR/index.bin"
INDEX_LABELS_PATH="$DATA_DIR/index_labels.json"
FACES_DIR="$DATA_DIR/faces"
BACKUP_DIR="${BACKUP_DIR:-$DATA_DIR/backups}"
BACKUP_PREFIX="${BACKUP_PREFIX:-db-backup}"
BACKUP_COUNT="${BACKUP_COUNT:-3}"

cleanup() {
    if [ -n "${STAGING_DIR:-}" ] && [ -d "$STAGING_DIR" ]; then
        rm -rf "$STAGING_DIR"
    fi
}
trap cleanup EXIT

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

fatal() {
    echo "[backup_faces] $1" >&2
    exit 1
}

if ! [[ "$BACKUP_COUNT" =~ ^[0-9]+$ ]] || [ "$BACKUP_COUNT" -lt 2 ]; then
    fatal "BACKUP_COUNT debe ser un entero >= 2 (valor actual: $BACKUP_COUNT)"
fi

if [ ! -d "$DATA_DIR" ]; then
    fatal "No se encontró DATA_DIR en $DATA_DIR"
fi

mkdir -p "$BACKUP_DIR"
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/face-db-backup.XXXXXX")"
TIMESTAMP="$(date +%Y-%m-%dT%H:%M:%S%z)"

# Backup SQLite database (usa sqlite3 para snapshot consistente si está disponible)
if [ -f "$DB_PATH" ]; then
    if have_cmd sqlite3; then
        sqlite3 "$DB_PATH" ".backup '$STAGING_DIR/db.sqlite'"
    else
        echo "[backup_faces] Advertencia: sqlite3 no disponible, copiando archivo directamente" >&2
        cp -a "$DB_PATH" "$STAGING_DIR/db.sqlite"
    fi
else
    echo "[backup_faces] Advertencia: no se encontró $DB_PATH" >&2
fi

# Copia índice HNSW si existe
if [ -f "$INDEX_PATH" ]; then
    cp -a "$INDEX_PATH" "$STAGING_DIR/index.bin"
fi

if [ -f "$INDEX_LABELS_PATH" ]; then
    cp -a "$INDEX_LABELS_PATH" "$STAGING_DIR/index_labels.json"
fi

# Copia miniaturas de rostros (si rsync está disponible se usa para preservar permisos)
if [ -d "$FACES_DIR" ]; then
    mkdir -p "$STAGING_DIR/faces"
    if have_cmd rsync; then
        rsync -a "$FACES_DIR/" "$STAGING_DIR/faces/"
    else
        echo "[backup_faces] Advertencia: rsync no disponible, usando cp -a" >&2
        cp -a "$FACES_DIR"/* "$STAGING_DIR/faces/" 2>/dev/null || true
    fi
fi

cat > "$STAGING_DIR/backup-info.txt" <<EOF
Cantina Face backup
Timestamp: $TIMESTAMP
Fuente: $DATA_DIR
Archivos incluidos: db.sqlite, index.bin, index_labels.json, faces/
EOF

slot_path() {
    local slot="$1"
    printf '%s/%s-%02d' "$BACKUP_DIR" "$BACKUP_PREFIX" "$slot"
}

max_slot="$BACKUP_COUNT"
oldest="$(slot_path "$max_slot")"
if [ -d "$oldest" ]; then
    rm -rf "$oldest"
fi

for ((slot=max_slot-1; slot>=1; slot--)); do
    src="$(slot_path "$slot")"
    if [ -d "$src" ]; then
        dest="$(slot_path $((slot+1)))"
        mv "$src" "$dest"
    fi
done

TARGET="$(slot_path 1)"
mv "$STAGING_DIR" "$TARGET"
STAGING_DIR=""

echo "[backup_faces] Backup completado en $TARGET"
