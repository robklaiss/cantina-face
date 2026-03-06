#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DEPLOY_DIR="${DEPLOY_DIR:-$ROOT_DIR/deploy}"

DEFAULT_TARGETS=(
    "$DEPLOY_DIR"
)

if [ "$#" -gt 0 ]; then
    TARGETS=("$@")
else
    TARGETS=("${DEFAULT_TARGETS[@]}")
fi

MISSING_TARGETS=()
for target in "${TARGETS[@]}"; do
    if [ ! -e "$target" ]; then
        MISSING_TARGETS+=("$target")
    fi
done

if [ ${#MISSING_TARGETS[@]} -gt 0 ]; then
    echo "ERROR: No se encontraron los siguientes paths para el guardrail:" >&2
    printf '  - %s\n' "${MISSING_TARGETS[@]}" >&2
    exit 1
fi

VERSION_PATTERN='python3\.(8|9|10|11|12|13)'
PKG_PATTERN='python3\.[0-9]+-(venv|dev|pip)'
PIP_PATTERN='pip3\.[0-9]+'
COMBINED_PATTERN="(${VERSION_PATTERN}|${PKG_PATTERN}|${PIP_PATTERN})"

run_check_with_rg() {
    rg --line-number --color never \
        --glob '!**/guardrails/**' \
        -e "$COMBINED_PATTERN" "${TARGETS[@]}"
}

run_check_with_grep() {
    grep -R -nE "$COMBINED_PATTERN" "${TARGETS[@]}" \
        --exclude-dir='guardrails'
}

MATCHES=""
if command -v rg >/dev/null 2>&1; then
    MATCHES="$(run_check_with_rg || true)"
else
    MATCHES="$(run_check_with_grep || true)"
fi

if [[ -n "$MATCHES" ]]; then
    cat <<'EOF'
ERROR: Se detectaron hardcodeos de versiones de Python. Usa python3, python3-venv, python3-dev y python3-pip genéricos.
EOF
    echo "$MATCHES"
    exit 1
fi

exit 0
