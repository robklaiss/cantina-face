#!/usr/bin/env bash
set -euo pipefail

LOGIN_URL="${1:-${LOGIN_URL:-http://localhost:8000/login.html}}"

if [[ -z "${LOGIN_URL}" ]]; then
    LOGIN_URL="http://localhost:8000/login.html"
fi

launch_chromium_like() {
    local binary="$1"
    shift
    if command -v "$binary" >/dev/null 2>&1; then
        "$binary" \
            --app="${LOGIN_URL}" \
            --kiosk \
            --start-fullscreen \
            --incognito \
            --noerrdialogs \
            --disable-infobars \
            --disable-session-crashed-bubble \
            --overscroll-history-navigation=0 \
            "$@" \
            >/dev/null 2>&1 &
        exit 0
    fi
}

launch_firefox() {
    if command -v firefox >/dev/null 2>&1; then
        firefox --kiosk "${LOGIN_URL}" >/dev/null 2>&1 &
        exit 0
    fi
}

launch_chromium_like chromium-browser
launch_chromium_like chromium
launch_chromium_like google-chrome
launch_chromium_like google-chrome-stable
launch_chromium_like brave-browser
launch_chromium_like microsoft-edge
launch_firefox

if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${LOGIN_URL}" >/dev/null 2>&1 &
    exit 0
fi

echo "[kiosk] No se encontró un navegador compatible para abrir ${LOGIN_URL}" >&2
exit 1
