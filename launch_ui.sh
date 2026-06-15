#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CAPSEL Training UI — one-command launcher
# Usage:  ./launch_ui.sh
#         bash launch_ui.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

PORT=7860
URL="http://localhost:${PORT}"

echo ""
echo "══════════════════════════════════════════════════"
echo "   CAPSEL Training UI"
echo "══════════════════════════════════════════════════"

# ── 1. Install Python deps if needed ─────────────────────────────────────────
echo "» Checking dependencies..."
pip install fastapi uvicorn pyyaml --quiet --break-system-packages 2>/dev/null || \
pip install fastapi uvicorn pyyaml --quiet 2>/dev/null || true

# ── 2. Kill any stale server on the same port ────────────────────────────────
if lsof -ti tcp:${PORT} &>/dev/null; then
    echo "» Stopping old server on port ${PORT}..."
    lsof -ti tcp:${PORT} | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# ── 3. Start FastAPI server in background ────────────────────────────────────
echo "» Starting server on ${URL} ..."
python app_server.py &
SERVER_PID=$!

# ── 4. Wait until server is ready (up to 15 s) ───────────────────────────────
echo -n "» Waiting for server"
for i in $(seq 1 15); do
    if curl -s "${URL}/api/health" &>/dev/null; then
        echo " ready!"
        break
    fi
    echo -n "."
    sleep 1
done

# ── 5. Open browser ──────────────────────────────────────────────────────────
echo "» Opening ${URL} ..."
case "$(uname -s)" in
    Darwin)  open "${URL}" ;;
    Linux)   xdg-open "${URL}" 2>/dev/null || echo "   → Open ${URL} manually" ;;
    MINGW*|CYGWIN*) start "${URL}" ;;
    *)       echo "   → Open ${URL} in your browser" ;;
esac

echo ""
echo "  Server PID : ${SERVER_PID}"
echo "  UI         : ${URL}"
echo "  Logs dir   : $(pwd)/results/"
echo ""
echo "  Press Ctrl-C to stop the server."
echo "══════════════════════════════════════════════════"

# ── 6. Wait for Ctrl-C then clean up ─────────────────────────────────────────
trap "echo ''; echo '» Stopping server...'; kill ${SERVER_PID} 2>/dev/null; exit 0" SIGINT SIGTERM
wait ${SERVER_PID}
