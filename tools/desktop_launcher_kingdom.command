#!/bin/zsh
set -euo pipefail

APP_DIR="/Users/dominicaleandri/KingDom"
APP_PY="$APP_DIR/app.py"
VENV_ACTIVATE="$APP_DIR/.venv/bin/activate"
LOCK_FILE="/tmp/kingdom_app.lock"
PORT="8050"
LOG_FILE="/tmp/kingdom_desktop.log"

cd "$APP_DIR" || exit 1

unset APCA_API_KEY_ID
unset APCA_API_SECRET_KEY
unset ALPACA_API_KEY
unset ALPACA_KEY_ID
unset ALPACA_SECRET_KEY
unset ALPACA_API_SECRET
unset ALPACA_PAPER
unset ALPACA_BASE_URL
unset APCA_API_BASE_URL
unset ALPACA_DATA_FEED

source "$VENV_ACTIVATE"
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export ORB_HOST=127.0.0.1
export ORB_PORT="$PORT"
export ORB_SINGLE_INSTANCE=1
export ORB_SCAN_WORKERS=8
export ORB_SCAN_WORKERS_MAX=8
export LOKY_MAX_CPU_COUNT=8

existing_pid=""
if [[ -f "$LOCK_FILE" ]]; then
  existing_pid=$(python3 - <<'PY'
from pathlib import Path
import re
text = Path('/tmp/kingdom_app.lock').read_text() if Path('/tmp/kingdom_app.lock').exists() else ''
m = re.search(r'pid=(\d+)', text)
print(m.group(1) if m else '')
PY
)
fi

if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
  echo "Stopping existing KingDom instance (PID $existing_pid)..."
  kill -TERM "$existing_pid" 2>/dev/null || true
  for _ in {1..30}; do
    if ! kill -0 "$existing_pid" 2>/dev/null; then
      break
    fi
    sleep 0.2
  done
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "Existing instance did not exit cleanly; force killing PID $existing_pid"
    kill -KILL "$existing_pid" 2>/dev/null || true
  fi
fi

echo "Starting KingDom from $(pwd)"
echo "Open: http://127.0.0.1:$PORT"
exec python "$APP_PY" 2>&1 | tee "$LOG_FILE"
