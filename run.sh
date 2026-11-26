#!/usr/bin/env bash
# Start the reddit-alpha dashboard (localhost only).
#   ./run.sh            # http://127.0.0.1:8792
#   PORT=8899 ./run.sh  # elsewhere
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8792}"
HOST="${HOST:-127.0.0.1}"

# prefer the workspace virtualenv: it already has fastapi/uvicorn/numpy/model2vec
PY="../.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

echo "dashboard -> http://$HOST:$PORT"
echo "store     -> $(pwd)/data/reddit.db"
exec "$PY" -m uvicorn dashboard:app --host "$HOST" --port "$PORT" --log-level info
