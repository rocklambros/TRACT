#!/usr/bin/env bash
# opencre_import.sh — Import TRACT-generated CSV into a local OpenCRE fork.
#
# Usage: ./scripts/opencre_import.sh <csv_file>
#
# Prerequisites:
#   - OpenCRE fork at ~/github_projects/OpenCRE
#   - Fork initialized: CRE_ALLOW_IMPORT=1 python cre.py --upstream_sync --cache_file cre.db

set -euo pipefail

CSV_FILE="${1:?Usage: $0 <csv_file>}"
OPENCRE_DIR="${HOME}/github_projects/OpenCRE"
PORT=5001
PID_FILE="/tmp/opencre_import_flask.pid"

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

if [ ! -d "$OPENCRE_DIR" ]; then
    echo "Error: OpenCRE fork not found at $OPENCRE_DIR"
    exit 1
fi

cleanup() {
    if [ -f "$PID_FILE" ]; then
        kill "$(cat "$PID_FILE")" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
}
trap cleanup EXIT

echo "Starting OpenCRE Flask app on port $PORT..."
cd "$OPENCRE_DIR"
CRE_ALLOW_IMPORT=1 FLASK_APP=cre.py flask run --port "$PORT" &
echo $! > "$PID_FILE"

echo "Waiting for app to be ready..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:$PORT/rest/v1/root_cres" > /dev/null 2>&1; then
        echo "App ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Error: App did not start within 30 seconds"
        exit 1
    fi
    sleep 1
done

echo "Uploading CSV: $CSV_FILE"
RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST "http://localhost:$PORT/rest/v1/cre_csv_import" \
    -F "cre_csv=@${CSV_FILE}")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "HTTP Status: $HTTP_CODE"
echo "Response: $BODY"

if [ "$HTTP_CODE" -eq 200 ]; then
    echo "Import successful."
else
    echo "Import failed (HTTP $HTTP_CODE)."
    exit 1
fi
