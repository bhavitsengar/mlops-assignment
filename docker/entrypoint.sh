#!/usr/bin/env sh
set -e

# Defaults (override with env vars)
: "${APP_HOST:=0.0.0.0}"
: "${APP_PORT:=8000}"
: "${SQLITE_WEB_PORT:=8080}"
: "${LOG_DB:=/app/logs.db}"   # adjust if your code uses another path

echo "Launching sqlite-web on ${SQLITE_WEB_PORT}, DB=${LOG_DB}"
# Note: add --read-only if you don’t want edits in the UI
sqlite_web "${LOG_DB}" --host 0.0.0.0 --port "${SQLITE_WEB_PORT}" &

echo "Launching FastAPI on ${APP_PORT}"
# Keep uvicorn in the foreground so container lifecycle follows the API
exec uvicorn src.app:app --host "${APP_HOST}" --port "${APP_PORT}"