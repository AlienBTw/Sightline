#!/usr/bin/env bash
# Throwaway test backend for the camera rig (never reads .env; login exists only in this process).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK="${SIGHTLINE_WORK_DIR:-$REPO/work}"
cd "$REPO"
W="$WORK/camsim/backend_test"; mkdir -p "$W"; chmod 700 "$W"
PY="${SIGHTLINE_PY:-python}"
PASS=$("$PY" -c "import secrets;print(secrets.token_urlsafe(18))")
printf 'CAMTEST_USER=camtest_admin\nCAMTEST_PASS=%s\n' "$PASS" > "$W/creds"; chmod 600 "$W/creds"
export SIGHTLINE_NO_DOTENV=1 SIGHTLINE_DEV=0
export SIGHTLINE_JWT_SECRET=$("$PY" -c "import secrets;print(secrets.token_urlsafe(48))")
export SIGHTLINE_ADMIN_USER=camtest_admin
export SIGHTLINE_ADMIN_PASSWORD_HASH=$(PASS="$PASS" "$PY" -c "import os;from backend.security import hash_password;print(hash_password(os.environ['PASS']))")
export SIGHTLINE_SEED_DEMO_CAMERA=0 SIGHTLINE_AUTOSTART=0 SIGHTLINE_PUSH=0
export SIGHTLINE_BACKEND_DIR="$W/db" SIGHTLINE_EVENTS_DIR="$W/events" SIGHTLINE_UPLOAD_DIR="$W/uploads"
export SIGHTLINE_GPUS="${CAMTEST_GPUS:-cuda:0,cuda:1}" SIGHTLINE_CAMERAS_PER_GPU=3
export SIGHTLINE_LOG_DIR="${SIGHTLINE_LOG_DIR:-$WORK/logs}"
exec bash backend/scripts/run_backend.sh
