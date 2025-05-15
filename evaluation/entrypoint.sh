#!/bin/bash
set -e

echo "[ENTRYPOINT] Starting..."

PROJECT_NAME="sbs-comparison"
PROJECT_DIR="/label-studio/projects/${PROJECT_NAME}"
CONFIG_PATH="${PROJECT_DIR}/config.xml"
DATA_PATH="${PROJECT_DIR}/data/data.json"

if [ ! -f "${PROJECT_DIR}/label_studio.sqlite3" ]; then
  echo "[ENTRYPOINT] Creating project..."
  label-studio init "${PROJECT_NAME}" --label-config "${CONFIG_PATH}"
else
  echo "[ENTRYPOINT] Project already exists."
fi

echo "[ENTRYPOINT] Starting Label Studio..."
label-studio start \
  --username "${LABEL_STUDIO_USERNAME}" \
  --password "${LABEL_STUDIO_PASSWORD}" \
  --no-browser
