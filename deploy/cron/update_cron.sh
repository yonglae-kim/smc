#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/kquant.log"
MARKER="kquant_smc_reporter"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing virtualenv at ${PYTHON_BIN}. Create it under ${PROJECT_ROOT}/.venv first."
  exit 1
fi

mkdir -p "${LOG_DIR}"

CRON_ENTRY="0 19 * * * ${PYTHON_BIN} ${PROJECT_ROOT}/main.py --config ${CONFIG_FILE} >> ${LOG_FILE} 2>&1"

CURRENT_CRON="$(crontab -l 2>/dev/null || true)"

if echo "${CURRENT_CRON}" | grep -q "${MARKER}"; then
  echo "Cron entry already registered. Skipping."
  exit 0
fi

{
  if [[ -n "${CURRENT_CRON}" ]]; then
    echo "${CURRENT_CRON}"
  fi
  echo "# ${MARKER}"
  echo "${CRON_ENTRY}"
} | crontab -

echo "Cron entry registered:"
echo "${CRON_ENTRY}"
