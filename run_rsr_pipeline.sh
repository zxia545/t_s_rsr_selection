#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG_JSON [extra rsr_pipeline.py args...]"
  exit 1
fi

CONFIG_PATH="$1"
shift

python "${ROOT_DIR}/rsr_pipeline.py" --config "${CONFIG_PATH}" "$@"
