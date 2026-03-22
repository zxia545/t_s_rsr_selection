#!/usr/bin/env bash
set -euo pipefail

source activate rsr_selection

conda activate rsr_selection

# check does current conda env is rsr_selection
if [[ "$(conda info --envs | grep '*' | awk '{print $1}')" != "rsr_selection" ]]; then
  echo "Please activate the conda environment 'rsr_selection' before running this script."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG_JSON [extra rsr_pipeline.py args...]"
  exit 1
fi

CONFIG_PATH="$1"
shift

python "${ROOT_DIR}/rsr_pipeline.py" --config "${CONFIG_PATH}" "$@"
