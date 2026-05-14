#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  ./scripts/setup_unix.sh
fi
source .venv/bin/activate
python -m edit_with_my_voice.app
