#!/bin/bash
# Start the pipeline (inbox watcher + worker + dashboard) and keep the Mac awake while it runs.
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo "No virtualenv found. Run scripts/setup.sh first." >&2
  exit 1
fi
exec caffeinate -i .venv/bin/python -m mashuptech serve "$@"
