#!/usr/bin/env bash
# Build a delivery from another program of the same JSON/WAV format.
# Usage: ./run_on_input.sh PATH/TO/program.json PATH/TO/learned_room.json OUTPUT_DIR
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 program.json learned_room.json output_dir" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PROGRAM="$1"
ROOM="$2"
OUTDIR="$3"

python3 -m soundbench build \
  --program "$PROGRAM" \
  --room "$ROOM" \
  --outdir "$OUTDIR" \
  --strict

echo "New delivery: $OUTDIR/speaker_track.wav"
echo "Report:       $OUTDIR/verification.json"
