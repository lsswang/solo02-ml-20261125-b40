#!/usr/bin/env bash
# Judge another readings-only clip of the same JSON format.
# Usage: scripts/judge_another_clip.sh PATH_TO_CLIP.json [OUTPUT_STEM]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIP="${1:?usage: judge_another_clip.sh CLIP.json [OUTPUT_STEM]}"
if command -v realpath >/dev/null 2>&1; then
  CLIP="$(realpath "$CLIP")"
fi
STEM="${2:-${ROOT}/deliverables/replacement_decision}"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 "$ROOT/run.py" judge \
  --clip "$CLIP" \
  --model "$ROOT/models/model_v1.json" \
  --route "$ROOT/data/route_card.json" \
  --out-json "${STEM}.json" \
  --out-md "${STEM}.md"

echo "Done: ${STEM}.json and ${STEM}.md"
