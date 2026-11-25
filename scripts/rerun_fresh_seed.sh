#!/usr/bin/env bash
# Generate a fresh synthetic bundle with another seed and produce deliverables.
# Usage: scripts/rerun_fresh_seed.sh [SEED] [OUTPUT_DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEED="${1:-777}"
OUT="${2:-/tmp/boulder_fresh_${SEED}}"

mkdir -p "$OUT"
python3 -m boulder_judge.cli generate-sample --outdir "$OUT" --seed "$SEED"
python3 -m boulder_judge.cli handcheck \
  --out-json "$OUT/hand_examples.json" \
  --out-md "$OUT/hand_examples.md" --strict
python3 -m boulder_judge.cli train \
  --data "$OUT/data/synthetic_corpus.json" \
  --out "$OUT/model_v1.json" \
  --version "fresh-seed-${SEED}"
python3 -m boulder_judge.cli heldout \
  --data "$OUT/data/synthetic_corpus.json" \
  --model "$OUT/model_v1.json" \
  --route "$OUT/data/route_card.json" \
  --out-json "$OUT/heldout_report.json" \
  --out-md "$OUT/heldout_report.md"
python3 -m boulder_judge.cli judge \
  --clip "$OUT/data/new_attempt.json" \
  --model "$OUT/model_v1.json" \
  --route "$OUT/data/route_card.json" \
  --out-json "$OUT/new_decision.json" \
  --out-md "$OUT/new_decision.md"

echo "Fresh deliverables are in $OUT"
