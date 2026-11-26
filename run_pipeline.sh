#!/usr/bin/env bash
# Safe verifier entry point: refit and produce new deliverables from an input
# directory without regenerating or overwriting the swapped input files.
#
# Usage:
#   ./run_pipeline.sh [INPUT_DIR] [OUTPUT_DIR]
set -euo pipefail
cd "$(dirname "$0")"

INPUT_DIR="${1:-input}"
OUTPUT_DIR="${2:-deliverables/swapped_run}"

python3 -m coffee_blend pipeline \
  --input-dir "$INPUT_DIR" \
  --out-dir "$OUTPUT_DIR"

echo "Wrote deliverables to: $OUTPUT_DIR"
echo "Read: $OUTPUT_DIR/blind_evaluation.md"
echo "Read: $OUTPUT_DIR/pipeline_summary.md"
