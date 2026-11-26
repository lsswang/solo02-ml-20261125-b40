#!/usr/bin/env bash
# Reproduce every sample deliverable on CPU with the Python standard library.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/5] Generating same-format synthetic sensory inputs"
python3 -m coffee_blend generate-data --out input/sensory_votes.csv --scenario good
python3 -m coffee_blend generate-data --out input_bad_b/sensory_votes.csv --scenario bad_b

echo "[2/5] Training, leaving whole roasts blind, and running good shop"
python3 -m coffee_blend pipeline --input-dir input --out-dir deliverables/good >/tmp/coffee_good_pipeline.json

echo "[3/5] Refitting on swapped new-B-batch input"
python3 -m coffee_blend pipeline --input-dir input_bad_b --out-dir deliverables/bad_b >/tmp/coffee_bad_pipeline.json

echo "[4/5] Demonstrating an input half-bucket and running assertions"
python3 -m coffee_blend.extra_demos
python3 verify.py

echo "[5/5] Done. Key files:"
echo "  deliverables/good/blind_evaluation.md"
echo "  deliverables/good/production_recommendations.json"
echo "  deliverables/good/actual_successful_weighing_example.json"
echo "  deliverables/good/pipeline_summary.md"
echo "  deliverables/bad_b/pipeline_summary.md"
