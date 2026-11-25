#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

echo "== Generating reproducible local synthetic bundles =="
$PY -m guide_studio generate-sample --out sample_input --scenario baseline
$PY -m guide_studio generate-sample --out sample_input_west_lin --scenario west_lin
$PY -m guide_studio generate-sample --out sample_input_high_fee --scenario high_fee
$PY -m guide_studio generate-sample --out sample_input_revised_script --scenario revised_script

rm -rf out/baseline out/closed_west out/west_lin out/high_fee out/revised_script

echo "== Baseline: fit, proofread, optimize wall+pen, render WAV/replays =="
$PY -m guide_studio run --input sample_input --output out/baseline
$PY -m guide_studio verify --summary out/baseline/run_summary.json

echo "== Close West gate: only East route remains =="
$PY -m guide_studio run --input sample_input --output out/closed_west --close west_gate

echo "== Replacement listening records: West also points to Lin =="
$PY -m guide_studio run --input sample_input_west_lin --output out/west_lin

echo "== Script replacement: West manuscript uses full name =="
$PY -m guide_studio run --input sample_input_revised_script --output out/revised_script --no-audio

echo "== Cost sensitivity: expensive extra short recording =="
$PY -m guide_studio run --input sample_input_high_fee --output out/high_fee --no-audio

echo "== Plate-string replay examples =="
$PY -m guide_studio replay \
  --plan out/baseline/plans/pen_zh.json \
  --plates east_gate,lin_hall,silent_porch,portrait \
  | tee out/baseline/replays/demo_pen_zh_east_console.txt

$PY -m guide_studio replay \
  --plan out/baseline/plans/wall_zh.json \
  --plates west_gate,zhou_hall,silent_porch,portrait \
  | tee out/baseline/replays/demo_wall_zh_west_console.txt

echo "== Closed-West wall replay =="
$PY -m guide_studio replay \
  --plan out/closed_west/plans/wall_zh.json \
  --plates east_gate,lin_hall,silent_porch,portrait \
  | tee out/closed_west/replays/demo_wall_zh_east_console.txt

echo "== Unit/scenario tests =="
$PY -m unittest discover -s tests

echo "Done. See out/baseline/run_summary.md first."
