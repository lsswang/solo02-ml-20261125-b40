#!/usr/bin/env bash
# Reproducible end-to-end run.  Pure Python 3 (stdlib only), no network.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}

echo "== 1. generate synthetic material =="
$PY -m proofing.cli gen-data --outdir data

echo "== 2. fit learned response + hold-out report =="
$PY -m proofing.cli fit --data data --out artifacts/response.json

echo "== 3. write example job specs =="
$PY -m proofing.cli gen-jobs --outdir jobs

echo "== 4. plan every example job (route 1: direct output) =="
for j in wide_block wide_block_locked_reverse wide_block_paper2 \
         edge_image edge_image_locked_zero scrap_demo triple_demo tone_ramp; do
  echo "--- $j ---"
  $PY -m proofing.cli plan --data data --job "jobs/$j.json" \
      --out "artifacts/plan_$j.json"
done

echo "== 5. per-offset verification of the edge image =="
echo "--- edge_image / forward (must fail under some offsets) ---"
$PY -m proofing.cli report --data data --job jobs/edge_image.json --recipe forward || true
echo "--- edge_image_locked_zero / forward (passes at the only offset) ---"
$PY -m proofing.cli report --data data --job jobs/edge_image_locked_zero.json --recipe forward

echo "== 6. pressman feedback: recheck entered offset =="
$PY -m proofing.cli recheck --data data --job jobs/wide_block.json \
    --plan artifacts/plan_wide_block.json --offset '{"A":0,"B":0}'

echo "== 7. tests =="
$PY -m unittest discover -s tests -v

echo "done. outputs in data/, jobs/, artifacts/"
