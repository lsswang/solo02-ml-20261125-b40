#!/usr/bin/env bash
# Reproduce the complete CPU-only reference demonstration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APPEAL_NO="APP-2026-0001"
rm -rf "cases/${APPEAL_NO}"
rm -f deliverables/appeal_v2_preview.json deliverables/appeal_v2_preview.md \
      models/model_v2_recal.json

# 1. Deterministic small synthetic materials (<=16 frames/clip).
python3 -m boulder_judge.cli generate-sample --outdir "$ROOT"

# 2. Hand-enumerable probability invariants.
python3 -m boulder_judge.cli handcheck --strict

# 3. Learn joint transitions and paired emission errors.
python3 -m boulder_judge.cli train \
  --data data/synthetic_corpus.json \
  --out models/model_v1.json \
  --version cal-v1-initial

python3 -m boulder_judge.cli train \
  --data data/synthetic_corpus.json \
  --data data/calibration_extra.json \
  --data data/calibration_false_top_revision.json \
  --out models/model_v2_revised.json \
  --version cal-v2-paired-error-revision

# 4. Check joint contact distribution on whole held-out clips.
python3 -m boulder_judge.cli heldout \
  --data data/synthetic_corpus.json \
  --model models/model_v1.json \
  --route data/route_card.json \
  --out-json deliverables/heldout_report_v1.json \
  --out-md deliverables/heldout_report_v1.md

python3 -m boulder_judge.cli heldout \
  --data data/synthetic_corpus.json \
  --model models/model_v2_revised.json \
  --route data/route_card.json \
  --out-json deliverables/heldout_report_v2_recalibrated.json \
  --out-md deliverables/heldout_report_v2_recalibrated.md

# Keep canonical filenames too.
cp deliverables/heldout_report_v1.json deliverables/heldout_report.json
cp deliverables/heldout_report_v1.md deliverables/heldout_report.md

# 5. A new clip submitted for adjudication.
python3 -m boulder_judge.cli judge \
  --clip data/new_attempt.json \
  --model models/model_v1.json \
  --route data/route_card.json \
  --out-json deliverables/new_decision_v1.json \
  --out-md deliverables/new_decision_v1.md

# 6. Another attempt: submission -> signed TOP -> calibration appeal -> ZONE.
python3 -m boulder_judge.cli judge \
  --clip data/appeal_attempt.json \
  --model models/model_v1.json \
  --route data/route_card.json \
  --out-json deliverables/appeal_initial_v1.json \
  --out-md deliverables/appeal_initial_v1.md \
  --case-root cases \
  --appeal-number "$APPEAL_NO"

python3 -m boulder_judge.cli appeal \
  --case-dir "cases/${APPEAL_NO}" \
  --new-model models/model_v2_revised.json \
  --reason "新人工标注校准批次发现 TOP/TOP 成对读数可由 ZONE/ZONE 的相关误识别产生" \
  --appellant head_judge_review

# Optional comparison: what the other, readings-destroyed venue can seal.
python3 -m boulder_judge.cli sealed-preview \
  --clip data/appeal_attempt.json \
  --model models/model_v1.json \
  --route data/route_card.json \
  --testimony-a data/sealed_testimony_a_top.json \
  --testimony-b data/sealed_testimony_b_zone.json \
  --appeal-number SEALED-COMPARISON-A17 \
  --bundle-json deliverables/sealed_evidence_bundle.json \
  --out-json deliverables/sealed_preview.json \
  --out-md deliverables/sealed_preview.md

# 7. Automated checks.
python3 -m unittest discover -s tests

echo
echo "DEMO OK. Key deliverables:"
echo "  deliverables/hand_examples.md"
echo "  deliverables/heldout_report_v1.md"
echo "  deliverables/new_decision_v1.md"
echo "  cases/${APPEAL_NO}/DOSSIER.md"
