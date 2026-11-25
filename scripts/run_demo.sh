#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="${1:-sample_inputs}"
OUT="${2:-out/demo}"
rm -rf "$OUT"
mkdir -p "$OUT/state"

echo "### 1. Fit original and genuinely recalibrated response matrices"
python3 -m observatory calibrate \
  --input "$INPUT/calibration_original" \
  --out "$OUT/calibration_original.json"
python3 -m observatory calibrate \
  --input "$INPUT/calibration_recalibrated" \
  --out "$OUT/calibration_recalibrated.json"

echo "### 2. Point-query the canonical drop (6,3)"
python3 -m observatory query \
  --calibration "$INPUT/calibration_textbook_exact.json" \
  --observation "$INPUT/observations/drop_6_3.json" \
  --uncertainty none \
  --out "$OUT/query_textbook_exact.json"
python3 -m observatory query \
  --calibration "$OUT/calibration_original.json" \
  --observation "$INPUT/observations/drop_6_3.json" \
  --out "$OUT/query_original.json"
python3 -m observatory query \
  --calibration "$OUT/calibration_recalibrated.json" \
  --observation "$INPUT/observations/drop_6_3.json" \
  --out "$OUT/query_recalibrated.json"

echo "### 3. Three nights / four ephemerides: fixed baseline and selected voucher"
python3 -m observatory plan \
  --application "$INPUT/applications/order_four.json" \
  --shifts "$INPUT/shifts/shifts_three.csv" \
  --query "$OUT/query_original.json" \
  --method both \
  --out "$OUT/plan_four.json"

python3 -m observatory book \
  --application "$INPUT/applications/order_four.json" \
  --shifts "$INPUT/shifts/shifts_three.csv" \
  --query "$OUT/query_original.json" \
  --state-dir "$OUT/state" \
  --registry "$OUT/state/registry.json" \
  --reservation-id RSV-FOUR-BRIGHT \
  --voucher-id VCH-FOUR-BRIGHT
python3 -m observatory fill-first \
  --state-dir "$OUT/state" --registry "$OUT/state/registry.json" \
  --reservation-id RSV-FOUR-BRIGHT --result bright
python3 -m observatory fill-second \
  --state-dir "$OUT/state" --registry "$OUT/state/registry.json" \
  --reservation-id RSV-FOUR-BRIGHT --result bright

python3 -m observatory book \
  --application "$INPUT/applications/order_four.json" \
  --shifts "$INPUT/shifts/shifts_three.csv" \
  --query "$OUT/query_original.json" \
  --state-dir "$OUT/state" \
  --registry "$OUT/state/registry.json" \
  --reservation-id RSV-FOUR-DIM \
  --voucher-id VCH-FOUR-DIM
python3 -m observatory fill-first \
  --state-dir "$OUT/state" --registry "$OUT/state/registry.json" \
  --reservation-id RSV-FOUR-DIM --result dim

echo "### 4. Same AB application, original calibration, open night 4 vs night 5"
python3 -m observatory plan \
  --application "$INPUT/applications/order_ab.json" \
  --shifts "$INPUT/shifts/shifts_ab_open4.csv" \
  --query "$OUT/query_original.json" \
  --method voucher \
  --out "$OUT/plan_ab_original_open4.json"
python3 -m observatory plan \
  --application "$INPUT/applications/order_ab.json" \
  --shifts "$INPUT/shifts/shifts_ab_open5.csv" \
  --query "$OUT/query_original.json" \
  --method voucher \
  --out "$OUT/plan_ab_original_open5.json"

python3 -m observatory book \
  --application "$INPUT/applications/order_ab.json" \
  --shifts "$INPUT/shifts/shifts_ab_open4.csv" \
  --query "$OUT/query_original.json" \
  --state-dir "$OUT/state" \
  --registry "$OUT/state/registry.json" \
  --reservation-id RSV-AB-ORIG \
  --voucher-id VCH-AB-ORIG
python3 -m observatory fill-first \
  --state-dir "$OUT/state" --registry "$OUT/state/registry.json" \
  --reservation-id RSV-AB-ORIG --result bright
python3 -m observatory fill-second \
  --state-dir "$OUT/state" --registry "$OUT/state/registry.json" \
  --reservation-id RSV-AB-ORIG --result dim

echo "### 5. Recalibration restores A4; night 4 can no longer guarantee uniqueness"
python3 -m observatory plan \
  --application "$INPUT/applications/order_ab.json" \
  --shifts "$INPUT/shifts/shifts_ab_open4.csv" \
  --query "$OUT/query_recalibrated.json" \
  --method voucher \
  --out "$OUT/plan_ab_recalibrated_open4.json"
python3 -m observatory book \
  --application "$INPUT/applications/order_ab.json" \
  --shifts "$INPUT/shifts/shifts_ab_open4.csv" \
  --query "$OUT/query_recalibrated.json" \
  --state-dir "$OUT/state" \
  --registry "$OUT/state/registry.json" \
  --reservation-id RSV-AB-RECAL \
  --voucher-id VCH-AB-RECAL
python3 -m observatory fill-first \
  --state-dir "$OUT/state" --registry "$OUT/state/registry.json" \
  --reservation-id RSV-AB-RECAL --result bright
python3 -m observatory fill-second \
  --state-dir "$OUT/state" --registry "$OUT/state/registry.json" \
  --reservation-id RSV-AB-RECAL --result dim

echo "### 6. A voucher cannot be promised to a second group"
if python3 -m observatory book \
  --application "$INPUT/applications/order_ab.json" \
  --shifts "$INPUT/shifts/shifts_ab_open5.csv" \
  --query "$OUT/query_original.json" \
  --state-dir "$OUT/state" \
  --registry "$OUT/state/registry.json" \
  --reservation-id RSV-SHOULD-FAIL \
  --voucher-id VCH-FOUR-BRIGHT >/dev/null 2>&1; then
  echo "ERROR: duplicate voucher booking was accepted" >&2
  exit 1
else
  echo "duplicate voucher correctly rejected"
fi

echo "### 7. Unit tests"
python3 -m unittest discover -s tests

echo
echo "Demo complete. Key JSON deliverables are in $OUT"
