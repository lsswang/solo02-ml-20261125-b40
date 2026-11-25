#!/usr/bin/env bash
# Reproducible CPU demo: calibration, a replaceable new scan, both milling
# methods, formal supplements, cut lookup by order number, and enumeration.
set -euo pipefail
cd "$(dirname "$0")"

OUT=out/demo
STATE="$OUT/state"
rm -rf "$OUT"
mkdir -p "$STATE" data

expect_rc() {
  local expected="$1"; shift
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ "$rc" != "$expected" ]]; then
    echo "Expected exit code $expected, got $rc from: $*" >&2
    exit 1
  fi
}

echo "== [1/8] Generate small synthetic calibration coupons and new scans =="
python3 -m relief_workshop generate --out data/generated_2x4 --seed 15 --rows 2 --cols 4
python3 -m relief_workshop generate --out data/generated_3x6 --seed 36 --rows 3 --cols 6

echo "== [2/8] Calibrate -2..2 slope probabilities; score a whole held-out coupon =="
python3 -m relief_workshop calibrate \
  --train data/generated_2x4/calibration_train.csv \
  --holdout data/generated_2x4/calibration_holdout.csv \
  --model "$OUT/model.json" \
  --report "$OUT/calibration_report.json" | tee "$OUT/calibration_console.json"

echo "== [3/8] Predict every directed edge of another 2x4 scan and a 3x6 scan =="
python3 -m relief_workshop scan-edges \
  --scan data/generated_2x4/new_scan.csv \
  --model "$OUT/model.json" \
  --out "$OUT/generated_2x4_edges.json" 2>&1
python3 -m relief_workshop scan-edges \
  --scan data/generated_3x6/new_scan.csv \
  --model "$OUT/model.json" \
  --out "$OUT/generated_3x6_edges.json" 2>&1

echo "== [4/8] Start a generated job and independently enumerate its 2^? grid =="
python3 -m relief_workshop start \
  --edges "$OUT/generated_2x4_edges.json" \
  --order data/generated_2x4/new_scan_work_order.json \
  --state-dir "$STATE" \
  --out "$OUT/generated_start.json"
python3 -m relief_workshop verify \
  --edges "$OUT/generated_2x4_edges.json" \
  --order data/generated_2x4/new_scan_work_order.json \
  --out "$OUT/generated_verify.json"
python3 -m relief_workshop start \
  --edges "$OUT/generated_3x6_edges.json" \
  --order data/generated_3x6/new_scan_work_order.json \
  --state-dir "$STATE" \
  --out "$OUT/generated_3x6_start.json" > "$OUT/generated_3x6_compact.json"

echo "== [5/8] Canonical 2x2 stamp: 0, then +1 gives 2, and pre-locking 2 is infeasible =="
python3 -m relief_workshop local-check \
  --edges examples/canonical_2x2/edge_probabilities.json \
  --order examples/canonical_2x2/order_original.json \
  --out "$OUT/canonical_2x2_local_modes.json" || true
python3 -m relief_workshop start \
  --edges examples/canonical_2x2/edge_probabilities.json \
  --order examples/canonical_2x2/order_original.json \
  --holes examples/canonical_2x2/holes_left.csv \
  --state-dir "$STATE" \
  --out "$OUT/start_2x2_original.json"
python3 -m relief_workshop verify \
  --edges examples/canonical_2x2/edge_probabilities.json \
  --order examples/canonical_2x2/order_original.json \
  --out "$OUT/verify_2x2_original.json"
python3 -m relief_workshop start \
  --edges examples/canonical_2x2/edge_probabilities.json \
  --order examples/canonical_2x2/order_lower_plus.json \
  --state-dir "$STATE" \
  --out "$OUT/start_2x2_lower_plus.json"
expect_rc 3 python3 -m relief_workshop start \
  --edges examples/canonical_2x2/edge_probabilities.json \
  --order examples/canonical_2x2/order_lock_right2.json \
  --state-dir "$STATE" \
  --out "$OUT/start_2x2_locked2_infeasible.json"

echo "== [6/8] Canonical 2x3: compare whole, middle seam, and right seam risk =="
python3 -m relief_workshop start \
  --edges examples/canonical_2x3/edge_probabilities.json \
  --order examples/canonical_2x3/order.json \
  --holes examples/canonical_2x3/holes_left.csv \
  --state-dir "$STATE" \
  --out "$OUT/start_2x3.json"
python3 -m relief_workshop verify \
  --edges examples/canonical_2x3/edge_probabilities.json \
  --order examples/canonical_2x3/order.json \
  --out "$OUT/verify_2x3.json"

echo "== [7/8] Replay formal tickets: middle conflicts, right completes the seam =="
expect_rc 4 python3 -m relief_workshop supplement \
  --state-dir "$STATE" \
  --order-id WO-2X3-SEAM \
  --ticket examples/canonical_2x3/supplement_mid2.json \
  --out "$OUT/supplement_mid_conflict.json"
python3 -m relief_workshop cuts \
  --state-dir "$STATE" \
  --order-id WO-2X3-SEAM > "$OUT/cuts_after_conflict.json"
python3 -m relief_workshop supplement \
  --state-dir "$STATE" \
  --order-id WO-2X3-SEAM \
  --ticket examples/canonical_2x3/supplement_right2.json \
  --out "$OUT/supplement_right_accepted.json"
python3 -m relief_workshop cuts \
  --state-dir "$STATE" \
  --order-id WO-2X3-SEAM > "$OUT/cuts_WO-2X3-SEAM.json"
python3 -m relief_workshop cuts \
  --state-dir "$STATE" \
  --order-id WO-2X3-RIGHT2 > "$OUT/cuts_WO-2X3-RIGHT2.json"

echo "== [8/8] Automated checks =="
python3 -m unittest discover -s tests

cat <<EOF

Demo complete.
  Calibration report : $OUT/calibration_report.json
  Start decisions    : $OUT/start_*.json
  Supplement results : $OUT/supplement_*.json
  Cut lookup examples: $OUT/cuts_*.json
  State store        : $STATE/orders/

Try another same-format scan:
  python3 -m relief_workshop generate --out data/my_scan --seed 123 --rows 3 --cols 5
  python3 -m relief_workshop scan-edges --scan data/my_scan/new_scan.csv \\
    --model $OUT/model.json --out /tmp/my_edges.json
EOF
