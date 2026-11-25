#!/usr/bin/env bash
# End-to-end minimal two-face tray demonstration.
# Regenerates inputs, learns orientation weights, computes the optimal
# workbench policy, then registers a real persistent case and walks through
# bright/dark entry until checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STATE_DIR="runs/booth"
mkdir -p "$STATE_DIR/cases"
rm -f "$STATE_DIR/cases/DEMO-001.json"

python3 run.py generate --out examples
python3 run.py all \
  --input examples/minimal_tray.json \
  --out deliverables/minimal

echo
echo "================ 1. 登记新矿片 ================"
python3 run.py register \
  --id DEMO-001 \
  --input deliverables/minimal/resolved_input.json \
  --state-dir "$STATE_DIR"

echo
echo "================ 2. 正面试验：实际看到明 ================"
python3 run.py observe \
  --id DEMO-001 --action front_signal --result bright \
  --state-dir "$STATE_DIR"

echo
echo "================ 3. 中途退出后，凭片号查状态 ================"
python3 run.py status --id DEMO-001 --state-dir "$STATE_DIR"

echo
echo "================ 4. 翻一次面 ================"
python3 run.py flip --id DEMO-001 --state-dir "$STATE_DIR"

echo
echo "================ 5. 背面第二件仪器：实际看到明 ================"
python3 run.py observe \
  --id DEMO-001 --action back_signal --result bright \
  --state-dir "$STATE_DIR"

echo
echo "================ 6. 概率达标，签出 ================"
python3 run.py checkout --id DEMO-001 --state-dir "$STATE_DIR"

echo
echo "完成。病例文件：$STATE_DIR/cases/DEMO-001.json"
