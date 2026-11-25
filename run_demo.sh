#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/4] 微型枚举校准"
python3 -m wordboard check --json-out outputs/calibration.json

echo "[2/4] 生成可复跑合成输入"
python3 -m wordboard generate --seed 2026 --out inputs/sample.json

echo "[3/4] 训练、排卡、策略优化、会话回放"
python3 -m wordboard all --input inputs/sample.json --outdir outputs/sample

echo "[4/4] 单元测试"
python3 -m unittest discover -s tests

echo "完成：阅读 outputs/sample/report.md"
