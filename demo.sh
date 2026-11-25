#!/usr/bin/env bash
# LedgerScribe 端到端演示：从零生成材料、训练、对平、复核到模型检查。
# 纯标准库、普通 CPU、固定随机种子，可复跑。
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. 生成合成材料（书写员 A/B 训练集 + 检验页 + 新书写员页） =="
python3 -m ledgerscribe.cli generate --data data

echo "== 2. 用只有金额标签的短行训练两个书写员模型（约束 EM） =="
python3 -m ledgerscribe.cli train --data data --writer A --out data/model_A.json
python3 -m ledgerscribe.cli train --data data --writer B --out data/model_B.json

echo "== 3. 行报告：每个候选金额的后验与笔段->字归属 =="
python3 -m ledgerscribe.cli line --model data/model_A.json \
  --lines data/heldout_lines.json --line-id A-line-1 \
  --page data/pages.json --page-id A-040

echo "== 4a. 合计 40：两种对平读法，两行都留白 =="
python3 -m ledgerscribe.cli page --model data/model_A.json \
  --lines data/heldout_lines.json --page data/pages.json --page-id A-040

echo "== 4b. 合计 35：唯一对平读法，直接签 12 与 23 =="
python3 -m ledgerscribe.cli page --model data/model_A.json \
  --lines data/heldout_lines.json --page data/pages.json --page-id A-035

echo "== 4c. 第四候选小页：只有第四候选 34 能对平 86 =="
python3 -m ledgerscribe.cli page --model data/model_A.json \
  --lines data/heldout_lines.json --page data/pages.json --page-id A-086

echo "== 5. 路线 A 复核规划（两格工时，最坏整字答复下最大化签行） =="
python3 -m ledgerscribe.cli plan --model data/model_A.json \
  --lines data/heldout_lines.json --page data/pages.json --page-id A-040

echo "== 6. 收到馆员答复（末笔段是 7）后重跑对平，签 17 与 23 =="
python3 -m ledgerscribe.cli reply --model data/model_A.json \
  --lines data/heldout_lines.json --page data/pages.json --page-id A-040 \
  --reply replies/reply_A-040_digit7.json

echo "== 7. 新书写员 B + 新页 B-040：17 跌出门槛，只签 12 与 28 =="
python3 -m ledgerscribe.cli page --model data/model_B.json \
  --lines data/newpage_lines.json --page data/newpage_page.json --page-id B-040

echo "== 8. 独立模型检查（前向 DP 与递归枚举逐金额对平、后验归一） =="
python3 -m ledgerscribe.cli check --model data/model_A.json \
  --lines data/heldout_lines.json --page data/pages.json --page-id A-040

echo "== 9. 单元测试 =="
python3 -m unittest tests.test_ledgerscribe
