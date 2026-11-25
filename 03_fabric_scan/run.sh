#!/usr/bin/env bash
# 一键全流程：测试 -> 材料 -> 学习 -> 演示 -> 正常下单入口 -> 同布号重开单
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out

echo "########## 1/6 单元测试（含独立暴力枚举对照） ##########"
python3 -m unittest tests.test_system -v

echo "########## 2/6 生成小型合成材料（带种子，可重跑） ##########"
python3 -m fabric_scan.cli gen-data --material material_a --out out/material_a.json
python3 -m fabric_scan.cli gen-data --material material_b --out out/material_b.json

echo "########## 3/6 学习先验与误读分布，留出整条布给完整图概率 ##########"
python3 -m fabric_scan.cli learn --data out/material_a.json \
    --out out/model_a.json --holdout-report out/holdout_a.json
python3 -m fabric_scan.cli learn --data out/material_b.json \
    --out out/model_b.json --holdout-report out/holdout_b.json

echo "########## 4/6 题目两格玩具场景 + 费用逐项核对（断言） ##########"
python3 -m fabric_scan.cli demo --out-dir out

echo "########## 5/6 正常下单入口（便携头，演示未出现的读数，含缺读） ##########"
rm -f out/cust_ledger.json
python3 -m fabric_scan.cli place-order \
    --model out/model_a.json --order examples/order_007.json \
    --precision examples/precision_007.json --ledger out/cust_ledger.json

echo "########## 6/6 同布号重开单：档案 blocked 生效，剪口避开已裁段 ##########"
python3 -m fabric_scan.cli place-order \
    --model out/model_a.json --order examples/order_008.json \
    --precision examples/precision_007.json --ledger out/cust_ledger.json

echo
echo "全部完成。交付物在 out/，布号档案见 out/cust_ledger.json。"
