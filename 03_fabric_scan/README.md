# 03　两处布面都像跳纱，该先复扫还是下剪

从空目录搭建的可重跑布面复扫与裁切软件。纯 Python 3 标准库，普通 CPU 秒级完成；
随机数全部来自带种子的 `random.Random`，同一份配置重跑逐字节一致。

## 问题对应关系

| 题目要求 | 实现位置 |
| --- | --- |
| 每布至多六格，低清读头一次跨相邻两格 | `common.all_maps`（枚举 2^n 候选图）、`datagen` 生成 n-1 次相邻观测 |
| 学整幅疵点图先验 + 相邻格给定时的误读分布 | `learn.learn_model`：先验与 4×4 误读表均做加性平滑 |
| 留出整条布上完整图的概率 | `learn.evaluate`：每条留出布给出真值图概率、对数损失、MAP、Brier |
| 缺读只略去该次观测 | `common.observation_likelihood` 跳过 `null`，其余观测格位不错位 |
| 精检前两格坏布概率相同、精检后裁法相反 | `demo.py` 两个两格玩具模型（恰坏一格 / 同好同坏） |
| 每格损失、留样格、初始头位、单据 | 下单 JSON（见 `examples/order_007.json` 示例） |
| 最多剪一个连续区段 | `policy.feasible_segments` 枚举全部可行区段（含不剪） |
| 便携头：两单格、第二格按第一次结果决定、按距离+启动付费 | `policy.portable_plan`，自适应分支 `branches.first_0/first_1` |
| 桥式头：锁定至多两格连续窗口、一趟看清、一次开机费 | `policy.bridge_plan`（对照路线，同一份裁切单也能交） |
| 下单只决定值不值得扫，收读数后才开剪口 | `workflow.place_order` 三步：`prepared → reading → cut` |
| 按设备可行操作最小化期望损失 | 全部策略枚举取最小；测试用独立暴力枚举复核 |
| 布号档案：复扫/读数/裁断，重开单看出哪段不能剪 | `order.py` 写 ledger，`blocked_cells` 硬拦截冲突剪口 |

## 目录

```
fabric_scan/
  common.py    位串图、缺读处理、后验 P(map|读数,精检)
  datagen.py   带种子的合成材料（material_a / material_b）
  learn.py     先验、误读分布学习与留出评估
  policy.py    两种设备的可行操作枚举与期望损失
  order.py     单据与布号状态档案
  workflow.py  正常下单入口
  demo.py      题目指定两格场景 + 费用逐项核对（带断言）
  cli.py       命令行
tests/test_system.py
examples/      正常下单入口的换入输入（单据 JSON + 该台读数 JSON）
run.sh         一键全流程
```

## 一键运行

```bash
./run.sh
```

依次执行：单元测试 → 生成材料 → 学习+留出评估 → 题目演示（全部断言）
→ 用未出现在演示里的读数走正常下单入口 → 同布号重开单。

## 分步命令（换一份相同格式输入即可取得新交付物）

```bash
# 1. 生成材料（改 MATERIALS 配置或直接提供同格式 JSON）
python3 -m fabric_scan.cli gen-data --material material_a --out out/material_a.json
# 2. 学习并在留出整条布上给完整图概率
python3 -m fabric_scan.cli learn --data out/material_a.json \
    --out out/model_a.json --holdout-report out/holdout_a.json
# 3. 题目两格玩具场景（含费用手算核对，断言失败即报错）
python3 -m fabric_scan.cli demo --out-dir out
# 4. 正常下单入口：order.json + precision.json 换一份就是新单
python3 -m fabric_scan.cli place-order \
    --model out/model_a.json --order examples/order_007.json \
    --precision examples/precision_007.json --ledger out/cust_ledger.json
```

### 输入格式

`order.json`

```json
{
  "order_id": "CUST-2026-007",
  "fabric_id": "FAB-7741",
  "device": "portable",            // 或 "bridge"
  "losses": [2, 1, 1, 3, 1, 2],    // 每格损失
  "retain": [0],                   // 客户留样，不可剪
  "head0": 3,                      // 初始头位
  "startup_fee": 0.6, "move_fee": 0.25, "bridge_fee": 1.2,
  "readings": [[1,0],[0,0],null,[1,0],[0,1]]  // null=缺读，只略去该次
}
```

`precision.json`：格号字符串 → `0`好/`1`坏，只给该台实际能返回的格
（便携头按策略可能自适应扫到第二格，桥式头按锁定窗口取数；缺格会明确报错）。

## 两格玩具结论（demo 自动断言）

先验一“整图恰坏一格”`{01,10}` 与先验二“同好同坏”`{00,11}`，
精检前两格坏布概率都各是 0.5：

1. 恰坏一格，精检格0为好 → 剪格1，期望/实际损失 0；
2. 同好同坏，精检格0为好 → 两格保留，损失 0；
3. 固定模型与精检、把格1列留样 → 剪口被迫为空，对真值 `01` 计漏剪损失 1。

费用小单（head0=2，开机 3，每格移动 2）：扫格4 收 3+2×2=7，
再扫格0 只收 2×4=8（开机只计一次），合计 15；剪口损失三种情形各为 0/1/1。

## 两条路线

默认开发并演示 **便携头**；`bridge_plan` 与 `place-order --device bridge`
给出桥式头的并行结果，用于理解差别：桥式头开机前定窗、一趟付费、
无自适应；便携头可按第一次读数改第二格，但移动与启动逐项计费。
