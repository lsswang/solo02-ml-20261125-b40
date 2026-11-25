# 展台矿片光学识别工作台

这是一个从零做的、可离线复跑的博物馆矿片识别教具。项目**只选择题目中的第二条路线：讲解员操作的自适应工作台**；不交付最多三张二分卡的纸质试验匣。工作台会根据每片矿片当前的证据，更新“材质 × 取向”联合概率，并决定下一项试验、翻面、签出或留未定。

全部代码只依赖 Python 3.11 标准库；样本很小，普通 CPU 几秒内可跑完。

## 1. 一分钟复跑

```bash
# 生成全部合成输入到 examples/
python3 run.py generate --out examples

# 内置断言：三个规定情形 + 极小展盘最优认出率 + 持久化端到端
python3 run.py selftest

# 输出题目指定情形的精确后验/预算曲线
python3 run.py scenario-report --out deliverables/scenario_checks.json

# 极小正反两面展盘：学习、留出核对、求最优策略、评估
python3 run.py all \
  --input examples/minimal_tray.json \
  --out deliverables/minimal

# 完整交互演示：登记 -> 输照明暗 -> 中途查号 -> 翻面 -> 换仪器 -> 签出
./scripts/demo_minimal.sh
```

也可以一次复跑全部五个输入：

```bash
./scripts/reproduce_all.sh
```

## 2. 观众中途退出后，凭片号查询

每个片号对应 `runs/<state-dir>/cases/<片号>.json`，关闭终端后仍然存在。

```bash
# 登记新片；resolved_input.json 已带有从训练记录学到的取向分布
python3 run.py register \
  --id S-001 \
  --input deliverables/minimal/resolved_input.json \
  --state-dir runs/booth

# 录入实际明暗：bright/1/明，或 dark/0/暗
python3 run.py observe \
  --id S-001 --action front_signal --result bright \
  --state-dir runs/booth

# 中途离开后回来查：实际花费、剩余预算、当前面、可翻次数、
# 现在/翻面后可做的动作、会失效的动作、材质后验、联合概率、当前判断
python3 run.py status --id S-001 --state-dir runs/booth

# 翻面；当前面专属试验之后会被程序拒绝
python3 run.py flip --id S-001 --state-dir runs/booth

# 录入第二面读数；第二件仪器第一次使用时自动加调机费
python3 run.py observe \
  --id S-001 --action back_signal --result bright \
  --state-dir runs/booth

# 后验 >= 0.90 才允许签出
python3 run.py checkout --id S-001 --state-dir runs/booth
```

需要机器可读状态时，在 `status/observe/flip/checkout` 后加 `--json`。

## 3. 换入另一份相同格式的输入

验题人不需要修改程序。把另一份符合下面结构的 JSON 放到任意路径，然后：

```bash
python3 run.py all \
  --input /path/to/another_tray.json \
  --out /tmp/new_deliverables

python3 run.py register \
  --id NEW-001 \
  --input /tmp/new_deliverables/resolved_input.json \
  --state-dir /tmp/new_booth

python3 run.py status --id NEW-001 --state-dir /tmp/new_booth
```

`all` 会在输出目录生成一套新的交付物；仓库已附带一份 `deliverables/scenario_checks.json`：

| 文件 | 内容 |
|---|---|
| `scenario_checks.json` | 三个规定情形的精确分数后验，以及极小展盘预算曲线 |
| `learned_model.json` | 从部分试验、已有材质标签的训练记录学到的各材质取向分布 |
| `validation_report.json` | 在整片留出标本上核对未参与推断读数的 Brier/log loss/准确率 |
| `workbench_policy.json` | 精确动态规划求出的最优工作台、预算曲线和状态决策表 |
| `evaluation_report.json` | 贝叶斯总体认出率、期望花费，以及合成新矿片逐片路径 |
| `resolved_input.json` | 原输入加学习结果；可直接用于 `register` |

## 4. 输入格式（`mineral-optical-tray/v1`）

完整例子见 `examples/sample_input.json`，极小例子见 `examples/minimal_tray.json`。主要字段：

```jsonc
{
  "schema_version": "mineral-optical-tray/v1",
  "confidence_threshold": 0.9,   // 签出阈值
  "budget": 3,                   // 每片硬预算
  "max_flips": 1,                // 本题通常为还能翻一次
  "initial_side": "front",
  "second_instrument_setup_fee": 1, // 也可写在 instruments[1].setup_fee
  "instruments": [
    {"id": "front_eye", "setup_fee": 0},
    {"id": "back_bench", "setup_fee": 1}
  ],
  "initial_paid_instruments": ["front_eye"],
  "tests": [
    {
      "id": "front_signal",
      "instrument": "front_eye",
      "side": "front",           // front | back | either
      "cost": 1,
      "sealed": false            // true 可模拟被封住的试验
    }
  ],
  "materials": [
    {"id": "A", "name": "甲", "prior": 0.5},
    {"id": "B", "name": "乙", "prior": 0.5}
  ],
  "templates": [
    {
      "material": "A",
      "orientation": "A-clear",
      "readings": {"front_signal": 1, "back_signal": 1} // 1=明, 0=暗
    }
  ],
  "training_records": [
    {"id": "t1", "material": "A", "readings": {"front_signal": 1}}
  ],
  "heldout_records": [
    {
      "id": "h1",
      "material": "A",
      "orientation": "A-clear",
      "readings": {"front_signal": 1, "back_signal": 1},
      "evidence_readings": {"front_signal": 1},
      "check_readings": {"back_signal": 1}
    }
  ],
  "new_specimens": [
    {"id": "demo-A", "material": "A", "orientation": "A-clear"}
  ]
}
```

约定：

- 最多五项光学试验；翻面动作单独放在 `flip_action`。
- 读数支持 `0/1`、`dark/bright`、`暗/明/亮/不透/透光`。
- `training_records` 只需材质标签和部分读数，不需要取向标签；取向由 EM 学习。
- `heldout_records` 可用 `evidence_readings + check_readings` 指定哪些读数不得参与推断；不写时默认做留一读数预测。
- `new_specimens` 与其中的真值只用于离线评估和自动演示；真实展台登记时不需要、也不向观众展示真值，向导只需用 `observe --result ...` 录入实际明暗。
- 模板中的 `weight` 只是生成样本时的真值/可选初始化参考；默认 EM 从均匀起点学习，不把它当作已知取向分布。

## 5. 概率与决策方法

### 5.1 学习取向分布

设材质为 \(m\)，取向模板为 \(o\)。先验为 \(\pi_m\)，材质内部取向概率为 \(\theta_{m,o}\)。训练记录只有材质标签和部分明暗读数；在确定性模板下，一条记录对所有兼容取向产生 EM 责任度：

\[
r_{i,o}=\frac{\theta_{m,o}\mathbf 1[\text{记录 }i\text{ 与模板 }o\text{ 兼容}]}
{\sum_{o'}\theta_{m,o'}\mathbf 1[\text{兼容}]}
\]

随后更新期望计数。代码见 `mineral_bench/learning.py`。

### 5.2 新矿片的联合更新

新片登记时：

\[
P(m,o)=\pi_m\theta_{m,o}.
\]

每录入一项试验结果 \(y_a\)，所有规定结果与实际明暗不符的 \((m,o)\) 假设被删除：

\[
P(m,o\mid y_{1:k})\propto
\pi_m\theta_{m,o}\prod_{a=1}^{k}
\mathbf 1[T_{m,o,a}=y_a].
\]

片号状态中同时输出材质边缘后验和完整“材质 × 取向”联合后验。

### 5.3 工作台最优策略

规划器是一个精确有限状态动态规划。状态为：

```text
(存活假设掩码, 当前正反面, 剩余翻面次数, 已做试验掩码,
 已付调机费仪器掩码, 剩余预算)
```

目标按题目排序：

1. 在硬预算内最大化达到材质后验 \(\ge 0.90\) 的概率（认出率）；
2. 认出率相同时最小化期望花费；
3. 若前两项仍相同，再偏好签出时更确定的分支。

第一次使用某仪器会加入其 `setup_fee`；之后切换走不再重复支付。翻面改变可做动作集合，正面专属动作若未在翻面之前完成，之后会被判定为不可用。

## 6. 题目指定情形的核对结果

`python run.py selftest` 会精确断言以下结果：

| 合成输入 | 读数/动作 | 结果 |
|---|---|---|
| `perfect_transmission.json` | 透光为明 | \(P(A\mid 明)=1\)，可签甲 |
| `perfect_transmission.json` | 透光为暗 | \(P(B\mid 暗)=1\)，可签乙 |
| `half_b_transmits.json` | 乙有一半取向透光；透光为明 | \(P(A\mid 明)=\frac23<0.90\)，留未定 |
| `half_b_transmits.json` | 透光为暗 | 乙后验为 1，可签乙 |
| `upside_down_sealed.json` | 透光被封，唯一余项甲乙相同 | 后验仍各为 \(1/2\)，留未定 |
| `minimal_tray.json` | 预算 0/1/2/3 | 最佳认出率 \(0 / 0.25 / 0.25 / 1\) |

极小展盘的预算曲线：

| 预算 | 最佳认出率 | 该认出率下期望花费 | 说明 |
|---:|---:|---:|---|
| 0 | 0 | 0 | 先验各半，不能签出 |
| 1 | 0.25 | 1 | 先做正面项；正面暗分支质量为 \(1/4\)，可签乙 |
| 2 | 0.25 | 1 | 正面明后付不起“试验 1 + 第二仪器调机 1”；先做背面项也只能得到相同认出率但期望花费 2 |
| 3 | 1.00 | 2.5 | 正面项 1；以 \(3/4\) 概率进入明分支，再付背面试验 1 + 调机 1，故期望 \(1+\frac34(2)=2.5\)，最长实际路径花费 3 |

端到端 `DEMO-001` 走的是：

```text
登记（A=50%, B=50%）
→ front_signal=bright（A=2/3，留未定，花费1）
→ 凭 DEMO-001 中途查询
→ flip（到背面，front_signal 失效）
→ back_signal=bright（A=1，花费3，其中调机费1）
→ checkout A
```

## 7. 目录导航

```text
run.py                         # CLI 入口
mineral_bench/core.py          # schema、读数归一化、联合假设
mineral_bench/learning.py      # EM 学习 + 留出读数预测核对
mineral_bench/planner.py       # 精确自适应工作台 DP
mineral_bench/workbench.py     # 片号持久化、花费/动作/判断
mineral_bench/pipeline.py      # all 命令和评估报告
mineral_bench/generate.py      # 固定随机种子的小合成样本
mineral_bench/selftest.py      # 内置断言
examples/                      # 可换入/参考的同格式输入
scripts/demo_minimal.sh        # 端到端演示
scripts/reproduce_all.sh       # 全量复跑
deliverables/                  # 最近一次生成的交付报告
```
