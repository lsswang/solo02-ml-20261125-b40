# 老拼配与新豆批：咖啡工坊拼配试样 + 营业台账

这是一个从零写起、只依赖 Python 3.10+ 标准库的小型可复跑项目。它做两件事：

1. 从盲评记录学习**豆批 + 四分之一配比 + 豆间协同**对“像店内标准杯”概率的影响，并用未参与拟合的整次烘焙报告预测和 log loss。
2. 把模型概率接入一个本地、事件溯源、不可回写的咖啡店台账，支持整桶/单杯试样两种营业路线、扫码看称料、半桶续杯候选、缺料/质量拒单和余料查询。

> 题面允许两条营业路线二选一。主小店例使用**整桶路线**；为了核对题面指定的贝叶斯读数，也额外实现并运行了**单杯试样路线**。两套结果都在 `deliverables/` 下，便于理解差别。

## 一键复跑

```bash
bash run_demo.sh
# 或只跑断言：
python3 verify.py
```

`run_demo.sh` 会重新生成小型合成盲评数据、训练/留次评估、运行三种豆和三张订单的小店例、坏乙批换入例、单杯试杯两种读数分支、半桶输入例，并执行 38 项断言。

无需 `pip install`，普通 CPU 几秒内完成。

## 目录与关键交付物

```text
coffee_blend/
  synthetic.py      # 生成同格式合成盲评 CSV（好批次 / 坏新乙批）
  data.py           # CSV 读取、烘次-配方连通组划分与审计
  model.py          # 二次混合 logistic 模型 + 纯 Python Newton 拟合
  experiment.py     # 整烘次留出预测、log loss、Markdown 报告
  recommend.py      # 枚举四分之一配方并按库存算可做杯数
  shop.py           #  append-only 接单/开桶/试杯/出杯/扫码/余料
  pipeline.py       # 输入目录 -> 新交付物的一键流程
  cli.py            # python -m coffee_blend ...
input/              # 好甲、好乙、丙三种豆；主小店例（验题人可直接换入同格式文件）
input_bad_b/        # 数量不变，只换入对半拼配低于门槛的新乙批
deliverables/good/
  blind_evaluation.md                 # 整烘次盲评预测 + log loss
  model.json                           # 全量重拟合后的生产模型
  production_recommendations.json      # 配方门槛与可兑现杯数
  actual_successful_weighing_example.json
  pipeline_summary.md
  shops/01_bucket_three_orders/
    scan_O1_half_bucket_refills_O2.json
    scan_O2_actual_beans_from_bucket.json
    ledger.jsonl                       # 不可变事件流水
  shops/03_trial_same_order_like/
  shops/04_trial_same_order_unlike/
deliverables/bad_b/                   # 换入坏乙批后的新交付物
verify.py
```

## 验题人如何换入另一份输入

准备一个目录，至少包含同格式的 `sensory_votes.csv` 和 `inventory.json`；可参考 `input/plan.json` 增加自动营业动作。然后：

```bash
./run_pipeline.sh /path/to/new_input /path/to/new_output
# 等价于：
python3 -m coffee_blend pipeline \
  --input-dir /path/to/new_input \
  --out-dir   /path/to/new_output
```

`run_pipeline.sh` 不会重新生成或覆盖换入的输入；`run_demo.sh` 才会重建随仓样本。

程序会在新输出目录重新生成：

- `blind_evaluation.json/.md`：留出整烘次的预测与 log loss；
- `model.json` / `blind_model.json`：生产模型和仅用训练划分拟合的模型；
- `split_audit.json`：烘次、配方、评尝轮次是否被拆裂的审计；
- `production_recommendations.json`：每个四分之一配方的概率、是否过 `0.80` 门槛、库存可做杯数；
- `shops/<run>/`：每单决策、扫码 JSON、最终库存和 append-only `ledger.jsonl`；
- `actual_successful_weighing_example.json`：一次真实落账的成功称料。

即使新 CSV 的烘次/配方连接关系导致严格连通组无法留出训练集，程序也会退化为整烘次留出，并在 `split_audit.json` 写明策略和原因，而不是静默随机拆票。

### 盲评 CSV 格式

规范格式是**一位评委一行一票**：

```csv
vote_id,roast_id,round_no,formula_id,judge_id,composition,vote
V00001,R-BLIND-11,1,F-J2-Y2,J01,JIA-2409:2|YI-2409:2,1
V00002,R-BLIND-11,1,F-J2-Y2,J02,JIA-2409:2|YI-2409:2,0
```

- `composition` 用 `批次ID:份数|批次ID:份数`；每份是四分之一，因此总份数为 4。
- `vote=1/similar/像` 表示像标准杯，`0/different/不像` 表示不同。
- 同一 `formula_id` 的多名评委、同一 `roast_id` 的整次烘焙、同一 `(roast_id, formula_id, round_no)` 的多轮盲尝都被保留。
- 也支持汇总计数列 `similar_votes,dissimilar_votes`，以及 `parts_<BATCH>` 宽列。

重新生成样本：

```bash
python3 -m coffee_blend generate-data --out input/sensory_votes.csv --scenario good --seed 231125
python3 -m coffee_blend generate-data --out input_bad_b/sensory_votes.csv --scenario bad_b
```

## 模型：为什么单喝不接近，混起来可以接近

使用带 L2 的二次混合 logistic 模型：

\[
\operatorname{logit}P(Y=1)
= \alpha+\sum_b x_b\beta_b+\sum_{b<c}x_bx_c\gamma_{bc},
\qquad x_b=\frac{\text{batch parts}}{4}.
\]

- `beta_b` 学习豆批/单喝倾向；
- `x_b` 学习配比变化；
- `gamma_{JIA,YI}>0` 是甲乙协同项，使甲乙单喝都低、`2:2` 对半时反而高。

样本结果（固定随机种子，可能因代码中的小数值优化有末位差异）：

- 甲单喝预测相似率约 `0.08–0.13`；
- 乙单喝约 `0.08–0.20`；
- 甲 2 + 乙 2 约 **`0.84`**，超过 `0.80` 门槛；
- `3:1`、`1:3` 均低于门槛；
- 盲评整烘次 `R-BLIND-11` 及其重烘连通次 `R-BLIND-12` 均未参与拟合，逐票 log loss 约 `0.37`，常数基线约 `1.27`。

### 数据划分如何满足“同配方评委同划分”

`data.py` 把每个 `roast_id` 和每个 `formula_id` 作为图节点，每票把所属烘次和配方连边，再用并查集取连通块。留出包含指定烘次的连通块，因此样本中：

- 该烘次的所有票都在盲评；
- 该配方在另一烘次的重烘也整体在盲评；
- 同一配方的所有评委不会跨训练/盲评；
- 审计字段 `whole_roast_violations=[]`、`same_formula_split_violations=[]`。

生产 `model.json` 在诚实评估完成后用全部记录重拟合；`blind_model.json` 才是未看留出烘次的模型。

## 营业规则

### 概率、门槛与不合格损失

示例配置在 `input/inventory.json`：

```json
{
  "gate_threshold": 0.80,
  "trial_accuracy": 0.90,
  "good_cup_reward": 2.0,
  "bad_cup_penalty": 8.0
}
```

每杯期望分为：

\[
E[U]=2p-8(1-p).
\]

因此 `p>=0.80` 时期望分非负，和题面的八成接单门槛一致。接单输出同时给出：

- 从散豆最多可称多少杯；
- 可匹配的已开桶/半桶；
- 能否开整桶（整桶全部份料立即占用）；
- 当前单和整桶的期望分；
- 开桶、用旧桶、质量拒单或缺料拒单的理由。

### 路线 A：整桶事先拼好

主例从两个空桶 `KEG-1`、`KEG-2` 开始；三种豆的散豆库存均为 4 份，每杯 4 份。只有 `JIA-2409:2|YI-2409:2` 过门槛，库存恰好支持两杯。

`plan.json` 自动执行：

1. `O1`：p≈0.843，开 `KEG-1`（2 杯量），立即扣掉甲 4 份、乙 4 份，称料为甲 60 g + 乙 60 g；
2. `O2`：从同桶预留第二杯，配比相同；
3. 出 `O1` 后，`KEG-1` 余 1 杯，即容量的一半；扫码 `O1` 会显示这半桶可续给 `O2`；
4. 出 `O2`，桶空；
5. `O3` 报 `rejected_material`，不会无中生有。

查这个已跑完的状态：

```bash
python3 -m coffee_blend scan \
  --state deliverables/good/shops/01_bucket_three_orders --id O1

python3 -m coffee_blend stock \
  --state deliverables/good/shops/01_bucket_three_orders
```

新开一个可交互状态：

```bash
python3 -m coffee_blend init-shop \
  --state /tmp/coffee-shop \
  --inventory input/inventory.json \
  --model deliverables/good/model.json --reset

python3 -m coffee_blend order \
  --state /tmp/coffee-shop \
  --id O1 --composition 'JIA-2409:2|YI-2409:2' \
  --roast R-DEMO --mode bucket

python3 -m coffee_blend serve --state /tmp/coffee-shop --id O1
python3 -m coffee_blend scan  --state /tmp/coffee-shop --id O1
python3 -m coffee_blend stock --state /tmp/coffee-shop
```

### 坏新乙批：数量不动，第一单拒售

`input_bad_b/` 把 `YI-2409` 换成新到的 `YI-2418`，各批份数仍是 4。换入新 CSV 重拟合后：

- 对半拼配 `JIA2:YI-NEW2` 的预测概率约 `0.17`，低于 `0.80`；
- 没有任何库存可做且过门槛的配方；
- 第一张单 `B1` 为 `rejected_quality`，不开桶、不扣豆；
- 最终库存仍为甲 4、新乙 4、丙 4。

这对应题面“第一单就要更换配方或拒售”；本样本选择明确拒售并留料，`production_recommendations.json` 说明没有可直接更换的过门槛配方。

### 提走三份甲，两杯承诺失效

独立演练 `02_three_jia_withdrawal_invalidates_two_cups` 从原始 4 份甲中 append 一笔提料/盘减 3 份，甲剩 1 份。此后 2 杯订单 `COMMIT2` 需要 4 份甲，因此报 `rejected_material`。

系统不会删改原场景或偷偷改已售配方；提料是新事件。对于已接受但尚未出杯的单，盘减事件会把它们标成 `unfulfillable_after_stocktake`，原订单和历史事件仍保留。演练 `02b_two_commitments_then_three_jia_stocktake` 先留下 P1/P2 两个承诺，再盘减三份甲；两张单不被改写删除，而是保留原配方并标记为无法兑现，继续出杯会被拒绝。

### 路线 B：单杯试样与两个可能读数

题面指定的先验和试杯可靠性写在 `input/inventory.json` 的 `R-HALF-PRIOR/F-J2-Y2`：

- 试杯前合格概率：`0.5`；
- 对称信号准确率：`0.9`，即认对概率 90%、认错概率 10%；
- 接单门槛：`0.8`。

贝叶斯更新：

\[
P(Q\mid \text{like})=\frac{0.5(0.9)}{0.5(0.9)+0.5(0.1)}=0.9,
\]

\[
P(Q\mid \text{unlike})=\frac{0.5(0.1)}{0.5(0.1)+0.5(0.9)}=0.1.
\]

同一订单 `T1` 在两个独立、同初始状态的目录中分别遇到两种读数：

```bash
# 像：后验 0.9，接单并出杯
cat deliverables/good/shops/03_trial_same_order_like/actions.json

# 不像：后验 0.1，拒单；试杯用掉的 2 甲 2 乙不回滚
cat deliverables/good/shops/04_trial_same_order_unlike/actions.json
```

像分支：试杯耗 2 甲 2 乙，读数后接单，正式杯再耗 2 甲 2 乙，最终甲/乙均为 0。  
不像分支：拒单，试杯耗料保留，剩余甲/乙各 2 份不会被回补。

每个班次只允许一次试杯；第二次 `trial_prepared` 会被拒绝。读数写入 `trial_states["<roast>||<formula>"]`，只影响之后对同烘次、同配方的概率查询。

### 已称好的半桶

`inventory.json` 支持：

```json
"weighed_buckets": [{
  "bucket_id": "KEG-HALF-INPUT",
  "capacity_cups": 4,
  "remaining_cups": 2,
  "roast_id": "R-PREWEIGHED",
  "formula_id": "F-J2-Y2",
  "composition": "JIA-2409:2|YI-2409:2"
}]
```

已称桶的豆视为已占用，不计入 `loose_stock_parts`。运行：

```bash
python3 -m coffee_blend.extra_demos
```

会生成 `deliverables/good/preweighed_half_bucket_input_scan.json`，其中 4 杯桶余 2 杯（恰好半桶），并列出可续给的订单 `REFILL-2`。

## 不可回写保证

`ShopService` 的权威状态是每个营业目录下的 `ledger.jsonl`：

- 只 append，不提供修改/删除订单和称料的命令；
- 开桶事件立即记录整桶扣料；
- 试杯称料在读数前就落账，坏读数也不回滚；
- 已 `served` 的订单再次 `serve` 会报错，且不会新增事件；
- `state.json` 只是可由 JSONL 重放的快照。

## 直接可用的 CLI 命令

也可以用根目录 `python3 main.py ...` 或包装脚本 `./coffee ...` 代替 `python -m coffee_blend`。

```text
python -m coffee_blend generate-data ...
python -m coffee_blend pipeline ...
python -m coffee_blend init-shop ...
python -m coffee_blend order ...
python -m coffee_blend trial ...
python -m coffee_blend serve ...
python -m coffee_blend scan ...
python -m coffee_blend stock ...
python -m coffee_blend withdraw ...
python -m coffee_blend recommend ...
```

查看参数：

```bash
python -m coffee_blend --help
python -m coffee_blend order --help
```

## 给验题人的最小验收路径

```bash
bash run_demo.sh
```

末尾应出现：

```text
ALL 38 CHECKS PASSED
```

然后可检查：

1. `deliverables/good/blind_evaluation.md` 有整烘次 log loss；
2. `actual_successful_weighing_example.json` 是甲、乙各 4 份（各 60 g）的成功整桶称料；
3. `scan_O1_half_bucket_refills_O2.json` 显示 `"is_half_bucket": true` 和 `O2`；
4. 坏乙批输出中第一张单因质量被拒且库存不变；
5. 两个试杯分支的后验分别恰为 `0.9`、`0.1`，决定分别为接受、拒绝。
