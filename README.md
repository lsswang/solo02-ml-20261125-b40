# 双开关词板：训练、排卡、预置扫描与回放

这是一个只依赖 Python 3.10+ 标准库的小型可复跑项目。它从空目录实现了：

1. 家庭 / 课堂两个场景各自的带平滑一阶词转移模型；
2. 对“只记得选中过某组词”的模糊记录，枚举全部相容词序；
3. 在完整留出句上比较一阶转移模型和直接按词频猜测；
4. 从任意当前词查询后继概率；
5. 最多六词、每页两格、只能向前循环翻页的模拟词板；
6. 两种方案的微型枚举和期望成本优化：
   - **主方案：固定词卡 + 预置起扫格规则**；
   - **对照方案：可换词卡 + 进课堂花 4 拍熟悉一对互换生词**；
7. 按会话号持久化实际页位，关闭程序后可恢复；
8. 对使用者实际新句做逐拍动作回放。

## 快速运行

```bash
# 1. 题设最后给出的微型校准；会逐项核对选择、等待、翻页、调整费用
python3 -m wordboard check

# 2. 生成确定性合成输入
python3 -m wordboard generate --seed 2026 --out inputs/sample.json

# 3. 训练、评估、排卡、优化规则、回放并写报告
python3 -m wordboard all --input inputs/sample.json --outdir outputs/sample

# 查询后继概率
python3 -m wordboard query --input inputs/sample.json --scene school --current 丁

# 不修改 JSON，直接回放一条新句
python3 -m wordboard replay \
  --input inputs/sample.json --outdir outputs/cli-replay \
  --session demo-1 --scene home --sentence 甲 乙 丙 --close-after 3

# 4. 测试
python3 -m unittest discover -s tests -v
```

也可以直接执行：

```bash
./run_demo.sh
```

打开 `outputs/sample/report.md` 查看中文报告，或查看 `outputs/sample/summary.json` 获取机器摘要。

## 换入另一份输入

生成另一个种子的同格式输入并运行：

```bash
python3 -m wordboard generate --seed 42 --out inputs/another.json
python3 -m wordboard all --input inputs/another.json --outdir outputs/another
```

验题人也可以直接替换自己的 JSON：

```bash
python3 -m wordboard all --input /path/to/input.json --outdir /path/to/out
```

输入 schema 为 `wordboard-input-v1`，样例见 `inputs/sample.json`。最小结构如下：

```json
{
  "config": {
    "alpha": 1.0,
    "page_turn_cost": 3,
    "select_cost": 1,
    "wait_cost": 1,
    "toggle_cost": 0.5,
    "card_swap_cost": 4
  },
  "start_page": 1,
  "board": {
    "num_pages": 2,
    "cards": [
      {"word": "甲", "page": 1, "slot": 1, "familiar": true, "fixed": true}
    ]
  },
  "corpus": {
    "home": {
      "sentences": [["甲", "乙", "丙"]],
      "groups": [["乙", "丙"]]
    },
    "school": {
      "sentences": [["甲", "丁", "戊"]],
      "groups": [["丁", "戊"]]
    }
  },
  "heldout": {
    "home": [["甲", "乙"]],
    "school": [["丁", "戊"]]
  },
  "tasks": {
    "home": ["甲", "乙", "丙"],
    "school": ["丁", "戊", "己"]
  },
  "query": {"scene": "home", "current": "甲"},
  "replays": [
    {
      "session_id": "s1",
      "scene": "home",
      "start_page": 1,
      "sentence": ["甲", "乙"],
      "close_after": 1
    }
  ]
}
```

`tasks.home` 和 `tasks.school` 各为一段 1–3 词的任务；也支持每场景多段任务（JSON 中为句子数组的数组）。

## 概率模型

### 相容词序

- `sentences` 是已知完整词序；
- `groups` 是只知道选中过哪些词的旧记录；
- 对每个 group 调用 `itertools.permutations` 给出所有不同词序，重复词只保留唯一排列。

默认 `config.ambiguity_weighting = "uniform"`：一条旧记录的质量平均分给所有相容词序。设某组有 \(m\) 个相容顺序，则每个顺序权重为 `记录权重 / m`。如果希望把每个相容顺序都完整计一次，可设置：

```json
"ambiguity_weighting": "each_order"
```

所有枚举结果写入 `models.json` 的 `compatible_order_enumerations`。

### 平滑转移

两个场景使用同一份词表、分别计数：

\[
P(w_i\mid w_{i-1},s)
=
\frac{c_s(w_{i-1},w_i)+\alpha}
     {c_s(w_{i-1})+\alpha |V|}.
\]

句首使用 BOS 计数和同样的 add-\(\alpha\) 平滑。默认 \(\alpha=1\)（Laplace），可在 `config.alpha` 修改。

### 留出句比较

`evaluation.json` 对每句输出逐词比较，并汇总：

- 一阶转移模型 Top-1 准确率；
- 直接猜场景最高频词的 Top-1 准确率；
- 总负对数似然和困惑度。

合成数据刻意让家庭和课堂具有不同转移偏好，因此通常能看到转移模型优于固定词频猜测。

### 查询后继概率

```bash
python3 -m wordboard query \
  --input inputs/sample.json \
  --scene school \
  --current 丁
```

省略 `--current` 时查询平滑后的句首分布。

## 模拟词板与费用

内部页、格均为 0-based；JSON 和报告中显示为 1-based。

设当前页为 \(p\)，目标词在第 \(q\) 页，板上共 \(P\) 页，则只向前循环的翻页次数为：

\[
d=(q-p)\bmod P.
\]

若词表大于板上六词，规划时使用条件分布 \(P(w\mid u,w\in B)\) 并重归一；每个上下文被板覆盖的原始概率质量 `on_board_probability_mass` 会写入 `plan.json`，避免悄悄丢弃模型质量。

一次选词的完整费用拆成：

\[
\text{cost}
=
dC_{\text{page}}
+
\mathbf 1[\text{预置到非默认格}]C_{\text{toggle}}
+
\mathbf 1[\text{目标格不是扫描起点}]C_{\text{wait}}
+
C_{\text{select}}.
\]

默认参数对应题设校准：

- 选词：1 拍；
- 等到另一格：加 1 拍；
- 翻一页：3 拍；
- 进课堂熟悉一对互换生词：4 拍；
- 拨到非默认起扫格的费用由输入的 `toggle_cost` 给出（样例用 0.5 拍演示）。

默认设备行为是“翻页后落在该页首格”，由 `config.page_turn_resets_slot = true` 表示。若真实设备翻页后仍保留两格轮扫头的相位，把它设为 `false`，费用枚举会改用预置格作为翻页后的起扫格。

## 两条路线

### 主方案：固定词卡 + 预置规则

所有词位全程不动。主优化目标直接使用输入的两段任务：

- 每段第一个任务词按给定页位精确计费；
- 之后每个待选词，规则只能读取该段中**已经选中的前一个词**和当前场景；
- 起扫格的优劣按 learned \(P(\text{下一词}\mid\text{刚选的词},s)\) 对整块板枚举求期望；
- 家庭任务结束后的实际页位带到课堂，作为课堂第一个词的起始页。

此外，`plan.json` 还给出一个“纯模型滚动”目标：只使用任务长度和句首/转移分布传播状态，不使用具体任务词。两种目标的差别都保留在交付物中。

对每个场景 \(s\) 和每个可能刚选完的词 \(u\)，枚举两个起扫格：

- 首格（默认，无调整费）；
- 另一格（付 `toggle_cost`）。

规则只接收：

```python
PresetRule.start_slot(scene, previous_word)
```

它没有当前目标词参数，因此不会把“正在说的词”提前透露给规则。离线优化时使用的是：

\[
\mathbb E_{w\sim P(\cdot\mid u,s)}[\text{action\_cost}(w)],
\]

而不是回放句中的实际下一词。

无翻页、两格分别为首格/另一格时，拨到另一格的盈亏平衡概率为：

\[
P(\text{另一格}) > \frac12+\frac{C_{\text{toggle}}}{2C_{\text{wait}}}.
\]

一般有翻页、循环页和多页的情形不靠这个简化公式，而是对板上所有词逐一计算分项费用后取期望。

### 对照方案：可换词卡

家庭排卡可自由优化；进入课堂时允许：

- 不换卡；或
- 仅交换一对非熟词/新生词，并支付一次 `card_swap_cost = 4`。

途中不再换卡。程序枚举所有合法家庭布局和所有可换词对，报告：

- 最有利的一对词；
- 课堂任务的毛节省；
- 扣除 4 拍后的净收益；
- 若每次进课堂都重新付费，需要多少次课堂任务才能盈亏平衡。

如果四拍是一次性熟悉成本而非每次进入成本，只需判断首次净收益是否大于 0。

## 页位和会话恢复

`SessionStore` 把每个会话写入 JSON。每次选词后，实际页位更新为目标词所在页并立即落盘。

回放中的 `close_after` 会：

1. 关闭会话；
2. 用一个**新的 store 对象**从磁盘重新读取；
3. 断言恢复页等于实际目标页；
4. 再继续或保持关闭。

命令行手动恢复：

```bash
python3 -m wordboard resume \
  --state outputs/sample/sessions.json \
  --session one-word-from-page-1
```

换场调用 `switch_scene` 时只改场景，不改页码，因此页码按题设保留。

## 微型校准

`python -m wordboard check` 构造“甲锁第一页首格、两页、翻页 3 拍”的三板词场景，并枚举全部 6 种乙/丙摆法。它验证：

- 第一页开始、不预调，\(P(乙)=0.8,P(丙)=0.2\)：乙在第一页另一格；
- 交换概率后：该格留给丙；
- 第二页开始且保持原概率：乙在第二页首格；
- 乙在第一页另一格：等待 1 + 选择 1 = 2 拍；
- 丙在第二页首格：翻页 3 + 选择 1 = 4 拍；
- 预置另一格时，调整费、等待减免和选择费分别记录。

完整候选表写到 `outputs/calibration.json` 或 `all` 输出目录中的 `calibration.json`。

若要换入新的概率而不改代码，可在输入 JSON 中提供 `probability_scenarios`：

```json
"probability_scenarios": [{
  "id": "my-case",
  "num_pages": 2,
  "start_page": 1,
  "locked_word": "甲",
  "distribution": {"乙": 0.3, "丙": 0.7}
}]
```

`all` 会枚举该场景的全部摆法并写出 `probability_scenarios.json`。

## 输出文件

一次 `all` 运行产生：

| 文件 | 内容 |
|---|---|
| `report.md` | 给使用者/验题人阅读的完整中文报告 |
| `summary.json` | 机器摘要 |
| `models.json` | 两场景计数、平滑概率、相容词序枚举 |
| `evaluation.json` | 留出句逐词预测和指标 |
| `plan.json` | 两条路线的布局、规则、换卡决策 |
| `layout_enumeration.csv` | 固定卡路线的全部布局得分 |
| `calibration.json` | 微型枚举所有候选和分项费用 |
| `query_example.json` | 示例后继概率查询 |
| `replay_fixed.json` | 固定卡路线的逐拍回放 |
| `replay_swappable.json` | 可换卡路线的逐拍对照 |
| `sessions.json` | 固定卡路线的会话页位状态 |
| `sessions-swappable.json` | 可换卡路线的会话状态 |

## 代码结构

```text
wordboard/
  model.py        # 相容词序枚举、平滑模型、留出评估
  board.py        # 词板位置、布局枚举、分项费用
  planner.py      # 期望成本 DP、排卡和预置格优化
  replay.py       # 不窥视目标词的规则与实际句回放
  session.py      # JSON 会话持久化和页位恢复
  calibration.py  # 题设微型枚举断言
  generate.py     # 确定性合成输入生成器
  pipeline.py     # 端到端输出和 Markdown 报告
  cli.py          # 命令行
tests/            # 单元测试
inputs/           # 样例输入
```
