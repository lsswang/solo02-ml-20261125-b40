# 双语导览校稿与录音制作程序

纯 Python 3.10+ 标准库实现，不需要 numpy/pandas/网络或系统 TTS。它完成两件事：

1. 根据听辨试验记录，为中文、英文分别学习“称谓 → 候选人物”的完整概率分布，按整篇 `script_id` 留出，输出对数损失、两条实际误听记录和校稿结论。
2. 从展线、措辞、费用、时长输入出发，分别求解：
   - **墙机模式 `wall`**：只按“当前站 + 上一站门牌”选录音；同门牌汇入必须共用，并把唯一的“一站增录一版”机会用在最优处。
   - **语音笔模式 `pen`**：按“当前站 + 最近两个已访门牌”选录音；不同录音数量受笔容量约束，程序决定在哪里合并、把容量留给哪一版。

题中两种制作权限都实现了，便于比较；若只要求任选一条，直接看 `pen` 或 `wall` 均可。

## 30 秒复跑

```bash
cd /workspace
./run_demo.sh
```

脚本会重新生成四套本地输入、运行三个规定场景加两个附加场景、执行门牌回放和单元测试。

也可以手动运行：

```bash
python3 -m guide_studio generate-sample --out sample_input --scenario baseline
python3 -m guide_studio run --input sample_input --output out/baseline

python3 -m guide_studio replay \
  --plan out/baseline/plans/pen_zh.json \
  --plates east_gate,lin_hall,silent_porch,portrait
```

先阅读 `out/baseline/run_summary.md`。

## 目录

```text
guide_studio/
  bundle.py       # CSV/JSON 输入与校验
  model.py        # 分组留出、分层概率模型、全称/短称校稿
  proofread.py    # 整篇稿件校稿表
  optimize.py     # wall/pen 的精确枚举优化与状态转移
  audio.py        # 生成可播放 WAV 占位母带、字幕和 M3U
  replay.py       # 门牌串回放
  pipeline.py     # 端到端交付物
  sample.py       # 可复现合成样本生成器
  cli.py
sample_input/                 # 基准输入
sample_input_west_lin/        # 替换听辨材料：西路也指向林
sample_input_revised_script/  # 替换稿件：西路改用全称
sample_input_high_fee/        # 费用敏感性附加样本
tests/test_pipeline.py
out/                          # 最近一次生成的交付物
```

## 输入格式

一个输入目录包含：

| 文件 | 内容 |
|---|---|
| `config.json` | 种子、留出比例、平滑参数、概率门槛、笔容量、墙机增录站数 |
| `persons.csv` | `language,person_id,display_name`，每种语言的全部候选人 |
| `appellations.csv` | 称谓别名；`kind=full` 且有 `person_id` 时按全称唯一映射 |
| `listening_records.csv` | 听辨记录：记录号、整篇稿 ID、语言、站点、先前人物、听见措辞、实际指认、预定人物 |
| `manuscript_lines.csv` | 待校稿件 |
| `stations.csv` | 真实门牌站号 |
| `routes.csv` | 路线及给定权重 |
| `route_stops.csv` | 路线顺序、站点、该站预定人物 |
| `wordings.csv` | 每站 2–3 种措辞、录音费、时长、预定人物、播后最近人物状态 |

`prior_person_ids` 用 `|` 分隔，最后一个是最近讲过的人物。`state_person_id` 表示播完这版后台词使哪位人物成为“最近人物”；留空则保留进入该站时的状态。因此前站选择会影响后站短称概率。

### 馆方门槛

`config.json` 中：

```json
"short_form_threshold": {"zh": 0.8, "en": 0.8}
```

短称只有在：

1. 预定人物在完整分布中排名第一；并且
2. 预定人物概率 ≥ 该语言门槛

时才允许进入制作方案。全称直接映射唯一人物，不依赖统计概率。

## 概率模型与留出

程序按 `config.seed` 用 `random.Random(seed)` 打乱整篇 `script_id`，整组划入训练/测试，不会让同一篇稿的行同时出现在两侧。

概率是四级 Dirichlet 平滑混合模型：

- 语言候选人先验；
- `(语言, 措辞)`；
- `(语言, 措辞, 最近人物)`；
- `(语言, 措辞, 站点, 最近人物)`。

每一级都对该语言全部候选人给出非零概率，混合后归一化。`referent_model.json` 保存各级完整分布；每条校稿结论也带完整 `distribution`。

生产方案使用**全部记录重新拟合**的模型；`holdout_log_loss.json` 则只报告训练稿拟合后在留出整篇稿上的损失。默认样本结果约为：

- 全局对数损失：`0.1691 nats`；
- 中文、英文损失相同（合成数据按同构计数生成）；
- 两条重点误听：`R-ZH-0078`、`R-EN-0158`，中/英各一条；
- `two_mishearing_records.json/.csv`：两条实际指认≠预定人物的记录，含模型概率和单条损失；
- `mishearing_records_by_language.*`：每种语言各自最显著的两条。

同一数据、同一 `seed` 重跑会得到相同划分、概率、方案和记录选择。

## 优化目标与设备记忆

目标按语言独立计算：

```text
objective = Σ_路线 route_weight × time_cost_per_second × Σ_站 duration_sec
            + Σ_不同录音 recording_fee
```

默认 `time_cost_per_second=1`，路线权重为东路 `3`、西路 `2`。

### 墙机 `wall`

选择键：

```text
(language, current_station, previous_station_plate)
```

同一选择键下的所有路线出现必须使用同一版录音。示例中东西路都经过同一个 `silent_porch`，所以肖像站的上一站门牌相同，即使语音笔能区分更早来路，墙机在此也不能拆分。

本程序对“允许在一个站增录一版”的可执行解释是：

- 默认每站每语言一版；
- `wall_bonus_stations_per_language=1` 允许一个站有第二版，供不同上一站门牌选择；
- 同一门牌形成的组仍绝不能拆。

求解器会比较所有候选组合，自动决定增录站。

### 语音笔 `pen`

选择键：

```text
(language, current_station, second_previous_plate, previous_plate)
```

这里的“最近两站”指播放当前站前已访问的两个门牌；第一、二站缺失的记忆用 `∅` 表示。肖像站的两个键分别包含：

```text
(... portrait, prev2=lin_hall,  prev1=silent_porch)
(... portrait, prev2=zhou_hall, prev1=silent_porch)
```

所以笔能分辨东路、西路。每个不同录音资产占一个容量；默认 `pen_capacity_per_language=7`。

样本很小（6 站、每语言两路、最多 8 个记忆组），因此 `optimize.py` 使用确定性的 `itertools.product` 全枚举，而不是近似或贪心。

## 规定叙事的实际结果

以下数值来自当前生成数据；重新生成同一种子样本后应一致。

| 场景 | 墙机结果 | 语音笔结果 |
|---|---|---|
| 基准：东路语境指林，西路语境指周 | 每语言目标 `88`；肖像站两路共用 `*_portrait_full`；增录机会用于 `silent_porch` | 每语言目标 `87`，用满 7 个容量；东路肖像站用 `*_portrait_she`，西路仍用全称 |
| 关闭 `west_gate` | 每语言目标 `46`，只剩 4 段录音；肖像站用门槛合格的短称 | 同样用短称 |
| 换入 `west_lin` 听辨材料 | 目标 `80`；墙机肖像站共用稿自动重做为 `*_portrait_she` | 两路均用短称 |
| 附加：肖像短称增录费改为 10 | 仍共用全称 | 目标 `88`；省下来的容量改用于廊口分路版，肖像站两路全称 |
| 替换稿件，西路直接写全称 | `proofreading_report` 无拒绝项 | 同左 |

基准样本中肖像站两版费用相同（均为 4），全称时长 4 秒、短称 2 秒。因为东路权重为 3，短称节省的加权时长为 `3×2=6`，高于多录一版的增量费用 4，所以语音笔选择东路短称。高费用样本把该增量费用改为 10，程序便自动放弃。

前站状态依赖也有具体取舍：东门点名版 4 秒并留下 `last_person=lin`，使林岚站的“她”（2 秒）合格；通用东门版 3 秒但不留下人名，随后“她”达不到门槛，只能用 4 秒全称。两种组合的站间时长分别为 6 秒和 7 秒，所以最优回放会选择“东门点名 + 林岚站短称”。

## 回放一路

`--plates` 接受逗号分隔的真实门牌串。程序不使用路线名走后门，而是逐站根据设备记忆键从方案中取录音：

```bash
python3 -m guide_studio replay \
  --plan out/baseline/plans/wall_zh.json \
  --plates west_gate,zhou_hall,silent_porch,portrait
```

输出会逐站显示记忆、录音资产、时长和实际句子。流水线还自动生成：

- `out/baseline/replays/*.txt`：文本回放；
- `out/baseline/replays/*.json`：结构化回放；
- `out/baseline/replays/*.m3u`：播放列表；
- `out/baseline/audio/<lang>/*.wav`：与输入时长一致、可播放的轻声占位母带；
- 同名 `.txt` 字幕/台词文件。

仓库不依赖离线 TTS，所以 WAV 是确定性的提示音占位文件；真人录音或 TTS 文件可按 `recording_manifest.csv` 中的路径直接替换，保持文件名和时长即可。

## 验题人如何换入另一份输入

1. 按上述 CSV/JSON 格式准备一个新目录，例如 `another_input/`。
2. 运行：

   ```bash
   python3 -m guide_studio run --input another_input --output out/another
   ```

3. 若只替换听辨材料，保持其他文件不变，用同格式的新 `listening_records.csv` 覆盖后，可只重新拟合：

   ```bash
   python3 -m guide_studio fit --input another_input --output out/another_fit
   ```

   要同时重做录音方案和回放，则运行上面的完整 `run` 命令。
4. 若关闭某个门或站：

   ```bash
   python3 -m guide_studio run \
     --input another_input \
     --output out/closed \
     --close west_gate
   ```

5. 回放新方案：

   ```bash
   python3 -m guide_studio replay \
     --plan out/another/plans/pen_zh.json \
     --plates gate_a,station_b,portrait
   ```

新的 `run_summary.json/.md`、模型分布、误听记录、校稿表、方案、录音单和回放都会重新生成。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖：

- 两份误听记录确实是实际误听；
- 中英文指标和完整概率分布存在且归一；
- 基准墙机肖像站共用全称；
- 基准语音笔东路短称、西路全称且不超容量；
- 关闭西门后两种设备都用合格短称；
- 替换西路听辨材料后墙机共用短称；
- 门牌记忆回放正确；
- 同一种子结果可复现。

## 主要交付文件

```text
out/baseline/run_summary.md
out/baseline/referent_model.json
out/baseline/holdout_log_loss.json
out/baseline/two_mishearing_records.json
out/baseline/proofreading_report.csv
out/baseline/optimization.json
out/baseline/plans/wall_zh.json
out/baseline/plans/wall_en.json
out/baseline/plans/pen_zh.json
out/baseline/plans/pen_en.json
out/baseline/recording_manifest.csv
out/baseline/replays/
out/baseline/audio/
```
