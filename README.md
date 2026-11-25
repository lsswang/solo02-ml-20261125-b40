# 两只手都碰过顶点，这一趟抱石算完攀吗？

一个只依赖 Python 标准库的小型、可复跑裁判程序。它把每一帧的两只手腕读数建成
**联合隐状态** `(左手触点, 右手触点)`，学习：

- 9 个联合触点状态之间的转移概率；
- 条件于联合状态的 **成对读数误差** `P(左读数, 右读数 | 左触点, 右触点)`，因此身体同时遮挡、双手相关误识别不会被错误假设成两只手独立出错。

程序随后在整段留出短片上核对联合触点分布；对新片输出左手+右手曾在哪些帧同触顶点、逐帧后验证据，以及 `TOP / ZONE / NO_ZONE` 三类全轨迹概率。任一类概率达到 `0.95` 才正式签单，否则状态为可申诉。

本实现选择的申诉档案路线是：**允许留存腕部读数的场馆**。同一申诉号下保存原始成对读数、路线卡、初始模型快照、初始判单、历次校准模型和改判记录；赛后校准变更时先从旧读数和旧模型复现旧判，再用新校准重审。

---

## 1. 一键复跑

需要 Python 3.10+；参考环境为 Python 3.11，无需安装第三方包。

```bash
bash scripts/run_demo.sh
```

该脚本会确定性地完成：

1. 生成不超过 16 帧的合成训练片、整段留出片、新片和申诉片；
2. 运行四个可手算小例（严格断言期望值）；
3. 训练 `model_v1` 和赛后修订校准 `model_v2`；
4. 输出留出片联合分布核对；
5. 对新片签单；
6. 对 `APP-2026-0001` 从交片、初判 `TOP` 到申诉改判 `ZONE`；
7. 运行单元测试。

主要交付物：

```text
deliverables/hand_examples.md                 # 手算概率核对
deliverables/heldout_report_v1.md             # 整段留出片联合分布
deliverables/new_decision_v1.md               # 一次新片的裁判结果
cases/APP-2026-0001/DOSSIER.md                # 同一申诉号的初判与改判链
cases/APP-2026-0001/dossier.json              # 机器可读完整档案
```

当前固定种子的参考结果为：

- 手算四例全部复现：`0.5 / 0 / 1 / 0`；
- 留出片联合状态分布 TV ≈ `0.061`；
- `public-new-W5-014`：`TOP≈0.9539`，达到 0.95，签 `TOP`；
- `APP-2026-0001` 初判：`TOP≈0.9817`，签 `TOP`；
- 新校准发现相关假 `TOP/TOP` 读数后，旧判复现误差为 `0`，重判 `ZONE≈0.9980`，改签 `ZONE`。

概率末位可能随不同 Python/平台的浮点求和有极微小变化，但结论和阈值跨越稳定。

---

## 2. 换入另一份相同格式输入

### 2.1 只换一部新片

如果新片仍使用同一组顶点/读数词表：

```bash
scripts/judge_another_clip.sh /path/to/another_clip.json /tmp/another_decision
```

或直接：

```bash
python3 -m boulder_judge.cli judge \
  --clip /path/to/another_clip.json \
  --model models/model_v1.json \
  --route data/route_card.json \
  --out-json /tmp/another_decision.json \
  --out-md /tmp/another_decision.md
```

输入可以是单个 clip 对象，也可以是含 `clips: [...]` 的语料并加 `--clip-id`。

最小新片格式见 `data/same_format_template.json`：

```json
{
  "format": "boulder-clip/v1",
  "clip_id": "replace-me",
  "frames": [
    {"frame": 0, "left_reading": "NONE", "right_reading": "NONE"},
    {"frame": 1, "left_reading": "TOP",  "right_reading": "TOP"}
  ]
}
```

未知字段会被保留；裁判使用 `frame` 数组顺序作为零基帧号。若使用了模型没见过的读数符号，程序会明确报错；此时应先用含该读数的人工标注语料重新训练。

### 2.2 换一个新种子的完整合成材料包

```bash
scripts/rerun_fresh_seed.sh 31337 /tmp/boulder_fresh_31337
```

该目录会得到新的语料、路线卡、模型、留出报告和新片判决。验题人也可以把 `31337` 换成任意整数，或按同样 JSON schema 提供自己的语料。

### 2.3 换一份新的标注语料后训练

```bash
python3 -m boulder_judge.cli train \
  --data /path/to/new_corpus.json \
  --out /tmp/new_model.json \
  --version my-calibration

python3 -m boulder_judge.cli heldout \
  --data /path/to/new_corpus.json \
  --model /tmp/new_model.json \
  --route data/route_card.json
```

语料格式见 `data/synthetic_corpus.json`。每个训练帧必须同时包含：

```json
{
  "frame": 7,
  "left_contact": "TOP",
  "right_contact": "TOP",
  "left_reading": "OCCLUDED",
  "right_reading": "OCCLUDED"
}
```

顶层 `vertices` 和 `reading_values` 可替换为另一场馆词表；程序会构造相应大小的联合状态和成对读数表。

---

## 3. 路线卡如何独立决定分数

示例路线卡 `data/route_card.json`：

```json
{
  "route_id": "W5-qualifier",
  "top_vertex": "TOP",
  "zone_vertex": "ZONE",
  "stable_frames": 2,
  "deadline_frame": 11,
  "frame_index_base": 0,
  "zone_requires": "any_hand",
  "threshold": 0.95
}
```

规则实现为：

1. **TOP（完攀）**：存在一段至少 `stable_frames` 长的连续帧，每帧联合状态都是 `(TOP, TOP)`，且这段稳定序列在 `t <= deadline_frame` 时已经达成。
2. **ZONE（区点）**：不满足 TOP，但截止前任意一只手到过 `ZONE`。
3. **NO_ZONE（无区点）**：以上皆否。

截止帧之后的读数仍可通过 HMM 前后向推理帮助回顾截止前状态，但不会再设置 TOP/ZONE 规则标记。

---

## 4. 概率计算

设联合状态为 \(s_t=(L_t,R_t)\)，成对读数为 \(o_t=(r^L_t,r^R_t)\)。模型为：

\[
\pi_s=P(s_0=s),\qquad
A_{ij}=P(s_{t+1}=j\mid s_t=i),\qquad
E_{j,o}=P(o_t=o\mid s_t=j).
\]

训练直接使用人工标签计数并做加性平滑：

- 初始联合状态计数；
- 联合状态转移计数；
- 每个联合状态到每个**读数对**的发射计数。

推理包括：

- log-space forward-backward，给出每帧 \(P(L_t=\text{TOP},R_t=\text{TOP}\mid o_{0:T})\)；
- Viterbi 最可能完整联合轨迹；
- 一个带 `(当前状态, 当前连续 TOP/TOP 长度, 已达成有效 TOP, 已到区点)` 的精确 DP，把所有完整路径质量汇总到三类分数；
- 另一个布尔 DP 计算“全片/截止前是否曾有任意一帧双手同顶”的事件概率。

所有演示片段最多 16 帧；状态空间为 \(3^2=9\)，因此这些 DP 在普通 CPU 上瞬时完成，且不是抽样估计。

---

## 5. 题目要求的手算小例

运行：

```bash
python3 -m boulder_judge.cli handcheck --strict
```

| 小例 | 手算路径 | K=1 时 TOP 概率 |
|---|---|---:|
| 一帧中 `HH` 与 `EE` 各半 | \(P(HH)=P(EE)=0.5\)，只有 HH 完攀 | **0.5** |
| 四帧始终恰有一手握 | 两条路径 `EH,HE,EH,HE` 与 `HE,EH,HE,EH` 各 0.5，均无 HH | **0** |
| 先空、末帧双手确定同握，K=1 | 唯一路径 `EE -> HH` | **1** |
| 同上，稳定要求改为 K=2 | 没有连续两帧 HH | **0** |

前两例中每只手的边缘触顶概率都是 \(0.5\)，但联合状态不同，因此完攀概率分别为 `0.5` 和 `0`。这正是不能只建模两个手腕边缘概率的原因。

---

## 6. 新片会指出哪些帧、证据多强

见 `deliverables/new_decision_v1.md` 或任意判决 JSON 的 `frame_evidence`。每帧给出：

- `p_both_at_top_vertex`：左手和右手同时在指定顶点的后验概率；
- `evidence_band`：`signing_strength / strong_below_signing / ambiguous / weak / effectively_absent`；
- 完整 9 状态后验分布；
- 该帧实际成对读数；
- 后验最高联合状态及概率。

同时汇总：

- `high_confidence_simultaneous_top_frames`：单帧概率达到 0.95 的帧；
- `probable_simultaneous_top_frames`：概率 ≥0.5 的候选帧；
- `p_any_simultaneous_top_in_clip`；
- `p_any_simultaneous_top_before_deadline`。

注意：正式 0.95 阈值签的是**整条轨迹的类别概率**。某部片子可能没有任何单帧达到 0.95，但连续路径证据使 `TOP` 类别概率超过 0.95；参考新片正是这种情况。

---

## 7. 一次从交片到实际申诉的完整流程

```bash
# 初判并建立同一申诉号
python3 -m boulder_judge.cli judge \
  --clip data/appeal_attempt.json \
  --model models/model_v1.json \
  --route data/route_card.json \
  --case-root cases \
  --appeal-number APP-2026-0001

# 赛后校准变更后申诉
python3 -m boulder_judge.cli appeal \
  --case-dir cases/APP-2026-0001 \
  --new-model models/model_v2_revised.json \
  --reason "新标注批次发现相关假 TOP/TOP 成对读数"
```

档案目录：

```text
cases/APP-2026-0001/
├── DOSSIER.md
├── dossier.json
├── submission/clip.json                 # 保留的原始腕部读数
├── route_card.json                       # 独立路线卡
├── models/initial_model.json             # 初始模型快照
├── models/appeal-01-model.json           # 新校准模型
├── decisions/initial_decision.json
├── decisions/appeal-01-decision.json
└── appeals/appeal-01.json
```

申诉时程序强制做两件事：

1. 用封存的初始模型和保留读数重算初判，核对模型指纹、签单类别和概率差；参考结果的最大概率差为 `0.0`；
2. 用新校准重审，并把旧状态、新概率、处置（`revised` / `upheld` / `unsigned_after_review`）追加到**同一个申诉号**。

固定演示中的变化原因是：新的人工标注校准片显示，该追踪器版本会在真实状态为 `ZONE/ZONE` 时双手相关地一起报成 `TOP/TOP`。因此旧校准把连续四个 `TOP/TOP` 读数解释为顶点稳定；新联合误差表把同一组成对读数改判为区点证据。

---

## 8. 另一种“当日销毁读数”场馆的可运行对照

主申诉档案选择留存读数路线；仓库也提供一个**可选对照命令**，把另一条路线的关键证据实际算出来：

```bash
python3 -m boulder_judge.cli sealed-preview \
  --clip data/appeal_attempt.json \
  --model models/model_v1.json \
  --route data/route_card.json \
  --testimony-a data/sealed_testimony_a_top.json \
  --testimony-b data/sealed_testimony_b_zone.json \
  --bundle-json deliverables/sealed_evidence_bundle.json \
  --out-json deliverables/sealed_preview.json \
  --out-md deliverables/sealed_preview.md
```

若当日必须销毁原始读数，可在判决时导出并封存：

\[
F_t(s)=P(o_t\mid s_t),\qquad t=0,\ldots,T
\]

即**每一帧、每个联合触点状态的读数似然向量**，外加封存的转移矩阵、初始分布和路线卡。`deliverables/sealed_evidence_bundle.json` 不含 `left_reading/right_reading` 原字段；这些联合因子加 HMM 已足以重跑全部 DP。参考输出中，封存因子重判概率与初判完全相同（TOP≈0.9817），两条逐帧证词在第 6–9 帧抵触，封存模型下 TOP 证词相对 ZONE 证词的 log Bayes factor ≈ 10.39。

之后任意一条完整逐帧触点证词 \(\tau=(s_0,\ldots,s_T)\) 都可按封存模型评分：

\[
\log W(\tau)=
\log\pi_{s_0}+\log F_0(s_0)
+\sum_{t=1}^{T}\left[\log A_{s_{t-1},s_t}+\log F_t(s_t)\right].
\]

两条相抵触证词可比较 \(W(\tau_1)\)、\(W(\tau_2)\) 或相对总证据的后验路径概率，并逐帧列出似然比最大的冲突帧。因为保存的是 \(F_t(s)\) 而不是 `(left_reading,right_reading)`，原始离散读数可以按规销毁；但这些因子加 HMM 已足以重跑全部 DP 和类别汇总。

与本仓库选择的留存读数路线相比：

| 能力 | 留存腕部读数（本实现） | 当日销毁读数（替代路线） |
|---|---|---|
| 赛后换新校准 | 可直接用原始读数和新模型重审 | 只能用封存因子配合封存/指定模型；新读数误差表无法从已销毁读数重新提取 |
| 复现旧判 | 旧模型快照 + 原始读数 | 封存因子 + 封存模型 |
| 支持任意完整证词 | 天然支持，可重新计算所有发射概率 | 保存逐状态联合似然因子后即可评分 |
| 解释证词冲突 | 可同时展示原读数和因子 | 只能展示因子比值和逐帧冲突，不能再展示/导出原读数 |

---

## 9. 代码结构

```text
boulder_judge/
├── cli.py          # 命令行入口
├── synthetic.py    # 确定性合成短片和校准批次
├── model.py        # 联合 HMM、计数训练、模型指纹
├── inference.py    # 前后向、Viterbi、三类路径概率 DP
├── evaluate.py     # 整段留出片联合分布核对
├── handcheck.py    # 四个手算小例
├── archive.py      # 留存读数路线的申诉档案
├── sealed.py       # 可选：销毁读数路线的联合因子/证词对照
├── report.py       # Markdown 交付物
└── mathutils.py    # log-sum-exp / 分布距离
tests/test_judge.py # 标准库 unittest
scripts/            # 一键演示和换输入脚本
```

运行测试：

```bash
python3 -m unittest discover -s tests -v
```
