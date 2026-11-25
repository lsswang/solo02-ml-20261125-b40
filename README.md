# LedgerScribe：离线手写账页誊录

从空目录构建的离线誊录程序。馆员交来的整行材料只有**笔尖方向**（整数 0–7）、
**抬笔符号 `"U"`** 和最终金额；训练材料只标金额、不标字界。程序用同一书写员的
短行学习“数字与字界的关系”，留出整页检验。纯 Python 标准库，普通 CPU、固定
随机种子，可复跑。

## 一次跑通

```bash
bash demo.sh
```

或逐步：

```bash
python3 -m ledgerscribe.cli generate --data data          # 合成材料
python3 -m ledgerscribe.cli train --data data --writer A --out data/model_A.json
python3 -m ledgerscribe.cli train --data data --writer B --out data/model_B.json
python3 -m ledgerscribe.cli page  --model data/model_A.json \
        --lines data/heldout_lines.json --page data/pages.json --page-id A-040
python3 -m ledgerscribe.cli plan  --model data/model_A.json \
        --lines data/heldout_lines.json --page data/pages.json --page-id A-040
python3 -m ledgerscribe.cli reply --model data/model_A.json \
        --lines data/heldout_lines.json --page data/pages.json --page-id A-040 \
        --reply replies/reply_A-040_digit7.json
python3 -m ledgerscribe.cli check --model data/model_A.json \
        --lines data/heldout_lines.json --page data/pages.json --page-id A-040
python3 -m unittest tests.test_ledgerscribe
```

验题人换入另一份同格式输入（行 JSON + 页 JSON）即可得到新交付物，无需改代码。

## 数据格式

行 `data/heldout_lines.json`：`[{"line_id", "tokens": [0..7 或 "U", ...]}]`，
训练行 `data/A_train.jsonl` 额外带 `"amount"` 标签（**只有金额，没有字界**）。
页 `data/pages.json`：每页给 `threshold_ratio`（馆方比例）、`total`（全页合计）、
每行至多 8 个候选金额 `candidates`。`*_manifest.json` 记录真值字界，仅供核对，
训练与对平程序都不读取。

合成材料刻意包含：

* **一字两次抬笔**：数字 4 有三笔（`synth.TEMPLATES["4"]=(1,3,0)`），
  `A-line-3` 中 4 的两次字内抬笔全部保留；
* **两字连写**：训练集第 13 行强制字界不抬笔（manifest 的 `ligature_at`），
  检验行 `A-line-2` 字界无 U，23/28 靠无 U 连写切法共存。

## 模型

`ledgerscribe/model.py`：每个数字有固定笔数与每笔主方向；一笔发 1–2 个方向
token，笔与笔、字与字边界的抬笔服从 Bernoulli，不抬笔即连写。切分枚举递归
列出某金额下全部合法切分（含连写），同金额多种切法的概率相加。训练是只有
金额标签的约束 EM：E 步对该金额的全部切分软计数，M 步带模板先验伪计数归一。

行金额后验在**随页候选集内**归一（金额只在候选中取值），全程对数空间防下溢；
另加数字个数先验，避免少字假设靠更少边界罚分碾压多字真值。

## 入册规则（`ledgerscribe/page.py`）

1. 行金额概率 ≥ 本行最高概率 × `threshold_ratio` 才有资格；
2. 只在“有资格且整页合计等于 `total`”的读法组合里求解；
3. 所有对平读法一致的金额才签字，不一致**留白**，无对平读法出**冲突单**；
4. 界面只显示前三金额（`display.top3` + `n_more`），但签字覆盖全部候选——
   `A-086` 的 `34` 排在候选第四位仍被签出。

## 两条复核路线（本题实现路线 A）

馆里只有两格复核工时。

* **路线 A（局部辨字，本软件实现）**：至多问两次，每次标出一个确切笔段
  （原行 token 区间），馆员回答它单独构成哪个字，或“此处不是一个整字”。
  `plan` 子命令做 maximin 规划，选**最坏整字答复下仍能多签最多行**的回查，
  给出页号、笔段与每个答复后的第二问、签字依据；`not_whole_risk` 如实标注
  “不是整字”答复的后果。第二问可利用第一份答复。
* 路线 B（整行盲抄，仅作理解差别，未实现）：开工前只选一行，整行金额可靠
  但不提供字界，一次用完工时。区别在于路线 B 拿到的是行级金额、无法定位笔段，
  因此不能像路线 A 那样只排除某一种切法——本软件的切法层过滤（见下）是路线 A
  独有的能力。

复核过滤发生在**切法层而非金额层**：馆员锁定笔段（原行字符区间）后，不兼容的
切法被剔除，同金额的其他合法切法仍保留，再重新归一（`segments.apply_answer`）。

## 题面四个判定场景（`tests/test_ledgerscribe.py` 锁定）

| 场景 | 结果 |
| --- | --- |
| A-040，两行各支持 12/17、23/28，合计 40 | 两种对平读法（17+23、12+28），两行都留白 |
| 同上，答复末笔段是 7（或整行 17） | 切法过滤后签 **17 与 23** |
| 只把合计改成 35（A-035） | 唯一对平读法，直接签 **12 与 23** |
| 新书写员 B 的 B-040（新模型输出） | 17 跌出可信度门槛，只签 **12 与 28** |

另附：

* **第四候选小页 A-086**：合计 86，只有候选表第四位的 `34` 能对平，签 34 与 52；
* **独立模型检查**（`ledgerscribe/indep_check.py`，`check` 子命令）：用与主枚举
  完全独立的“按笔推进前向网格 DP”，在极短轨迹上逐金额复核后验，两套实现必须
  对平、后验必须归一。

## 目录

```
ledgerscribe/
  synth.py        合成书写员与检验页（含一字两抬笔、两字连写）
  model.py        切分枚举 + 约束 EM
  segments.py     笔段、字归属责任、局部辨字过滤
  page.py         资格门槛、对平、签字、路线 A 复核规划
  indep_check.py  独立前向 DP 模型检查
  cli.py          命令行
tests/            unittest 端到端测试
replies/          复核回复样例（digit 7 / not_whole）
data/             生成材料、模型（generate 子命令产出，可安全删除重建）
demo.sh           一键演示
```
