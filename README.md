# 07 海报边缘：叠两层墨，还是加烧一块专色版

从零开发的打样程序。它学习随**纸张、印序、双墨覆盖率交互**而变的色阶响应，
在**整条留出的中间覆盖率色阶**上检验，接新海报时把所用纸的响应交给制版员，
并在给定纸、版形、费用、套印偏移与逐处色差容限下，给出**最低成本合格印单**；
无解时出示**会报废的证据**。保存印单后，机手可录入本单实际读数 / 偏移，程序
复查**最差像素与总费**。

> 本题给出的两条路线——(1) 直接出片、(2) 先烧版打试纸——**只实现路线 (1)**。
> 涉及路线 (2) 的内容仅用于说明差别（见「两条路线的差别」）。

全部代码为纯 Python 3 标准库，无 numpy / pytest，可在普通 CPU 上确定性复跑。

---

## 1. 快速开始

```bash
./run_all.sh                 # 一键生成数据、拟合、出印单、逐偏移核查、跑测试
```

或分步：

```bash
python3 -m proofing.cli gen-data --outdir data      # 生成合成色阶条 + 隐藏真值
python3 -m proofing.cli fit      --data data --out artifacts/response.json
python3 -m proofing.cli gen-jobs --outdir jobs      # 写出示例海报规格
python3 -m proofing.cli plan     --data data --job jobs/wide_block.json \
                                 --out artifacts/plan_wide_block.json
python3 -m proofing.cli report   --data data --job jobs/edge_image.json --recipe forward
python3 -m proofing.cli handoff  --data data --paper Paper1 --json artifacts/handoff_Paper1.json
python3 -m proofing.cli recheck  --data data --job jobs/wide_block.json \
                                 --plan artifacts/plan_wide_block.json --offset '{"A":0,"B":0}'
python3 -m unittest discover -s tests -v
```

输出目录：`data/`（输入材料）、`jobs/`（海报规格）、`artifacts/`（响应、印单、交接单）。

---

## 2. 材料是怎样合成的（可换入新输入）

`proofing/synthetic.py` 里有一份**隐藏真值**（不给拟合器看），遵循同一物理形式：

* 单墨 u 在纸张上的透过率 `T_u(c) = 1 - phi(c,g)·(1-10^-s_u)`，`phi(c,g)=c+g·c(1-c)`；
  光密度 `D = -log10(T)`。`s_u` 是实地密度，`g` 是**网点扩大**（随纸变）。
* 双墨叠印按**印序**先下后上做透过率乘积，中间夹一个**油墨交互因子**
  `kappa(cu)=1-(1-k)·phi(cu,g_下)`，`k` **按纸、按印序、按通道**各学一个。

数据里两张纸：

| 纸 | 正序（A 压 B，先 B 后 A = `BA`） | 反序（`AB`） | 专色 S |
|----|----|----|----|
| Paper1 | **0.90** | 0.70 | **0.90** |
| Paper2（新纸） | **0.75** | 0.70 | **0.90** |

于是：中间覆盖率色阶中，0.90 / 0.70 / 专色各按真值响应；训练只给 0/¼/½/¾/1
的整格，**留出** ⅛/⅜/⅝/⅞ 等中间覆盖率做检验。三通道实地密度相同（宽色块目标
无色偏），但中间覆盖率**逐通道网点扩大不同**，所以必须做三通道拟合。

**换入另一份相同格式的输入**：写一份 `paper_spec` JSON（键为纸名，值为
`k_forward / k_reverse / D_forward / D_reverse / gain_scale`），然后

```bash
python3 -m proofing.cli gen-data --outdir newdata --spec my_spec.json
python3 -m proofing.cli fit --data newdata --out newresponse.json
```

一份新输入 → 一份新的拟合响应、交接单与印单。（`examples/alt_spec.json` 是示例。）

---

## 3. 学到的响应与中间色阶检验

`fit` 只用 `split=train` 的整格拟合参数，再用 `split=holdout` 的**中间覆盖率**色阶
报误差。本材料上：

```
Paper1: 单墨 holdout max|err|≈0.0027  双墨 holdout max|err|≈0.0051
Paper2: 单墨 holdout max|err|≈0.0034  双墨 holdout max|err|≈0.0036
```

均 **远小于 0.02 容差**。`handoff` 会把「正序 / 反序 / 专色的实地密度、逐通道参数、
留出中间色阶误差、某覆盖率的双墨扫描」整页交给制版员。

---

## 4. 制版 / 印刷模型

* 最多三版（A、B、S）。每个**印法（recipe）**是一张按**下墨顺序**排列的油墨表：
  * `forward`（正序）= `(B, A)`：先 B 后 A，即 A 压在 B 上；
  * `reverse`（反序）= `(A, B)`；
  * `spot` = `(S)`；
  * `triple` = `(B, A, S)`：用来演示「一处叠到三层便报废」。
* 小单只允许 `forward / reverse / spot` 三种（`recipes_allowed`）。
* 每版整张**左移 / 原位 / 右移一格**：`delta=+1` 表示整体右移一格（设计列 `c`
  的墨落到印张列 `c+1`）。允许偏移集合由作业参数 `offsets` 决定：
  * `{"mode":"zero"}`：只有零偏移（「没有对位误差的小单」）；
  * `{"mode":"all","range":[-1,0,1]}`：逐版 3 种偏移的全组合（宽五格边缘图）；
  * `{"mode":"list","values":[...]}`：显式列出。
* 一处叠到 **≥3 层 → 整张报废**（任何偏移下出现即判死）。
* 逐格目标：给定期望三通道光密度 `density` 与容差 `tol`；另外可要求「某几通道相等」
  （`equal_channels`，宽色块要求 `[1,2]`，即另两通道相等）。

### 印单成本

`成本 = 版数 × plate + (用专色 ? spot : 0)`。小单默认 `plate=100, spot=250`，
所以正序（2 版）= 200、反序（2 版）= 200、专色（1 版 + 专色启用费）= 350。
**同等可行时正序更便宜/更先被选**（tie-break：成本 → 版数 → 名称，故 `forward` 先于 `reverse`）。

---

## 5. 路线 (1)：直接出片（本实现）

装纸前**定好版与印序**，要求在**每一种允许偏移**下**整张合格**；扫所有允许偏移取
最差者，全部 ≤ 容差才算合格。在合格者里取**最低成本**。

CLI：`plan`。它耗时只与「印法数 × 偏移组合数 × 目标格数」成正比，全部合成小图瞬间完成。

### 示例海报与预期结论（`jobs/`）

| 作业 | 情节 | 结论 |
|---|---|---|
| `wide_block` | Paper1、无对位误差、未锁印序 | **正序**，成本 200 |
| `wide_block_locked_reverse` | 机床锁反序 | 反序仅 0.70 → **专色**，350 |
| `wide_block_paper2` | 新纸正序仅 0.75 | **专色**，350 |
| `edge_image` | Paper1、宽五格边缘图、逐版 ±1 全偏移 | **无解**：正/反/专色都在某些偏移下超差 → 出示报废证据 |
| `edge_image_locked_zero` | 同图但对位锁定为零 | **正序**，200 |
| `scrap_demo` | 只允许三版叠印 | **无解**：任何偏移都出现 3 层 → 报废证据 |
| `triple_demo` | 三版与专色可选 | 丢弃会报废的 `triple`，选 **专色**，350 |
| `tone_ramp` | 目标落在**中间覆盖率**的正序曲线上 | 仅**正序**命中；反序超差 0.06、专色超差 0.08 → 8 个场景中唯一证明「用到了中间响应」者 |

例如 `edge_image` 的 `report` 会逐偏移列出最差格：

```
offset={'B': 0, 'A': 0}   PASS
offset={'B': 0, 'A': 1}   FAIL relief=+0.0172  worst_cell=(0, 2)
offset={'B': 1, 'A': 1}   FAIL relief=+0.8800  worst_cell=(0, 2)
...
```

因为**留白的边缘格一旦被邻格墨移到就永远无法靠调覆盖率修好**（0.90 压到 0.70 或
叠出三层），所以路线 (1) 在此图下必然无解——这正是要用试纸路线的前兆。

### 无解证据

`plan` 在不可行时打印并写入 `evidence`：每个被考虑印法在**第一个失败偏移**上的
`worst_cell / worst_channel / relief`，或 `scrap_3_layers`（含三层格的坐标）。

---

## 6. 保存后的机手复查

`recheck` 读入已存印单 + 机手录入的**实际偏移**（JSON，如 `{"A":0,"B":1}`），用同一
已学响应重绘整张，报 `scrap_cells / worst_relief / worst_cell / worst_channel / sheet_ok /
total_cost`。合格返回 0，超差返回 1。（若机手录入的是「读数」而非偏移，等价于给定偏移。）

---

## 7. 两条路线的差别（路线 2 未实现，仅说明）

* **路线 (1) 直接出片**：印前就要求**所有**允许偏移都合格——最稳，但边缘图常无解。
* **路线 (2) 先烧版打试纸**：先付一次试纸费 + 第一块版费，印一张带对位标的试纸，
  只报告**给定位置**的对位读数；机手据此在开印前做**一次、给定范围内**的调整。
  难点是：(a) 第一次烧掉的版**照付**；(b) **无法区分的两种偏移**只能执行**同一调整**。
  它通常能在边缘图下换得更低的总成本，但本实现选择路线 (1)，故这些只在 README 说明、
  不产生结果。二者若要对比，只需把 `plan_direct` 换成路线 2 的成本模型，其余
  `render` / `evaluate` / `recheck` 全部复用。

---

## 8. 目录结构

```
proofing/
  model.py     物理形式 + Nelder–Mead/黄金分割/线性代数（纯 stdlib）+ 拟合与留出报告
  synthetic.py 隐藏真值与三通道合成色阶条（可 --spec 换入新材料）
  jobs.py      印法/作业几何/偏移枚举/示例海报规格
  render.py    按偏移重绘印张、逐格/逐偏移判定、三层报废
  planner.py   路线 (1)：最低成本合格印单 + 无解证据
  cli.py       子命令：gen-data / gen-jobs / fit / plan / report / handoff / recheck
tests/test_proofing.py   19 个 unittest（物理、拟合、逐偏移、报废、换纸、中间色阶）
examples/alt_spec.json   换入新材料用的示例 spec
run_all.sh               端到端复跑
```

## 9. 复跑与测试

```bash
python3 -m unittest discover -s tests -v     # 19 tests, ~0.07s
./run_all.sh                                 # 端到端
```
