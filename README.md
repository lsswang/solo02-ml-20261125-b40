# 浮雕工坊离线测坡与制版系统

纯 Python 标准库实现，不需要联网、GPU 或第三方包。它完成：

1. 用四通道偏振读数标定有向坡度 \(-2,-1,0,1,2\) 的概率；
2. 在**整块留出试片**上报告平均对数损失；
3. 为新扫描的每条有向格边保留五个坡度的完整概率；
4. 应用实测止规：被排除或概率为零的坡度成为硬禁用；
5. 在格点高度只能为 `0/1/2/3`、闭环必须闭合的条件下，精确求边对数概率总和最高的整板；
6. 比较整铣与留缝分段工法，按后续补孔概率和拒单损失选择实际刀位；
7. 正式补孔时检查不可回填刀位，冲突则保留原切记录并说明原因；
8. 按总工单号或补孔工单号查询实际刀位。

## 30 秒复跑

```bash
bash run_demo.sh
```

脚本会生成小型合成标定材料、训练模型、计算留出试片对数损失、复跑题中两个标准例、执行补孔/冲突/查刀位，并用独立枚举器和单元测试复核。

也可以直接使用可执行入口：

```bash
./relief --help
# 或
python3 -m relief_workshop --help
```

## 目录

- `relief_workshop/`：标定、概率预测、精确 DP、枚举复核、工单/刀位状态机和 CLI。
- `examples/canonical_2x2/`：题中的两排两列小章。
- `examples/canonical_2x3/`：题中的两排三列留缝与期望拒单损失例。
- `tests/test_scenarios.py`：标准结果、补孔冲突、DP 对拍枚举测试。
- `run_demo.sh`：一键完整流程。
- `data/`、`out/demo/`：运行脚本后生成的样本和交付物。

---

## 输入格式

所有面向人的行列坐标均为 **1 基**；程序内部转成 0 基。横边方向固定为左→右，竖边固定为上→下；边的坡度定义为终点高度减起点高度。正差表示沿该方向的终点成品更高（少铣/留料），负差表示终点更低（多铣/向下铣），由此回答“往上铣还是往下铣”。

### 1. 标定 CSV

表头：

```csv
sample_id,coupon_id,delta,ch0,ch1,ch2,ch3
S0001,C001,-2,......
```

`delta` 是该有向边的真实高度差，只取 `-2,-1,0,1,2`。相同 `coupon_id` 表示来自同一块试片；演示中的留出集 `HELDOUT-3X4` 是整块未参与训练的试片。

### 2. 新扫描 CSV

每行一条有向格边：

```csv
scan_id,rows,cols,edge_id,u_row,u_col,v_row,v_col,ch0,ch1,ch2,ch3
DEMO,2,2,H-r1-c1,1,1,1,2,...
DEMO,2,2,V-r1-c2,1,2,2,2,...
```

推荐边号：

- 横边：`H-r<行>-c<左列>`
- 竖边：`V-r<顶行>-c<列>`

`generate` 会写出覆盖全部横边和竖边的样例。

### 3. 逐边概率 JSON

`scan-edges` 的产物可直接交给 `start`。结构为：

```json
{
  "rows": 2,
  "cols": 2,
  "coordinate_base": 1,
  "edges": [
    {
      "edge_id": "H-r1-c1",
      "u_row": 1, "u_col": 1, "v_row": 1, "v_col": 2,
      "probabilities": {"-2": 0.01, "-1": 0.34, "0": 0.05, "1": 0.58, "2": 0.02}
    }
  ]
}
```

这就是“每条有向格边的完整概率”。若验题方已有同格式概率文件，可跳过偏振读数步骤，直接把该 JSON 传给 `start --edges`。

### 4. 工单 JSON

```json
{
  "order_id": "WO-EXAMPLE",
  "rows": 2,
  "cols": 3,
  "holes": [
    {"row": 1, "col": 1, "height": 1},
    {"row": 2, "col": 1, "height": 1}
  ],
  "gauge": {
    "vertical_allowed": [0],
    "horizontal_allowed": [-1, 0, 1],
    "edge_allowed": {"H-r2-c1": [-1]}
  },
  "method_policy": "any",
  "seam_columns": [2, 3],
  "contingencies": [
    {
      "contingency_id": "MID-2",
      "probability": 0.75,
      "rejection_loss": 1,
      "column": 2,
      "height": 2
    }
  ]
}
```

止规字段均可省略：

- 全局：`allowed_deltas`、`forbidden_deltas`
- 横向/竖向：`horizontal_allowed`、`horizontal_forbidden`、`vertical_allowed`、`vertical_forbidden`
- 单边：`edge_allowed`、`edge_forbidden`

原始概率为 `0` 的坡度也会自动禁用。止规是硬可行域：不允许坡度的对数权重为 `-∞`；最优得分直接累加幸存坡度的**原始标定概率对数**，因此验题方可逐边用输入 JSON 复核。程序另给止规后的条件归一化概率，仅供查看。

夹持孔图也可用 CSV 单独传入 `start --holes holes.csv`：

```csv
row,col,height,allowed_heights
1,1,1,
2,3,,0|1|2
```

---

## 分步运行

下面命令均在项目根目录执行。

### 生成另一份小型合成材料

```bash
python3 -m relief_workshop generate \
  --out data/my_scan \
  --seed 123 \
  --rows 3 --cols 6
```

输出包括：

- `calibration_train.csv`
- `calibration_holdout.csv`：整块留出试片；
- `new_scan.csv`：新扫描；
- `new_scan_true_heights.csv`：仅用于核对合成真值；
- `new_scan_work_order.json`：保证可开工的示例工单。

尺寸可改，题设范围内建议 `--rows 1..3 --cols 1..6`。

### 标定并报告留出对数损失

```bash
python3 -m relief_workshop calibrate \
  --train data/my_scan/calibration_train.csv \
  --holdout data/my_scan/calibration_holdout.csv \
  --model out/model.json \
  --report out/calibration_report.json
```

报告中的 `holdout_mean_log_loss_nats` 是整块留出试片全部有向边的平均交叉熵/对数损失。模型是带方差下限和轻标签平滑的高斯朴素贝叶斯；训练完全确定、可复跑。

### 预测新扫描的逐边完整概率

```bash
python3 -m relief_workshop scan-edges \
  --scan data/my_scan/new_scan.csv \
  --model out/model.json \
  --out out/my_edges.json
```

### 开工：求最高边对数概率整板并签首刀

```bash
python3 -m relief_workshop start \
  --edges out/my_edges.json \
  --order data/my_scan/new_scan_work_order.json \
  --state-dir out/state \
  --out out/my_start.json
```

输出包含：

- 全局最优高度图；
- DP 最优得分；
- 整铣和每个允许留缝列的候选；
- 各后续补孔类型的接受/拒绝判定；
- 期望拒单损失；
- 选中工法的实际刀位；
- 留缝工法的合法接缝高度余量。

### 补孔

```bash
python3 -m relief_workshop supplement \
  --state-dir out/state \
  --order-id WO-2X3-SEAM \
  --ticket examples/canonical_2x3/supplement_right2.json
```

也可直接给孔：

```bash
python3 -m relief_workshop supplement \
  --state-dir out/state \
  --order-id WO-2X3-SEAM \
  --ticket-id WO-LIVE-1 \
  --hole 1:3:2 --hole 2:3:2
```

若新孔与已切高度冲突，命令返回退出码 `4`，状态中的刀位一条都不会回填或改写；冲突明细和原刀位均保留。

### 按工单号查刀位

```bash
# 总工单：查整板所有实际刀位
python3 -m relief_workshop cuts \
  --state-dir out/state \
  --order-id WO-2X3-SEAM

# 补孔单号：查该单签下的接缝刀位，同时给出父工单
python3 -m relief_workshop cuts \
  --state-dir out/state \
  --order-id WO-2X3-RIGHT2
```

状态是普通 JSON：`out/state/orders/<工单号>.json`，工单/补孔号索引在 `ticket_index.json`。

### 小格枚举复核

```bash
python3 -m relief_workshop verify \
  --edges examples/canonical_2x3/edge_probabilities.json \
  --order examples/canonical_2x3/order.json
```

当格点数不超过 `--enumerate-limit`（默认 8，即枚举 \(4^8=65536\) 件）时，程序独立枚举所有高度并核对 DP 得分。更大的 ≤3×6 扫描使用沿线状态的精确 max-product DP，线宽不超过 3，每线最多 \(4^3=64\) 个状态。

### 查看“局部最可能方向不闭合”

```bash
python3 -m relief_workshop local-check \
  --edges examples/canonical_2x2/edge_probabilities.json \
  --order examples/canonical_2x2/order_original.json
```

它报告每条边的局部众数坡度和每个小方格的环向和；非零环向和说明这些局部最优方向不能合成实际高度。此命令发现不闭合时退出码为 `6`。

---

## 题中标准例的预期结果

### 两排两列小章

输入：`examples/canonical_2x2/`。

- 左列固定为 1；
- 竖边止规只允许 0；
- 上横边概率为 `+1=.58`、`-1=.34`，即偏爱 `+1` 但支持 `-1`；
- 原工单下横边只允许 `-1`。

局部众数为上 `+1`、下 `-1`、竖边 `0`，方格环向和为 `2`，不能闭合。整板 MAP 改为上横边 `-1`，得到：

```text
1 0
1 0
```

仅把下横边止规改为 `+1` 后，新工单 `WO-2X2-LOWER-PLUS` 的最优板为：

```text
1 2
1 2
```

若保持原扫描/原止规，在首刀前把右孔锁到高度 2，则下横边需要 `+1` 而止规只允许 `-1`，`WO-2X2-LOCK-RIGHT2` 报告无可行整板。

### 两排三列料

输入：`examples/canonical_2x3/`。

- 左列固定 1；
- 竖边只许 0；
- 两条横向关系允许 `-1,0,+1`，且严格偏好 0。

初始最优板是全 1：

```text
1 1 1
1 1 1
```

各开工方案对概率补孔的期望拒单损失：

| 开工方案 | 中列高 2，p=.75，拒单损失 1 | 右列高 2，p=.25，拒单损失 5 | 期望损失 |
|---|---:|---:|---:|
| 整铣 | 拒 | 拒 | `.75*1 + .25*5 = 2.00` |
| 中列留缝 | 接 | 拒（右列已切成 1） | `.25*5 = 1.25` |
| 右列留缝 | 拒（中列已切成 1） | 接 | `.75*1 = 0.75` |

因此选择**右列留缝分段工法**。首刀实际完成第 1、2 列共 4 个点，全部高 1；第 3 列是未切接缝并占用一道额外工位。

回放正式补孔：

1. `WO-2X3-MID2` 要求中列高 2，与已切中列高 1 冲突；系统保留 4 条首刀记录并给出两条冲突。
2. `WO-2X3-RIGHT2` 要求右列高 2；右列是接缝，可行，于是签下两条接缝刀位，成品为：

```text
1 1 2
1 1 2
```

运行：

```bash
python3 -m unittest discover -s tests -v
```

可复核以上全部数值以及 DP 与独立枚举的一致性。

---

## 换入另一份输入

验题人可按以下任一方式替换：

1. **完整四通道流程**：用相同表头放入新的 `new_scan.csv`（不超过 3×6），必要时提供新的标定 CSV；依次运行 `calibrate`、`scan-edges`、`start`。
2. **只换逐边概率**：提供同格式 `edge_probabilities.json`，再运行 `start`。
3. **只换孔图/止规/补孔概率**：复制一个工单 JSON 修改；扫描和概率文件不必改变。
4. **换夹持孔 CSV**：使用 `start --holes your_holes.csv`。

每件交付物都由当前输入重新计算；`out/state/orders/*.json` 是可归档的工单、决策、冲突和刀位账本。

## 退出码

| 码 | 含义 |
|---:|---|
| 0 | 命令成功；`start` 时表示存在可行板 |
| 2 | 输入文件/参数格式错误 |
| 3 | 开工约束下无可行整板 |
| 4 | 正式补孔被拒（冲突或无可行整板），原刀位保留 |
| 5 | DP 与枚举得分不一致（测试型错误） |
| 6 | 局部众数坡度存在闭环不闭合 |
