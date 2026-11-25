# 裁判结果 `attempt-A17`

- 路线卡：`W5-qualifier`
- 顶点：`TOP`；区点：`ZONE`；稳定帧数：`2`；截止帧（含）：`11`
- 模型：`cal-v1-initial`
- 模型指纹：`2d7e73780b939bfc…`
- 签单阈值：0.95
- 状态：**signed**
- 正式签单类别：**TOP**

## 三类分数的全轨迹概率

| 类别 | 含义 | 概率 |
|---|---|---:|
| `TOP` | 完攀 | 0.981736 |
| `ZONE` | 区点 | 0.018263 |
| `NO_ZONE` | 无区点 | 0.000000 |

- 全片至少一帧双手同触顶点的后验概率：0.9987
- 截止前至少一帧双手同触顶点的后验概率：0.9987
- 达到 0.95 逐帧证据阈值的帧：[7, 8]
- 后验概率 ≥0.5 的候选同顶帧：[6, 7, 8, 9]

## 逐帧联合证据

| 帧 | 读数(左/右) | P(左手且右手在顶点) | 证据档 | 后验最高联合状态 |
|---:|---|---:|---|---|
| 0 | NONE/NONE | 0.0000 | effectively_absent | NONE/NONE (0.997) |
| 1 | NONE/NONE | 0.0000 | effectively_absent | NONE/NONE (0.913) |
| 2 | NONE/ZONE | 0.0002 | effectively_absent | NONE/NONE (0.334) |
| 3 | ZONE/ZONE | 0.0026 | effectively_absent | ZONE/ZONE (0.970) |
| 4 | ZONE/ZONE | 0.0088 | effectively_absent | ZONE/ZONE (0.972) |
| 5 | ZONE/ZONE | 0.0536 | weak | ZONE/ZONE (0.799) |
| 6 | TOP/TOP | 0.6835 | ambiguous | TOP/TOP (0.684) |
| 7 | TOP/TOP | 0.9505 | signing_strength | TOP/TOP (0.951) |
| 8 | TOP/TOP | 0.9653 | signing_strength | TOP/TOP (0.965) |
| 9 | TOP/TOP | 0.9210 | strong_below_signing | TOP/TOP (0.921) |

## 最可能的完整联合轨迹

| 帧 | 左手触点 | 右手触点 |
|---:|---|---|
| 0 | NONE | NONE |
| 1 | NONE | NONE |
| 2 | NONE | NONE |
| 3 | ZONE | ZONE |
| 4 | ZONE | ZONE |
| 5 | ZONE | ZONE |
| 6 | TOP | TOP |
| 7 | TOP | TOP |
| 8 | TOP | TOP |
| 9 | TOP | TOP |
