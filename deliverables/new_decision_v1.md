# 裁判结果 `public-new-W5-014`

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
| `TOP` | 完攀 | 0.953941 |
| `ZONE` | 区点 | 0.046037 |
| `NO_ZONE` | 无区点 | 0.000022 |

- 全片至少一帧双手同触顶点的后验概率：0.9929
- 截止前至少一帧双手同触顶点的后验概率：0.9929
- 达到 0.95 逐帧证据阈值的帧：无
- 后验概率 ≥0.5 的候选同顶帧：[7, 8, 9, 10]

## 逐帧联合证据

| 帧 | 读数(左/右) | P(左手且右手在顶点) | 证据档 | 后验最高联合状态 |
|---:|---|---:|---|---|
| 0 | NONE/NONE | 0.0000 | effectively_absent | NONE/NONE (0.999) |
| 1 | NONE/NONE | 0.0000 | effectively_absent | NONE/NONE (0.987) |
| 2 | NONE/NONE | 0.0000 | effectively_absent | NONE/NONE (0.951) |
| 3 | OCCLUDED/NONE | 0.0001 | effectively_absent | NONE/NONE (0.695) |
| 4 | ZONE/NONE | 0.0004 | effectively_absent | ZONE/NONE (0.781) |
| 5 | ZONE/NONE | 0.0047 | effectively_absent | ZONE/NONE (0.716) |
| 6 | ZONE/ZONE | 0.1642 | weak | ZONE/ZONE (0.335) |
| 7 | TOP/TOP | 0.7999 | strong_below_signing | TOP/TOP (0.800) |
| 8 | OCCLUDED/TOP | 0.8800 | strong_below_signing | TOP/TOP (0.880) |
| 9 | TOP/TOP | 0.9366 | strong_below_signing | TOP/TOP (0.937) |
| 10 | OCCLUDED/OCCLUDED | 0.7503 | strong_below_signing | TOP/TOP (0.750) |
| 11 | TOP/ZONE | 0.1293 | weak | TOP/ZONE (0.529) |

## 最可能的完整联合轨迹

| 帧 | 左手触点 | 右手触点 |
|---:|---|---|
| 0 | NONE | NONE |
| 1 | NONE | NONE |
| 2 | NONE | NONE |
| 3 | NONE | NONE |
| 4 | ZONE | NONE |
| 5 | ZONE | NONE |
| 6 | TOP | ZONE |
| 7 | TOP | TOP |
| 8 | TOP | TOP |
| 9 | TOP | TOP |
| 10 | TOP | TOP |
| 11 | TOP | ZONE |
