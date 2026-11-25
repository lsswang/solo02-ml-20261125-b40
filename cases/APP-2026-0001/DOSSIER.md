# 申诉档案 APP-2026-0001

- 档案模式：retain_wrist_readings（保留双腕离散读数）
- 片段：`attempt-A17`；路线卡：`W5-qualifier`
- 建档时间：2026-11-25T16:07:59+00:00
- 当前状态：**signed**；当前签单类别：**ZONE**

## 最初签了什么

- 初始模型：`cal-v1-initial` (`2d7e73780b93…`)
- 初始概率：完攀 0.9817, 区点 0.0183, 无区点 0.0000
- 初始签单：**TOP**（signed）

## 申诉与改判链

### APP-2026-0001-R01 — revised

- 理由：新人工标注校准批次发现 TOP/TOP 成对读数可由 ZONE/ZONE 的相关误识别产生
- 上一判复现：成功；最大概率差 0
- 最初判单复现：成功
- 新校准：`cal-v2-paired-error-revision` (`f3a7098961cb…`)
- 新概率：完攀 0.0020, 区点 0.9980, 无区点 0.0000
- 新签单：**ZONE**
- 是否改变结果：**是**

## 最新逐帧触顶证据

| 帧 | P(左,右均在顶点) | 证据档 | 读数 |
|---:|---:|---|---|
| 0 | 0.0000 | effectively_absent | NONE/NONE |
| 1 | 0.0000 | effectively_absent | NONE/NONE |
| 2 | 0.0000 | effectively_absent | NONE/ZONE |
| 3 | 0.0001 | effectively_absent | ZONE/ZONE |
| 4 | 0.0001 | effectively_absent | ZONE/ZONE |
| 5 | 0.0001 | effectively_absent | ZONE/ZONE |
| 6 | 0.0006 | effectively_absent | TOP/TOP |
| 7 | 0.0011 | effectively_absent | TOP/TOP |
| 8 | 0.0020 | effectively_absent | TOP/TOP |
| 9 | 0.0036 | effectively_absent | TOP/TOP |
