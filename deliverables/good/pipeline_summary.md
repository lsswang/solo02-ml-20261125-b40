# 咖啡拼配交付摘要

- 盲评烘次：`R-BLIND-11, R-BLIND-12`
- 划分不变量通过：`True`
- 盲评 log loss：**0.366670**；常数基线：1.269838
- 成功称料样例：`W-001`

## 过门槛且库存可做的配方

| 配方 | 预测相似率 | 库存可做杯数 |
|---|---:|---:|
| `JIA-2409:2|YI-2409:2` | 0.843 | 2 |

## 营业演练

### 01_bucket_three_orders
- `O1` → `accepted_and_bucket_opened` (p=0.843)
- `O2` → `accepted` (p=0.843)
- 出杯 `O1` ×1 成功
- 出杯 `O2` ×1 成功
- `O3` → `rejected_material` (p=0.843)

### 02_three_jia_withdrawal_invalidates_two_cups
- 提料/盘减事件：`stock_withdrawn`
- `COMMIT2` → `rejected_material` (p=0.843)

### 02b_two_commitments_then_three_jia_stocktake
- `P1` → `accepted_direct` (p=0.920)
- `P2` → `accepted_direct` (p=0.920)
- 提料/盘减事件：`stock_withdrawn`
- `serve` 预期/非预期错误：cannot serve order P1 with status unfulfillable_after_stocktake

### 03_trial_same_order_like
- `T1` → `awaiting_trial_reading` (p=0.500)
- 试杯 `like`：后验 0.900 → `accepted`
- 出杯 `T1` ×1 成功

### 04_trial_same_order_unlike
- `T1` → `awaiting_trial_reading` (p=0.500)
- 试杯 `unlike`：后验 0.100 → `rejected_quality`
- `serve` 预期/非预期错误：cannot serve order T1 with status rejected

### 05_reliably_calibrated_whole_bucket_signoff
- `CAL1` → `accepted_and_bucket_opened` (p=0.920)
