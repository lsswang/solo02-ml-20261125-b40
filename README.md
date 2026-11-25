# 两颗星挤进同一光圈：离线测光和约班程序

纯 Python 标准库实现，不需要联网、数据库或第三方包。题目中的两种交班办法，本项目**选择实施“机动次班券”**；同时实现固定买两晚的只读计算，用来核对题目给出的对照结论。

## 一键复跑

```bash
cd /workspace

# 生成确定性的小型合成标定、历表和班表（仓库中已带一份生成结果）
python3 -m observatory generate-sample --out sample_inputs

# 跑完整示例、预约/回填流程、券互斥检查和单元测试
bash scripts/run_demo.sh
```

完整运行后，关键 JSON 在 `out/demo/`，可回填预约在 `out/demo/state/`。

也可以只跑测试：

```bash
python3 -m unittest discover -s tests -v
```

---

## 1. 测光标定与点查

### 输入格式

标定目录需含两个同格式 CSV：

- `backgrounds.csv`：每夜空场背景估计及其误差  
  列：`night,bg1,bg2,bg_sigma1,bg_sigma2`
- `calibration.csv`：轮流遮住一颗星后的已知亮度和原始带噪读数  
  列：`night,direction,masked_star,measured_star,brightness_level,known_flux,raw_ap1,raw_ap2,sigma_ap1,sigma_ap2`

`brightness_level` 相同的整组会在留出检验中一起留出。程序对每颗星、每个孔径做背景扣除后的加权非负过原点拟合 `reading - background = r * known_flux`，并逐亮度档执行 leave-one-level-out 检验。

### 重新标定

```bash
python3 -m observatory calibrate \
  --input sample_inputs/calibration_original \
  --out out/calibration_original.json

python3 -m observatory calibrate \
  --input sample_inputs/calibration_recalibrated \
  --out out/calibration_recalibrated.json
```

生成样本的拟合中心值为（固定随机种子，数值可复跑）：

| 标定 | 甲 `(r1,r2)` | 乙 `(r1,r2)` | 两列 95% 误差区间 |
|---|---:|---:|---|
| 原标定 |约 `(1.026, 0.996)` |约 `(1.017, 0.511)` | 不重合 |
| 真实重标定 |约 `(1.052, 0.588)` |约 `(1.030, 0.461)` | 重合 |

结果 JSON 中包含：

- `responses`：中心响应矩阵；
- `response_intervals_95`：响应误差区间；
- `columns_overlap_within_95`：两列是否在误差范围内重合；
- `holdout_validation`：按整组亮度档留出的 RMSE、最大残差和响应等价误差。

### 点查下降量 `(6,3)`

观测文件格式：

```json
{
  "observation_id": "DROP-6-3",
  "drops": [6.0, 3.0],
  "error_bounds": [0.30, 0.30],
  "response_uncertainty": "se95"
}
```

为了直接核对题设的精确单位响应 `甲=(1,1)`、`乙=(1,0.5)`，另带一个精确夹具：

```bash
python3 -m observatory query \
  --calibration sample_inputs/calibration_textbook_exact.json \
  --observation sample_inputs/observations/drop_6_3.json \
  --uncertainty none \
  --out out/query_textbook.json
```

结果只有乙可能，深度区间为：

```text
supported stars: ['乙']
乙 depth interval: [5.7, 6.3]
```

区间算法是：对每个孔径求满足 `|drop_k - d*r_k| <= error_k` 的深度区间，再取两孔径交集。若使用标定响应误差，则把 `r_k` 视为位于其响应区间内，并用最坏上下界扩展深度区间。

对两份真实拟合标定执行：

```bash
python3 -m observatory query \
  --calibration out/calibration_original.json \
  --observation sample_inputs/observations/drop_6_3.json \
  --out out/query_original.json

python3 -m observatory query \
  --calibration out/calibration_recalibrated.json \
  --observation sample_inputs/observations/drop_6_3.json \
  --out out/query_recalibrated.json
```

当前合成样本输出：

- 原标定：`supported_stars = ["乙"]`；
- 重标定：`supported_stars = ["甲","乙"]`。

---

## 2. 排班的最小最坏剩余准则

一份申请 JSON 包含：

- `order_id`
- `horizon_nights`：最多 `8`，程序强制；
- `ephemerides` 内联，或用 `ephemerides_file` 指向 CSV；
- `known_clear_nights` 与/或 `clear_records_file`；
- `voucher_fee`。

历表 CSV 列：

```text
ephemeris_id,star,transit_time,pattern,note
```

`pattern` 是长度等于 `horizon_nights` 的未来明暗串，`1=该晚预测变暗`，`0=未变暗`。

班表 CSV 列：

```text
night,date,cost,first_eligible,voucher_eligible,fixed_eligible,note
```

### 选择的办法：机动次班券

预约时选择：

1. 一个首班夜 `f`；
2. 对首班两种结果 `0/1`，分别从事先可兑现的后续券班中选一个兑现夜；
3. 券上列出两个分支实际可能兑现的夜晚；
4. 同一 `voucher_id` 在本地登记册中不能再发给另一组。

优化按字典序：

\[
\min \bigl(\text{最坏剩余历表数},\ \text{最坏费用}\bigr).
\]

费用定义为：

\[
\text{worst cost}=c_f+\text{voucher\_fee}
+\max_{b\in\{0,1\},\ b\text{ 可能}} c_{n_b}.
\]

其中 `n_b` 是首班结果为 `b` 时选定的兑现夜。给定已经达到的最小最坏剩余数后，每个分支只在“能把该分支压到该阈值内”的夜班中选最便宜的；同价时取最早夜，保证结果可复跑。

默认把机动券视为**必须兑现一次的次班产品**，因此即使某晚不能进一步减少历表，也会给出实际兑现夜（例如第 5 晚），便于验题人核对“守了也无法分辨”。若站长允许放弃兑现并要把边际费用也压到最低，可给 `plan/book` 加 `--allow-skip-redemption`；此时程序仍会报告每个开放夜班的剩余数，只是在省钱更优时把 `redeem_night` 置为 `null`。

### 对照办法：固定买两晚

`plan --method fixed` 枚举所有两个 `fixed_eligible` 夜晚，按两晚二进制观测分组，计算最大组大小和两晚总费用。它只用于核对差别；正式预约和回填流程使用机动券。

---

## 3. 三晚四份历表

样本：`sample_inputs/applications/order_four.json`，明暗串正是：

```text
000, 010, 100, 101
```

班表：`sample_inputs/shifts/shifts_three.csv`，首班为第 1 晚，券班为第 2、3 晚。

```bash
python3 -m observatory plan \
  --application sample_inputs/applications/order_four.json \
  --shifts sample_inputs/shifts/shifts_three.csv \
  --query out/query_original.json \
  --method both \
  --out out/plan_four.json
```

关键结果：

- 固定买两晚：任一对夜晚的最坏剩余都是 `2`，最优费用组合为第 1、2 晚，`worst_remaining=2`；
- 机动券：先守第 1 晚：
  - 首班亮 `0`：剩下 `000,010`，兑现第 2 晚即可分开；
  - 首班暗 `1`：剩下 `100,101`，兑现第 3 晚即可分开；
  - 因而 `worst_remaining=1`，四份全部可辨。

### 用预约号回填

```bash
python3 -m observatory book \
  --application sample_inputs/applications/order_four.json \
  --shifts sample_inputs/shifts/shifts_three.csv \
  --query out/query_original.json \
  --state-dir out/state \
  --registry out/state/voucher_registry.json \
  --reservation-id RSV-FOUR-001 \
  --voucher-id VCH-FOUR-001

python3 -m observatory fill-first \
  --state-dir out/state \
  --registry out/state/voucher_registry.json \
  --reservation-id RSV-FOUR-001 \
  --result bright      # 或 dim

# 程序会给出实际应兑现的夜晚；再回填该晚结果：
python3 -m observatory fill-second \
  --state-dir out/state \
  --registry out/state/voucher_registry.json \
  --reservation-id RSV-FOUR-001 \
  --result bright
```

也可用：

```bash
python3 -m observatory show --reservation-id RSV-FOUR-001 --state-dir out/state
```

查询预约状态。重复同一条 `book` 命令时可换新预约号，或加 `--force`（本地演示覆盖）；完整 `scripts/run_demo.sh` 会先清空自己的 `out/demo` 目录。

---

## 4. 甲、乙历表与新标定/新班表

同一份申请 `sample_inputs/applications/order_ab.json` 含三份历表，horizon=8：

- `B4`：乙，只在第 4 晚变暗，`00010000`；
- `B6`：乙，只在第 6 晚变暗，`00000100`；
- `A4`：甲，未来明暗串与 `B4` 完全相同，也是 `00010000`。

另有一条确知未变暗记录 `sample_inputs/records/clear_ab.csv`（第 2 晚），会进入相容性过滤；它不排除这三份样本历表。

班表有两份同格式文件：

- `shifts_ab_open4.csv`：第 1 晚首班，唯一机动兑现夜为第 4 晚；
- `shifts_ab_open5.csv`：第 1 晚首班，唯一机动兑现夜为第 5 晚。

矩阵：

| 测光/标定结果 | 班表 | 标定过滤后历表 | 最坏剩余 | 实际兑现夜 | 解释 |
|---|---|---:|---:|---:|---|
| 原标定，只支持乙 | 开放第 4 晚 | `B4,B6` | **1** | 4 | 第 4 晚暗→B4，亮→B6 |
| 原标定，只支持乙 | 唯一开放第 5 晚 | `B4,B6` | **2** | 5 | 两者第 5 晚都亮，无法辨 |
| 重标定，支持甲、乙 | 仍守第 4 晚 | `A4,B4,B6` | **2** | 4 | 第 4 晚暗会同时留下 `A4,B4` |
| 重标定，支持甲、乙 | 唯一开放第 5 晚 | `A4,B4,B6` | **3** | 5 | 三者第 5 晚都亮 |

复跑命令：

```bash
# 原标定 + 第4晚
python3 -m observatory plan \
  --application sample_inputs/applications/order_ab.json \
  --shifts sample_inputs/shifts/shifts_ab_open4.csv \
  --query out/query_original.json \
  --out out/plan_ab_orig_open4.json

# 原标定 + 第5晚
python3 -m observatory plan \
  --application sample_inputs/applications/order_ab.json \
  --shifts sample_inputs/shifts/shifts_ab_open5.csv \
  --query out/query_original.json \
  --out out/plan_ab_orig_open5.json

# 重标定 + 第4晚
python3 -m observatory plan \
  --application sample_inputs/applications/order_ab.json \
  --shifts sample_inputs/shifts/shifts_ab_open4.csv \
  --query out/query_recalibrated.json \
  --out out/plan_ab_recal_open4.json
```

重标定后可预约并回填第 1 晚（这些历表第 1 晚都亮），程序会指示守第 4 晚；再回填 `dim`，最终留下 `A4` 和 `B4`：

```bash
python3 -m observatory book \
  --application sample_inputs/applications/order_ab.json \
  --shifts sample_inputs/shifts/shifts_ab_open4.csv \
  --query out/query_recalibrated.json \
  --state-dir out/state \
  --registry out/state/voucher_registry.json \
  --reservation-id RSV-AB-RECAL \
  --voucher-id VCH-AB-RECAL

python3 -m observatory fill-first \
  --state-dir out/state --registry out/state/voucher_registry.json \
  --reservation-id RSV-AB-RECAL --result bright

python3 -m observatory fill-second \
  --state-dir out/state --registry out/state/voucher_registry.json \
  --reservation-id RSV-AB-RECAL --result dim
```

同一 `voucher_id` 第二次 `book` 给另一 `order_id` 会以退出码 2 拒绝。

---

## 5. 验题人如何换入另一份同格式输入

不需要改 Python 代码。准备新的同 schema 输入后：

```bash
# 1) 新标定目录（必须含 backgrounds.csv 和 calibration.csv）
python3 -m observatory calibrate \
  --input /path/to/new_calibration_dir \
  --out /path/to/out/new_calibration.json

# 2) 新观测点查
python3 -m observatory query \
  --calibration /path/to/out/new_calibration.json \
  --observation /path/to/new_observation.json \
  --out /path/to/out/new_query.json

# 3) 用新班表和新申请预约
python3 -m observatory book \
  --application /path/to/new_application.json \
  --shifts /path/to/new_shifts.csv \
  --query /path/to/out/new_query.json \
  --state-dir /path/to/out/state \
  --registry /path/to/out/state/voucher_registry.json \
  --reservation-id NEW-RSV-001 \
  --voucher-id NEW-VCH-001
```

注意：

- `horizon_nights` 和班表 `night` 都不能超过 `8`；
- 新 query 的 `supported_stars` 名称必须与历表 `star` 完全一致；
- 若不想按星过滤，可把 `--query ...` 换成 `--stars 甲,乙`，或省略星源；
- 班表中把某晚设为 `voucher_eligible,true` 即加入可兑现空班；设为 `false` 即关闭；
- 预约状态和券登记册都是普通 JSON，可复制、检查或交给接班员。

## 目录

```text
observatory/
  calibration.py    # CSV 标定读取、非负拟合、亮度档留出
  querying.py       # 有界误差深度区间和单星解释
  scheduling.py     # 过滤、固定班对照、机动券 minimax
  reservations.py   # 预约号、券互斥、首晚/次晚回填
  synthetic.py      # 确定性合成样本生成
  cli.py            # 命令行
sample_inputs/      # 已生成的同格式输入
scripts/run_demo.sh # 完整可复跑示例
tests/              # 单元测试和题设断言
```
