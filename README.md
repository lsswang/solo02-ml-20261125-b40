# 两扬声器 × 两听音点的小声场调试台

这是一个纯 Python 标准库实现的合成声音展调试台：没有 NumPy/SciPy 依赖，
全部素材都是短数组和 PCM-16 WAV，普通 CPU 上几秒钟即可重跑。

题目中的两条路线，本项目**做通“为每个容许隔板位置预制固定立体声轨”的路线**；
同时保留“换景空拍探测房态”的对照实验，用来理解两条路线的差别。探测不能区分的
`partition_half` / `partition_closed` 分支，会导出一条在两个学得房间上都逐项验证
通过的稳健音轨，而不是猜测房态。

## 一键复跑

```bash
./run_demo.sh
```

脚本会重新生成 `samples/`、标定、做留出验证、导出所有音轨、运行负例和单元测试。
环境只需 Python 3.10+；不需要安装第三方包。

也可以只跑测试：

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

## 给验题人：换入另一份同格式输入

### 换一个节目

节目格式见 `samples/programs/scene_change_program.json`。最简替换方式：

1. 放入新的解说 WAV（单声道、与房间相同采样率）；
2. 复制并修改一份节目 JSON，填写 `cues`、切换时刻、音量上限和泄漏阈值；
3. 运行：

```bash
./run_on_input.sh path/to/new_program.json samples/learned/original.json out/new_delivery
```

或直接运行等价命令：

```bash
PYTHONPATH=. python3 -m soundbench build \
  --program path/to/new_program.json \
  --room samples/learned/original.json \
  --outdir out/new_delivery \
  --strict
```

仓库附带一个可直接试的替换节目：

```bash
PYTHONPATH=. python3 -m soundbench build \
  --program examples/replacement_program.json \
  --room samples/learned/original.json \
  --outdir /tmp/replacement_delivery --strict
```

clip 路径相对于节目 JSON 解析；也可写绝对路径。

### 换一套校准录音

校准包是一个 4 通道 WAV 加同名/同目录 JSON sidecar：

- WAV 通道：`speaker_1_probe, speaker_2_probe, mic_near, mic_far`；
- JSON 字段：`format: "soundbench-recording-v1"`、`audio`、`sample_rate`、
  `partition_position`、`tap_delays`。

然后：

```bash
PYTHONPATH=. python3 -m soundbench calibrate \
  --recording path/to/new_calibration.json \
  --out out/new_room.json

PYTHONPATH=. python3 -m soundbench build \
  --program path/to/program.json \
  --room out/new_room.json \
  --outdir out/delivery --strict
```

`samples/recordings/cal_new_probe.*` 就是“换了一套探测音”的样例：探测序列和长度
都不同，但房态相同；用它重新标定后已成功导出
`samples/deliverables/scene_change_new_probe/speaker_track.wav`。

## 关键交付物

运行 `./run_demo.sh` 后可查看：

| 需求 | 文件/命令结果 |
|---|---|
| 四段解说、切换时刻、每拍柜位、音量/泄漏上限 | `samples/programs/curator_program.json` |
| 确有一拍延迟尾音的换景节目 | `samples/programs/scene_change_program.json` |
| 三个隔板位置的固定音轨 | `samples/deliverables/fixed_partition_{open,half,closed}/speaker_track.wav` |
| 一条在所有隔板位置上验证通过的通用固定轨 | `samples/deliverables/universal_all_states/speaker_track.wav` |
| 四段解说主节目音轨 | `samples/deliverables/curator_original/speaker_track.wav` |
| 新探测音重新标定后的现场音轨 | `samples/deliverables/scene_change_new_probe/speaker_track.wav` |
| 未使用双路声音的逐帧误差 | `samples/reports/holdout_frame_errors.csv`、`holdout_summary.json` |
| 同/差、远柜也求和、拔扬声器三个规定案例 | `samples/diagnostics/diagnostics_summary.md` |
| 固定轨 vs 空拍探测对照 | `samples/reports/route_comparison.md` / `.json` |
| 无法区分房态时的可播稳健轨 | `samples/deliverables/robust_half_closed/speaker_track.wav` |
| 导出 WAV 在学得房间上的回放验听 | 每个交付目录下的 `playback_mics.wav` |
| 负例：拔掉第二只扬声器 | `samples/deliverables/unplugged_negative/verification.json` |

每个 build 目录包含：

- `speaker_track.wav`：可直接回放的两路扬声器 PCM-16 立体声；
- `verification.json`：逐句主动区误差、泄漏 RMS/peak、扬声器峰值、时间线；
- `cue_metrics.csv`：逐 cue 指标；
- `pcm_verification.json`：把导出的 WAV 读回后在学得房间上卷积的结果；
- `playback_mics.wav`：模拟的近柜/远柜话筒信号，可直接“验听”；
- `naive_ignores_tails.wav`：忽略尾音的对照轨，会在换景拍产生更大泄漏。

## 数学模型与尾音处理

学得的房间是稀疏 FIR：

\[
y_m[n]=\sum_{s=1}^2\sum_d h_{m,s,d}\,x_s[n-d].
\]

样例房的零延迟矩阵正是题目规定的和/差矩阵：

\[
H_0=
\begin{bmatrix}
1&1\\
1&-1
\end{bmatrix},
\qquad
H_0^{-1}=\frac12
\begin{bmatrix}
1&1\\
1&-1
\end{bmatrix}.
\]

因此近柜要 `1`、远柜要 `0` 时，天然得到 `x1=x2=target/2`，两路峰值各为 `0.5`。

主房还在 `d=3` 和 exactly `d=beat_samples=1000`（一拍）处有非零系数。
渲染时不是逐拍独立做矩阵反解，而是逐样本递归扣除历史尾音：

\[
H_0x[n]=d[n]-\sum_{d>0}H_d x[n-d].
\]

节目开头有 `lead_silence_beats` 个静音拍；正文后追加零目标 flush 样本和至少一个
最大响应延迟的释放静音。报告中的 `post_track_tail` 峰值为 `0`，说明文件结束后
没有未覆盖的响应尾巴。

忽略尾音的朴素反解在一拍尾音节目上会失败；在当前样例中第二句的远柜泄漏约为
RMS `4.1e-2`、peak `1.15e-1`，超过约定的 `0.02 / 0.06`。尾音感知轨在浮点计算中
逐 cue 误差约 `1e-16`，写成 PCM-16 后仍远低于阈值。

## 三个规定诊断案例

`samples/diagnostics/diagnostics_summary.md` 由 `python -m soundbench diagnose` 生成：

1. **无尾音和/差校准小样**：近柜收到两路之和，远柜收到两路之差；
   校准句峰值为 1，导出的两路各为一半，峰值 `0.5`，近柜 RMSE 和远柜泄漏约为
   `1e-17`（浮点）/ PCM 一个 LSB 量级。
2. **远柜也收到两路之和**：两行声学矩阵相同，系统不可隔离。仍导出
   `far_sum_attempt.wav`，其中原来的两半句让两个点都听到目标；报告明确标记
   `isolable=false`。
3. **恢复和/差校准后拔掉扬声器 2**：无尾音时 `near=x1`、`far=x1`，不可能同时满足
   近柜为 1、远柜为 0。`unplugged_speaker2_attempt.wav` 和主节目负例都标记失败；
   `build --strict --unplug 2` 会以退出码 3 表示约束不可行。

## 固定轨与探测路线

合成房有三个容许隔板位置：`partition_open`、`partition_half`（原始标定状态）、
`partition_closed`。

- **固定轨路线（已做通）**：事先为三个位置分别运行 `build`，三个 `--strict` 构建全部通过；
  另外用 `build-robust --rooms open half closed` 生成了一条通用 WAV，并在全部三个房间上
  验证通过（最大泄漏 RMS 约 `1.5e-2`，低于 `0.02`）。因此既可以按已知隔板位置选轨，
  也可以直接使用这条通用固定轨，完全不占用换景空拍。
- **短探测对照**：在换景拍打 128 样本已知探测音。只看前 160 样本的早期窗口时，
  `open` 可与另外两个状态区分，但 `half` 和 `closed` 的早期路径被设计为相同，
  因而进入同一不可区分分支。
- **不可区分分支的真实可播音轨**：`build-robust --rooms half closed` 反解系数平均
  房间，再把同一条 WAV 分别放到两个学得房间上检查；两个状态下都通过阈值。
- **完整探测对照**：等到一拍反射进入话筒后可区分全部状态，但需要约 1176/4000 秒，
  即约 0.294 秒后才能安全开讲；这比样例中的一个 1000 样本空拍更长。详见
  `samples/reports/route_comparison.md`。

## 命令速查

```bash
# 生成材料
PYTHONPATH=. python3 -m soundbench generate --out samples

# 标定
PYTHONPATH=. python3 -m soundbench calibrate \
  --recording samples/recordings/cal_original.json \
  --out samples/learned/original.json

# 留出双路声音的逐帧误差
PYTHONPATH=. python3 -m soundbench evaluate-holdout \
  --learned samples/learned/original.json \
  --recording samples/recordings/holdout.json \
  --outdir samples/reports

# 导出/验听节目
PYTHONPATH=. python3 -m soundbench build \
  --program samples/programs/scene_change_program.json \
  --room samples/learned/original.json \
  --outdir /tmp/scene --strict

# 仅重新验证已有 WAV
PYTHONPATH=. python3 -m soundbench verify \
  --track /tmp/scene/speaker_track.wav \
  --room samples/learned/original.json \
  --out /tmp/scene/reverify.json \
  --mics-wav /tmp/scene/mics.wav

# 规定诊断
PYTHONPATH=. python3 -m soundbench diagnose --samples samples

# 房态探测与稳健轨
PYTHONPATH=. python3 -m soundbench classify-probe \
  --recording samples/probes/live_probe_partition_closed.json \
  --rooms samples/learned/partition_{open,half,closed}.json \
  --early

PYTHONPATH=. python3 -m soundbench build-robust \
  --program samples/programs/scene_change_program.json \
  --rooms samples/learned/partition_half.json samples/learned/partition_closed.json \
  --outdir /tmp/robust --strict
```

## 目录结构

```text
soundbench/
  wavio.py       # PCM-16 WAV 读写
  linalg.py      # 小二乘、2x2 求解
  ir.py          # 稀疏 FIR、卷积、平均房间
  synth.py       # 确定性合成房间/探测音/解说/节目
  calibrate.py   # 从 4 通道录音学习 IR
  metrics.py     # 逐帧误差与泄漏指标
  render.py      # 尾音感知反解、朴素对照、稳健设计
  verify.py      # 读回 WAV 并在学得房间上回放验证
  diagnose.py    # 三个规定教学案例
  routes.py      # 固定轨/探测路线对照
  cli.py         # 命令行
tests/           # 标准库 unittest 测试
examples/        # 可替换节目样例
samples/         # 一键脚本生成的素材、报告和交付物
```
