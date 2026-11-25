# 手算小例核对

全部期望值由程序复现：**是**

| 小例 | 手算路径 | 程序 TOP 概率 | 期望 |
|---|---|---:|---:|
| `half_HH_half_EE_one_frame_K1` | P(EE)=0.5，P(HH)=0.5；只有 HH 满足完攀，因此 P(TOP)=0.5。 | 0.5 | 0.5 |
| `always_exactly_one_hand_K1` | P(EH,HE,EH,HE)=0.5，P(HE,EH,HE,EH)=0.5。两条路径都没有 HH，所以即使每只手腕边缘概率均为 0.5，P(TOP)=0。 | 0 | 0 |
| `empty_then_final_HH_K1` | 唯一路径为 EE -> HH；K=1 时末帧满足，因此 P(TOP)=1。 | 1 | 1 |
| `empty_then_final_HH_K2` | 唯一路径为 EE -> HH。它没有连续两帧 HH，因此把 stable_frames 改为 2 后 P(TOP)=0。 | 0 | 0 |

## 手腕边缘概率（前两例）

### `half_HH_half_EE_one_frame_K1`
- frame 0: P(左手触顶/握)=0.500, P(右手触顶/握)=0.500
### `always_exactly_one_hand_K1`
- frame 0: P(左手触顶/握)=0.500, P(右手触顶/握)=0.500
- frame 1: P(左手触顶/握)=0.500, P(右手触顶/握)=0.500
- frame 2: P(左手触顶/握)=0.500, P(右手触顶/握)=0.500
- frame 3: P(左手触顶/握)=0.500, P(右手触顶/握)=0.500
