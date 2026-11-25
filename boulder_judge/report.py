"""Human-readable Markdown deliverables."""
from __future__ import annotations

from typing import Any


def render_decision_markdown(decision: dict[str, Any]) -> str:
    rsum = decision.get("route_summary", {})
    lines = [
        f"# 裁判结果 `{decision['clip_id']}`",
        "",
        f"- 路线卡：`{decision['route_id']}`",
        f"- 顶点：`{rsum.get('top_vertex')}`；区点：`{rsum.get('zone_vertex')}`；"
        f"稳定帧数：`{rsum.get('stable_frames')}`；截止帧（含）：`{rsum.get('deadline_frame')}`",
        f"- 模型：`{decision['model_version']}`",
        f"- 模型指纹：`{decision['model_fingerprint'][:16]}…`",
        f"- 签单阈值：{decision['threshold']:.2f}",
        f"- 状态：**{decision['status']}**",
        f"- 正式签单类别：**{decision['signed_class'] or '无；可申诉'}**",
        "",
        "## 三类分数的全轨迹概率",
        "",
        "| 类别 | 含义 | 概率 |",
        "|---|---|---:|",
    ]
    for row in decision["score_probabilities"]:
        lines.append(f"| `{row['code']}` | {row['label_zh']} | {row['probability']:.6f} |")
    lines.extend(
        [
            "",
            f"- 全片至少一帧双手同触顶点的后验概率：{decision['p_any_simultaneous_top_in_clip']:.4f}",
            f"- 截止前至少一帧双手同触顶点的后验概率：{decision['p_any_simultaneous_top_before_deadline']:.4f}",
            f"- 达到 {decision['threshold']:.2f} 逐帧证据阈值的帧：{decision['high_confidence_simultaneous_top_frames'] or '无'}",
            f"- 后验概率 ≥0.5 的候选同顶帧：{decision['probable_simultaneous_top_frames'] or '无'}",
            "",
            "## 逐帧联合证据",
            "",
            "| 帧 | 读数(左/右) | P(左手且右手在顶点) | 证据档 | 后验最高联合状态 |",
            "|---:|---|---:|---|---|",
        ]
    )
    for fr in decision["frame_evidence"]:
        lines.append(
            f"| {fr['frame']} | {fr['readings']['left']}/{fr['readings']['right']} | "
            f"{fr['p_both_at_top_vertex']:.4f} | {fr['evidence_band']} | "
            f"{fr['posterior_joint_state'][0]}/{fr['posterior_joint_state'][1]} "
            f"({fr['posterior_dominant_probability']:.3f}) |"
        )
    lines.extend(["", "## 最可能的完整联合轨迹", "", "| 帧 | 左手触点 | 右手触点 |", "|---:|---|---|"])
    for x in decision["most_likely_joint_path"]:
        lines.append(f"| {x['frame']} | {x['left_contact']} | {x['right_contact']} |")
    lines.append("")
    return "\n".join(lines)
