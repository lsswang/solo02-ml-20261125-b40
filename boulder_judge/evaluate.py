"""Held-out whole-clip checks against manual two-hand labels."""
from __future__ import annotations

from typing import Any

from .inference import CLASS_ORDER, adjudicate, posterior_marginals
from .mathutils import js_divergence, tv_distance
from .model import JointModel, pair_key


def labelled_route_class(clip: dict[str, Any], route: dict[str, Any]) -> str:
    """Oracle deterministic class from manual frame labels."""
    top = route["top_vertex"]
    zone = route["zone_vertex"]
    k = int(route["stable_frames"])
    deadline = route.get("deadline_frame")
    req = route.get("zone_requires", "any_hand")
    run = 0
    topped = False
    zoned = False
    for t, fr in enumerate(clip["frames"]):
        active = deadline is None or t <= int(deadline)
        lc, rc = fr["left_contact"], fr["right_contact"]
        if active and lc == top and rc == top:
            run += 1
            if run >= k:
                topped = True
        else:
            run = 0
        if active:
            hit = (lc == zone or rc == zone) if req == "any_hand" else (lc == zone and rc == zone)
            zoned = zoned or hit
    if topped:
        return "TOP"
    return "ZONE" if zoned else "NO_ZONE"


def evaluate_corpus(
    model: JointModel,
    corpus: dict[str, Any],
    route: dict[str, Any],
    *,
    split: str = "heldout",
) -> dict[str, Any]:
    clips = [c for c in corpus.get("clips", []) if c.get("split") == split]
    if not clips:
        # A swapped corpus may not mark splits; treat every clip as held-out
        # when the caller explicitly asks for evaluation and none match.
        clips = list(corpus.get("clips", []))
    if not clips:
        raise ValueError("no clips available for held-out check")

    state_counts = [0.0] * len(model.states)
    posterior_sums = [0.0] * len(model.states)
    total_frames = 0
    tt_index = model.state_index.get(pair_key((route["top_vertex"], route["top_vertex"])))
    brier_sum = 0.0
    argmax_correct = 0
    high_conf_tp = high_conf_fp = high_conf_fn = 0
    probable_tp = probable_fp = probable_fn = 0
    clip_rows = []
    class_count_true = {c: 0 for c in CLASS_ORDER}
    class_count_signed = {c: 0 for c in CLASS_ORDER}

    for clip in clips:
        decision = adjudicate(model, clip, route)
        gamma, _ = posterior_marginals(model, clip)
        true_class = labelled_route_class(clip, route)
        class_count_true[true_class] += 1
        if decision["signed_class"]:
            class_count_signed[decision["signed_class"]] += 1

        for t, fr in enumerate(clip["frames"]):
            si = model.state_index[pair_key((fr["left_contact"], fr["right_contact"]))]
            state_counts[si] += 1
            for s, p in enumerate(gamma[t]):
                posterior_sums[s] += p
            total_frames += 1
            if tt_index is not None:
                y = 1.0 if si == tt_index else 0.0
                p = gamma[t][tt_index]
                brier_sum += (p - y) ** 2
                pred_high = p >= float(route.get("threshold", 0.95))
                pred_prob = p >= 0.5
                if pred_high and y:
                    high_conf_tp += 1
                elif pred_high and not y:
                    high_conf_fp += 1
                elif not pred_high and y:
                    high_conf_fn += 1
                if pred_prob and y:
                    probable_tp += 1
                elif pred_prob and not y:
                    probable_fp += 1
                elif not pred_prob and y:
                    probable_fn += 1
            dominant = max(range(len(gamma[t])), key=lambda s: gamma[t][s])
            argmax_correct += int(dominant == si)

        clip_rows.append(
            {
                "clip_id": clip.get("clip_id"),
                "known_truth_outcome": clip.get("known_truth_outcome"),
                "manual_route_class": true_class,
                "predicted_probabilities": {
                    x["code"]: x["probability"] for x in decision["score_probabilities"]
                },
                "signed_class": decision["signed_class"],
                "status": decision["status"],
                "high_confidence_top_frames": decision["high_confidence_simultaneous_top_frames"],
                "p_any_tt": decision["p_any_simultaneous_top_before_deadline"],
            }
        )

    empirical = [c / total_frames for c in state_counts]
    posterior_mean = [p / total_frames for p in posterior_sums]
    state_table = []
    for s, state in enumerate(model.states):
        state_table.append(
            {
                "state": state,
                "manual_frequency": empirical[s],
                "mean_posterior_probability": posterior_mean[s],
                "absolute_error": abs(empirical[s] - posterior_mean[s]),
            }
        )

    def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = tp / (tp + fp) if tp + fp else 1.0 if fn == 0 else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    signed_correct = sum(
        1 for row in clip_rows if row["signed_class"] == row["manual_route_class"]
    )
    signed_total = sum(1 for row in clip_rows if row["signed_class"] is not None)
    return {
        "format": "boulder-heldout-report/v1",
        "split": split,
        "clip_count": len(clips),
        "frame_count": total_frames,
        "joint_distribution_check": {
            "table": state_table,
            "total_variation": tv_distance(empirical, posterior_mean),
            "jensen_shannon_nats": js_divergence(empirical, posterior_mean),
            "max_absolute_state_error": max(row["absolute_error"] for row in state_table),
        },
        "frame_joint_state_argmax_accuracy": argmax_correct / total_frames,
        "both_top_brier_score": brier_sum / total_frames if tt_index is not None else None,
        "both_top_detection_at_signing_threshold": prf(high_conf_tp, high_conf_fp, high_conf_fn),
        "both_top_detection_at_0.5": prf(probable_tp, probable_fp, probable_fn),
        "manual_class_counts": class_count_true,
        "signed_class_counts": class_count_signed,
        "signed_accuracy": signed_correct / signed_total if signed_total else None,
        "signed_coverage": signed_total / len(clips),
        "clips": clip_rows,
    }


def render_heldout_markdown(report: dict[str, Any]) -> str:
    j = report["joint_distribution_check"]
    hi = report["both_top_detection_at_signing_threshold"]
    mid = report["both_top_detection_at_0.5"]

    def fmt_prf(x: dict[str, float]) -> str:
        return (
            f"P={x['precision']:.3f}, R={x['recall']:.3f}, F1={x['f1']:.3f} "
            f"(TP={x['tp']}, FP={x['fp']}, FN={x['fn']})"
        )

    lines = [
        "# 整段留出短片核对报告",
        "",
        f"- 留出短片：{report['clip_count']} 段；总帧数：{report['frame_count']}",
        f"- 联合状态分布总变差 TV：**{j['total_variation']:.4f}**",
        f"- Jensen–Shannon 散度：{j['jensen_shannon_nats']:.5f} nats",
        f"- 最大单状态绝对误差：{j['max_absolute_state_error']:.4f}",
        f"- 逐帧联合状态 argmax 准确率：{report['frame_joint_state_argmax_accuracy']:.3f}",
        f"- 双手同顶 Brier 分数：{report['both_top_brier_score']:.4f}",
        f"- 单帧 ≥0.95 检测：{fmt_prf(hi)}",
        f"- 单帧 ≥0.50 候选检测：{fmt_prf(mid)}",
        f"- 签单覆盖率：{report['signed_coverage']:.3f}；已签单准确率：{report['signed_accuracy']}",
        "",
        "## 联合触点分布（不是两只手边缘分布的乘积）",
        "",
        "| 左手 | 右手 | 人工标签频率 | 平均后验 | 绝对差 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in j["table"]:
        lines.append(
            f"| {row['state'][0]} | {row['state'][1]} | {row['manual_frequency']:.4f} | "
            f"{row['mean_posterior_probability']:.4f} | {row['absolute_error']:.4f} |"
        )
    lines.extend(["", "## 每段短片", "", "| 片段 | 人工路线类别 | TOP | ZONE | NO_ZONE | 签单 |", "|---|---|---:|---:|---:|---|"])
    for c in report["clips"]:
        p = c["predicted_probabilities"]
        lines.append(
            f"| {c['clip_id']} | {c['manual_route_class']} | {p['TOP']:.3f} | {p['ZONE']:.3f} | "
            f"{p['NO_ZONE']:.3f} | {c['signed_class'] or '未签单'} |"
        )
    lines.append("")
    return "\n".join(lines)
