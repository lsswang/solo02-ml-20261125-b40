from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .data import grouped_blind_split, load_tasting_votes
from .model import BlendModel, fit_blend_model
from .utils import ensure_dir, safe_prob, write_json


def weighted_log_loss(items: Sequence[Tuple[int, float, float]]) -> float:
    """items are (y, probability, weight)."""
    total_w = 0.0
    total = 0.0
    for y, p, w in items:
        pc = safe_prob(p)
        total -= w * (y * math.log(pc) + (1 - y) * math.log(1 - pc))
        total_w += w
    if total_w <= 0:
        raise ValueError("no positive weights for log loss")
    return total / total_w


def train_and_evaluate(
    records_path: str,
    out_dir: str,
    *,
    holdout_roast: Optional[str] = None,
    threshold: float = 0.8,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    votes = load_tasting_votes(records_path)
    train_idx, blind_idx, split_audit = grouped_blind_split(votes, holdout_roast=holdout_roast)

    train_votes = [votes[i] for i in train_idx]
    blind_votes = [votes[i] for i in blind_idx]

    blind_fit = fit_blend_model(train_votes)
    production_model = fit_blend_model(votes)

    eval_items: List[Tuple[int, float, float]] = []
    baseline_items: List[Tuple[int, float, float]] = []
    train_pos_w = sum(v.y * v.weight for v in train_votes)
    train_w = sum(v.weight for v in train_votes)
    base_p = (train_pos_w + 0.5) / (train_w + 1.0)  # small Bayesian shrink

    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for v in blind_votes:
        p = blind_fit.predict_proba(v.composition)
        eval_items.append((v.y, p, v.weight))
        baseline_items.append((v.y, base_p, v.weight))
        key = (v.roast_id, v.formula_id, v.round_no)
        g = groups.setdefault(
            key,
            {
                "roast_id": v.roast_id,
                "formula_id": v.formula_id,
                "round_no": v.round_no,
                "judges": set(),
                "weight_n": 0.0,
                "yes_weight": 0.0,
                "predicted_probability": p,
                "composition": dict(sorted(v.composition.items())),
            },
        )
        g["weight_n"] += v.weight
        g["yes_weight"] += v.y * v.weight
        if v.judge_id and v.judge_id != "aggregate":
            g["judges"].add(v.judge_id)

    model_loss = weighted_log_loss(eval_items)
    baseline_loss = weighted_log_loss(baseline_items)

    group_reports: List[Dict[str, Any]] = []
    group_loss_items: List[Tuple[int, float, float]] = []
    for key, g in sorted(groups.items()):
        n = g["weight_n"]
        yes = g["yes_weight"]
        empirical = yes / n if n else None
        pred = float(g["predicted_probability"])
        # Expand only for group log-loss calculation via weighted binomial.
        if empirical is not None:
            group_loss_items.append((1, pred, yes))
            group_loss_items.append((0, pred, n - yes))
        group_reports.append(
            {
                "roast_id": g["roast_id"],
                "formula_id": g["formula_id"],
                "round_no": g["round_no"],
                "distinct_judges": len(g["judges"]),
                "vote_weight": n,
                "similar_weight": yes,
                "empirical_similarity": empirical,
                "predicted_similarity": pred,
                "prediction_error_abs": None if empirical is None else abs(pred - empirical),
                "passes_gate_at_threshold": bool(pred >= threshold),
                "composition": g["composition"],
            }
        )

    group_log_loss = weighted_log_loss(group_loss_items) if group_loss_items else None
    per_roast: Dict[str, Dict[str, float]] = {}
    for v in blind_votes:
        p = blind_fit.predict_proba(v.composition)
        r = per_roast.setdefault(v.roast_id, {"weight": 0.0, "loss_sum": 0.0, "yes": 0.0})
        pc = safe_prob(p)
        r["loss_sum"] -= v.weight * (v.y * math.log(pc) + (1 - v.y) * math.log(1 - pc))
        r["weight"] += v.weight
        r["yes"] += v.y * v.weight
    roast_report = [
        {
            "roast_id": rid,
            "vote_weight": vals["weight"],
            "empirical_similarity": vals["yes"] / vals["weight"],
            "log_loss": vals["loss_sum"] / vals["weight"],
        }
        for rid, vals in sorted(per_roast.items())
    ]

    report = {
        "records_path": os.path.abspath(records_path),
        "threshold": threshold,
        "split": split_audit,
        "metrics": {
            "blind_vote_log_loss": model_loss,
            "blind_group_log_loss": group_log_loss,
            "intercept_baseline_log_loss": baseline_loss,
            "log_loss_improvement_vs_baseline": baseline_loss - model_loss,
            "blind_weighted_votes": sum(w for _, _, w in eval_items),
        },
        "per_roast": roast_report,
        "groups": group_reports,
        "blind_model_coefficients": blind_fit.coefficient_map(),
    }

    write_json(os.path.join(out_dir, "split_audit.json"), split_audit)
    write_json(os.path.join(out_dir, "blind_evaluation.json"), report)
    write_json(os.path.join(out_dir, "blind_model.json"), blind_fit.to_dict())
    write_json(os.path.join(out_dir, "model.json"), production_model.to_dict())

    md = render_evaluation_markdown(report)
    with open(os.path.join(out_dir, "blind_evaluation.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return {
        "report": report,
        "blind_model": blind_fit,
        "production_model": production_model,
        "train_idx": train_idx,
        "blind_idx": blind_idx,
    }


def render_evaluation_markdown(report: Dict[str, Any]) -> str:
    m = report["metrics"]
    split = report["split"]
    lines = [
        "# 盲评留出报告",
        "",
        f"- 划分方式：`{split['strategy']}`",
        f"- 指定留次：`{split.get('holdout_roast_requested')}`",
        f"- 训练票数（含权重）：`{split['train_rows']}` 行；盲评行数：`{split['blind_rows']}`",
        f"- 盲评烘次：`{', '.join(split['blind_roasts'])}`",
        f"- 盲评配方：`{', '.join(split['blind_formulas'])}`",
        f"- 整烘次泄漏违规：`{len(split['whole_roast_violations'])}`",
        f"- 同配方拆分裂规：`{len(split['same_formula_split_violations'])}`",
        "",
        "## 对数损失",
        "",
        f"- 模型逐票 log loss：**{m['blind_vote_log_loss']:.6f}**",
        f"- 按评尝组聚合后的 log loss：`{m['blind_group_log_loss']}`",
        f"- 常数基线 log loss：`{m['intercept_baseline_log_loss']:.6f}`",
        f"- 相对基线改善：`{m['log_loss_improvement_vs_baseline']:.6f}`",
        "",
        "## 逐烘次",
        "",
        "| 烘次 | 票数 | 经验相似率 | log loss |",
        "|---|---:|---:|---:|",
    ]
    for r in report["per_roast"]:
        lines.append(
            f"| {r['roast_id']} | {r['vote_weight']:.0f} | "
            f"{r['empirical_similarity']:.3f} | {r['log_loss']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 盲评配方预测",
            "",
            "| 烘次 | 配方 | 轮次 | 评委数 | 预测 | 实际相似率 | 过门槛 |",
            "|---|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for g in report["groups"]:
        emp = "" if g["empirical_similarity"] is None else f"{g['empirical_similarity']:.3f}"
        judges = g["distinct_judges"] or int(g["vote_weight"])
        lines.append(
            f"| {g['roast_id']} | {g['formula_id']} | {g['round_no']} | {judges} | "
            f"{g['predicted_similarity']:.3f} | {emp} | "
            f"{'是' if g['passes_gate_at_threshold'] else '否'} |"
        )
    lines.append("")
    return "\n".join(lines)
