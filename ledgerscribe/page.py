"""全页对平、入册签字与两格工时的复核规划（路线 A：局部辨字）。

规则（题面）：
  1. 行金额概率 >= 本行最高金额概率 × threshold_ratio 才有资格；
  2. 只在“有资格且整页合计等于全页总数”的读法组合里求解；
  3. 所有对平读法一致的金额才能签字，不一致留白；不存在对平读法 -> 冲突单；
  4. 界面只显示前三金额，签字覆盖全部候选（第四候选也可被签出）。
"""
from __future__ import annotations

import itertools

from .segments import apply_answer, feasible_answers, split_segments

DISPLAY_TOP = 3


def eligible_amounts(inferred: dict, ratio: float) -> list[str]:
    if not inferred:
        return []
    pmax = max(e["prob"] for e in inferred.values())
    return sorted(
        [a for a, e in inferred.items() if e["prob"] >= ratio * pmax - 1e-12],
        key=lambda a: -inferred[a]["prob"],
    )


def reconcile(per_line_inferred: dict, page: dict) -> dict:
    ratio = page["threshold_ratio"]
    line_ids = page["line_ids"]
    eligible = {
        lid: eligible_amounts(per_line_inferred.get(lid, {}), ratio)
        for lid in line_ids
    }
    rankings = {
        lid: sorted(
            per_line_inferred.get(lid, {}).items(),
            key=lambda kv: -kv[1]["prob"],
        )
        for lid in line_ids
    }

    balanced = []
    if all(eligible.values()):
        for combo in itertools.product(*(eligible[lid] for lid in line_ids)):
            if sum(int(a) for a in combo) == page["total"]:
                balanced.append(dict(zip(line_ids, combo)))

    if not balanced:
        status = "conflict"
        signed = {lid: None for lid in line_ids}
    else:
        status = "reconciled"
        signed = {}
        for lid in line_ids:
            vals = {c[lid] for c in balanced}
            signed[lid] = next(iter(vals)) if len(vals) == 1 else None

    display = {}
    for lid in line_ids:
        top = [
            {"amount": a, "prob": round(e["prob"], 4)} for a, e in rankings[lid]
        ][:DISPLAY_TOP]
        n_more = max(0, len(rankings[lid]) - DISPLAY_TOP)
        display[lid] = {"top3": top, "n_more": n_more}

    return {
        "page_id": page["page_id"],
        "status": status,
        "eligible": eligible,
        "balanced_readings": balanced,
        "signed": signed,
        "display": display,
    }


def n_signed(result: dict) -> int:
    return sum(1 for v in result["signed"].values() if v is not None)


# --------------------------------------------------------------------------
# 路线 A 复核规划：至多两次局部辨字提问，选最坏答复下仍能多签最多行的回查
# --------------------------------------------------------------------------
def _answer_universe(inferred_state, tokens: list, lid: str, seg_id: int) -> list[dict]:
    """该笔段在当前后验下实际可达的答复空间（穷举切法收集，而非全 0..9）。"""
    return feasible_answers(inferred_state[lid], tokens, seg_id)


def plan_review(
    per_line_inferred: dict, tokens_by_line: dict, page: dict, budget: int = 2
) -> dict:
    """穷举至多 budget 次局部辨字提问，选最坏整字答复下签行最多的回查。

    返回结构化策略：首问笔段、每个整字答复后的第二问（若需要）与预期签行。
    not_whole 不指认字，单独在 not_whole_risk 中如实标注其后果。
    """
    base = reconcile(per_line_inferred, page)
    baseline = n_signed(base)
    line_ids = page["line_ids"]
    query_spots = [
        (lid, s["seg_id"])
        for lid in line_ids
        for s in split_segments(tokens_by_line[lid])
    ]

    def best_followup(state):
        """选一个提问使最坏整字答复的签行数最大，返回 (行数, 提问或 None)。"""
        best_n, best_spot = n_signed(reconcile(state, page)), None
        for lid, seg_id in query_spots:
            vals = []
            for ans in _answer_universe(state, tokens_by_line[lid], lid, seg_id):
                if ans["kind"] != "digit":
                    continue
                nxt = dict(state)
                nxt[lid] = apply_answer(state[lid],
                                        tokens_by_line[lid], seg_id, ans)
                vals.append(n_signed(reconcile(nxt, page)))
            if vals and min(vals) > best_n:
                best_n, best_spot = min(vals), {"line_id": lid, "seg_id": seg_id}
        return best_n, best_spot

    candidates = []
    for lid, seg_id in query_spots:
        answers = _answer_universe(per_line_inferred,
                                   tokens_by_line[lid], lid, seg_id)
        digit_answers = [a for a in answers if a["kind"] == "digit"]
        if not digit_answers:
            continue
        per_answer = []
        for ans in digit_answers:
            nxt = dict(per_line_inferred)
            nxt[lid] = apply_answer(per_line_inferred[lid],
                                    tokens_by_line[lid], seg_id, ans)
            immediate = n_signed(reconcile(nxt, page))
            if budget > 1:
                follow_n, follow_spot = best_followup(nxt)
            else:
                follow_n, follow_spot = immediate, None
            achieved = max(immediate, follow_n)
            per_answer.append({
                "answer": ans,
                "signed_after_answer": immediate,
                "second_query": follow_spot,
                "expected_signed": achieved,
            })
        worst_n = min(a["expected_signed"] for a in per_answer)
        candidates.append({
            "query": {"line_id": lid, "seg_id": seg_id},
            "worst_case_signed": worst_n,
            "answers": per_answer,
        })

    # 排序：先按最坏签行数，再按“无需第二问即可签出”的答复数，最后按笔段号。
    candidates.sort(key=lambda c: (
        -c["worst_case_signed"],
        -sum(1 for a in c["answers"]
             if a["second_query"] is None and a["signed_after_answer"] > 0),
        c["query"]["line_id"], c["query"]["seg_id"],
    ))
    chosen = candidates[0]
    not_whole_risk = []
    for lid, seg_id in query_spots:
        if any(a["kind"] == "not_whole"
               for a in _answer_universe(per_line_inferred,
                                         tokens_by_line[lid], lid, seg_id)):
            nxt = dict(per_line_inferred)
            nxt[lid] = apply_answer(per_line_inferred[lid],
                                    tokens_by_line[lid], seg_id,
                                    {"kind": "not_whole"})
            not_whole_risk.append({
                "line_id": lid, "seg_id": seg_id,
                "signed_if_not_whole": n_signed(reconcile(nxt, page)),
            })
    return {
        "page_id": page["page_id"],
        "baseline_signed": baseline,
        "worst_case_signed": chosen["worst_case_signed"],
        "strategy": chosen,
        "alternative_queries": candidates[1:],
        "not_whole_risk": not_whole_risk,
        "note": "strategy.query 为首问笔段；answers 给出每个整字答复后的第二问与签行。",
    }


def apply_review_dialogue(
    per_line_inferred: dict, tokens_by_line: dict, page: dict, replies: list[dict]
) -> dict:
    """收到实际复核回复后重跑对平。replies: [{line_id, seg_id, answer}]。"""
    state = {k: v for k, v in per_line_inferred.items()}
    transcript = []
    for r in replies:
        lid = r["line_id"]
        state[lid] = apply_answer(
            state[lid], tokens_by_line[lid], r["seg_id"], r["answer"]
        )
        transcript.append(r)
    result = reconcile(state, page)
    result["replies"] = transcript
    result["signing_basis"] = _basis(result)
    return result


def _basis(result: dict) -> list[str]:
    out = []
    for lid, v in result["signed"].items():
        if v is None:
            out.append(f"{lid}: 留白（对平读法不唯一或无解）")
        else:
            out.append(
                f"{lid}: 签 {v}，依据全部 {len(result['balanced_readings'])} 个"
                f"有资格且对平的读法在该行一致"
            )
    return out
