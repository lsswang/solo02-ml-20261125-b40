"""模型检查：极短轨迹上的独立第二实现（前向网格 DP，唯一转移语义）。

主实现 model.Model.enumerate_paths 是逐 token 的递归回溯；这里用完全不同的
“按笔推进”前向 DP：在进入每支笔时一次性决定它吞几个 token（1..MAX_LEN），
因此每条切分只有一条产生路径，不存在重复计数。状态为 (token 位置, 笔序号)，
直接累加到达终态的总概率质量，不构造 assign、不回溯。

两套实现必须在极短轨迹上逐金额对平，后验必须归一。
"""
from __future__ import annotations

import math

from .synth import TEMPLATES

EPS = 1e-12
MAX_LEN = 2


def _forward_log_mass(model, tokens: list, digits: str) -> float:
    """按笔推进的前向 DP（对数空间），返回未归一对数质量，避免下溢。"""
    nstr = [len(TEMPLATES[d]) for d in digits]
    flat = [
        (di, si, digits[di])
        for di in range(len(digits))
        for si in range(nstr[di])
    ]
    total_strokes = len(flat)
    n = len(tokens)
    neg = -math.inf
    logdp = {(0, 0): 0.0}
    queue = [(0, 0)]

    def relax(key, add_lp):
        cur = logdp.get(key, neg)
        if cur == neg:
            logdp[key] = add_lp
            queue.append(key)
        else:
            logdp[key] = cur + math.log1p(math.exp(add_lp - cur))

    while queue:
        pos, fi = queue.pop(0)
        if fi == total_strokes:
            continue
        di, si, ch = flat[fi]
        base = logdp[(pos, fi)]
        for length in range(1, MAX_LEN + 1):
            hi = pos + length
            if hi > n or any(tokens[t] == "U" for t in range(pos, hi)):
                continue
            pen_lp = sum(
                math.log(model.emit_p(ch, si, tokens[t]))
                for t in range(pos, hi)
            ) + math.log(model.len_p(ch, si, length))
            if fi + 1 == total_strokes:
                if hi == n:
                    relax((hi, fi + 1), base + pen_lp)
                continue
            nxt_di = flat[fi + 1][0]
            q = (
                model.u_internal_p(ch, si)
                if nxt_di == di
                else model.u_boundary_p
            )
            relax((hi, fi + 1), base + pen_lp + math.log(1.0 - q))
            if hi < n and tokens[hi] == "U":
                relax((hi + 1, fi + 1), base + pen_lp + math.log(q))
    return logdp.get((n, total_strokes), neg)


# 兼容旧名称
def _forward_mass(model, tokens: list, digits: str) -> float:
    return math.exp(_forward_log_mass(model, tokens, digits))


def independent_check(model, lines: dict, page: dict) -> dict:
    # 只挑极短轨迹（<=12 个 token），保证检查可人工复核。
    from .model import DIGIT_PRIOR

    report = {"page_id": page["page_id"], "lines": [], "all_match": True}
    for lid in page["line_ids"]:
        tokens = lines[lid]["tokens"]
        if len(tokens) > 12:
            continue
        main_infer = model.infer_line(tokens, page["candidates"][lid])
        dp_logs = {
            a: _forward_log_mass(model, tokens, a)
            + len(a) * math.log(DIGIT_PRIOR)
            for a in page["candidates"][lid]
            if len(a) <= 3
        }
        # 剔除独立 DP 判定为不可行（-inf）的金额，与主实现候选集保持一致。
        dp_logs = {a: v for a, v in dp_logs.items() if v != -math.inf}
        lse = max(dp_logs.values())
        denom = lse + math.log(
            sum(math.exp(v - lse) for v in dp_logs.values())
        )
        rows, norm_dp, norm_main = [], 0.0, 0.0
        for amount, dp_log in dp_logs.items():
            dp_post = math.exp(dp_log - denom)
            main_post = main_infer.get(amount, {"prob": 0.0})["prob"]
            norm_dp += dp_post
            norm_main += main_post
            match = abs(math.log(max(dp_post, EPS))
                        - math.log(max(main_post, EPS))) < 1e-6
            report["all_match"] &= match
            rows.append(
                {
                    "amount": amount,
                    "prob_main": round(main_post, 8),
                    "prob_independent_forward": round(dp_post, 8),
                    "match": match,
                }
            )
        report["lines"].append(
            {
                "line_id": lid,
                "n_tokens": len(tokens),
                "posterior_sums_to_one": (
                    abs(norm_main - 1.0) < 1e-9 and abs(norm_dp - 1.0) < 1e-9
                ),
                "amounts": rows,
            }
        )
    return report
