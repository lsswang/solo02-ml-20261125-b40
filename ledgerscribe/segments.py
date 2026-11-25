"""笔段、字归属责任与局部辨字复核（馆里两格工时的路线 A）。

笔段定义：原行 token 序列中被抬笔 U 切开的方向 token 区间，用 token 下标
[lo, hi] 表示。馆员锁定笔段时，约束的就是这个原行字符区间。

复核答复只有两种：
  * {"kind": "digit", "digit": "7"}   该笔段单独构成一个整字，字为 7；
  * {"kind": "not_whole"}              此处不是一个整字（即它跨字界或只是某字的一笔）。
一种切法被排除后，同金额的其他合法切法仍保留——过滤发生在切法层而非金额层。
"""
from __future__ import annotations

import math

from .model import DIGIT_PRIOR
from .synth import TEMPLATES


def split_segments(tokens: list) -> list[dict]:
    segs, lo = [], 0
    i = 0
    while i < len(tokens):
        if tokens[i] == "U":
            segs.append({"seg_id": len(segs), "lo": lo, "hi": i - 1})
            i += 1
            lo = i
            continue
        i += 1
    segs.append({"seg_id": len(segs), "lo": lo, "hi": len(tokens) - 1})
    return segs


def segment_digit_responsibility(
    tokens: list, inferred: dict, amount: str
) -> list[dict]:
    """行报告：每个笔段属于哪个字（数字位 0..k-1）的后验概率。

    同金额的多种切法按各自联合概率加权后相加。
    """
    segs = split_segments(tokens)
    entry = inferred.get(amount)
    if entry is None:
        return []
    z_log = max(p["logp"] for p in entry["paths"])
    log_norm = z_log + math.log(
        sum(math.exp(p["logp"] - z_log) for p in entry["paths"])
    )
    rows = []
    for s in segs:
        resp = {di: 0.0 for di in range(len(amount))}
        for p in entry["paths"]:
            w = math.exp(p["logp"] - log_norm)
            owners = set()
            for ti in range(s["lo"], s["hi"] + 1):
                a = p["assign"][ti]
                if a is not None:
                    owners.add(a[0])
            # 笔段跨字界（连写）时，对涉及的每个字各记一份责任。
            for di in owners:
                resp[di] += w
        rows.append(
            {
                "seg_id": s["seg_id"],
                "token_span": [s["lo"], s["hi"]],
                "responsibility": {str(k): round(v, 4) for k, v in resp.items()},
            }
        )
    return rows


def _segment_owner_digit(path: dict, seg: dict) -> tuple[bool, str | None]:
    """该切法下，笔段是否单独构成一个整字；是则返回该数字字符。

    判定：笔段的 token 区间恰好等于某个数字在该切法中的完整 span，
    且笔段不跨字界。多笔字若被一笔段完整覆盖（连写合并）也算整字。
    """
    owners = set()
    for ti in range(seg["lo"], seg["hi"] + 1):
        a = path["assign"][ti]
        if a is not None:
            owners.add(a[0])
    if len(owners) != 1:
        return False, None
    di = owners.pop()
    lo, hi = path["spans"][di]
    if lo == seg["lo"] and hi == seg["hi"]:
        digit = str(di)
        # 从 assign 反查该位数字字符不可得，由调用方按金额映射；
        # 这里返回位号，整字字符在 apply_answer 中结合 amount 解析。
        return True, digit
    return False, None


def apply_answer(
    inferred: dict, tokens: list, seg_id: int, answer: dict
) -> dict:
    """按一份复核答复过滤全部金额下的切法，并重新归一化。"""
    seg = split_segments(tokens)[seg_id]
    out = {}
    for amount, entry in inferred.items():
        kept = [
            p for p in entry["paths"]
            if _answer_ok(p, seg, answer, amount)
        ]
        if kept:
            log_total = max(p["logp"] for p in kept) + math.log(
                sum(math.exp(p["logp"] - max(q["logp"] for q in kept))
                    for p in kept)
            )
            out[amount] = {"log_prob": log_total, "paths": kept}
    if out:
        lse = max(
            e["log_prob"] + len(a) * math.log(DIGIT_PRIOR)
            for a, e in out.items()
        )
        denom = lse + math.log(
            sum(
                math.exp(
                    e["log_prob"] + len(a) * math.log(DIGIT_PRIOR) - lse
                )
                for a, e in out.items()
            )
        )
        for a, e in out.items():
            e["prob"] = math.exp(
                e["log_prob"] + len(a) * math.log(DIGIT_PRIOR) - denom
            )
    return out


def feasible_answers(inferred: dict, tokens: list, seg_id: int) -> list[dict]:
    """给定当前后验，该笔段在某条合法切法下实际可能收到的答复。

    穷举全部金额的切法，收集“笔段单独构成整字 d”与“不是整字”两类，
    避免规划器考虑从不被任何切法支持的答复。
    """
    seg = split_segments(tokens)[seg_id]
    digits, not_whole = set(), False
    for amount, entry in inferred.items():
        for p in entry["paths"]:
            whole, digit_index = _segment_owner_digit(p, seg)
            if whole:
                digits.add(amount[int(digit_index)])
            else:
                not_whole = True
    answers = [{"kind": "digit", "digit": d} for d in sorted(digits)]
    if not_whole:
        answers.append({"kind": "not_whole"})
    return answers


def _answer_ok(path: dict, seg: dict, answer: dict, amount: str) -> bool:
    whole, digit_index = _segment_owner_digit(path, seg)
    if answer["kind"] == "digit":
        return whole and amount[int(digit_index)] == answer["digit"]
    return not whole
