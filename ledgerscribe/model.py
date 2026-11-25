"""数字与字界关系的离线 HMM（递归切分枚举 + 约束 EM）。

隐结构：每个数字有固定笔数（见 synth.TEMPLATES），一笔发出 1..MAX_LEN 个
方向 token；笔与笔、字与字的边界处“抬笔 U”按 Bernoulli 出现，不出现即连写。
观察序列里的每个 U 都必须落在某个笔边界上；方向 token 处也允许无 U 换笔/换字。

训练只拿到金额字符串标签、拿不到字界：E 步枚举该金额下全部合法切分并做
软计数，M 步归一化（带模板先验伪计数）。
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from . import UP
from .synth import TEMPLATES

MAX_LEN = 2
LENGTHS = (1, 2)
EPS = 1e-12

# 假设空间内的数字个数先验：每个数字计一次。只在随页候选之间起作用，
# 避免少字假设靠更少边界罚分碾压多字真值；训练标签给定数字个数，EM 不受影响。
DIGIT_PRIOR = 0.85


def _log(x: float) -> float:
    return math.log(max(x, EPS))


class Model:
    def __init__(self, params: dict):
        self.params = params

    def emit_p(self, digit: str, stroke: int, d: int) -> float:
        return self.params["dir"][digit][stroke][d]

    def len_p(self, digit: str, stroke: int, length: int) -> float:
        return self.params["len"][digit][stroke][length - 1]

    def u_internal_p(self, digit: str, boundary: int) -> float:
        return self.params["u_internal"][digit][boundary]

    u_boundary_p = property(lambda self: self.params["u_boundary"])

    # ---- 切分枚举 ----------------------------------------------------------
    def enumerate_paths(self, tokens: list, digits: str) -> list[dict]:
        """递归枚举 digits 解释 tokens 的全部合法切分（含连写），按 assign 去重。

        状态 (pos, fi, used)：已消费到 pos，当前在第 fi 支平铺笔，该笔已吞
        used 个 token。每支笔吞 1..MAX_LEN 个 token，长度因子在笔收尾时计一次；
        边界观察到 U 用抬笔概率，否则用连写概率 (1-p)。
        单笔字可无 U 收尾跨字界，故同一轨迹兼容笔数不同的金额（12/17、23/28）。
        """
        nstr = [len(TEMPLATES[d]) for d in digits]
        flat = [
            (di, si, digits[di])
            for di in range(len(digits))
            for si in range(nstr[di])
        ]
        total_strokes = len(flat)

        @lru_cache(maxsize=None)
        def go(pos, fi, used):
            if fi == total_strokes:
                return [(0.0, ())] if pos == len(tokens) else []
            if pos == len(tokens):
                # 末笔在此收尾
                if fi == total_strokes - 1 and 1 <= used <= MAX_LEN:
                    di, si, ch = flat[fi]
                    return [(_log(self.len_p(ch, si, used)), ())]
                return []
            tok = tokens[pos]
            di, si, ch = flat[fi]
            out = []
            if tok == UP:
                # U 只能在笔边界：当前笔已吞 used>=1 后收尾，跨入下一笔。
                if not (1 <= used <= MAX_LEN) or fi + 1 >= total_strokes:
                    return []
                nxt_di = flat[fi + 1][0]
                q = (
                    self.u_internal_p(ch, si)
                    if nxt_di == di
                    else self.u_boundary_p
                )
                for lp, rest in go(pos + 1, fi + 1, 0):
                    out.append(
                        (
                            lp
                            + _log(self.len_p(ch, si, used))
                            + _log(q),
                            (None,) + rest,
                        )
                    )
                return out

            # 方向 token：本笔要么继续吞 tok（长度未到上限），
            # 要么在 tok 之前收尾、由下一笔吞 tok（无 U 连写）。
            # (a) tok 继续当前笔
            if used < MAX_LEN:
                for lp, rest in go(pos + 1, fi, used + 1):
                    out.append(
                        (
                            lp + _log(self.emit_p(ch, si, tok)),
                            ((di, si),) + rest,
                        )
                    )
            # (b) 当前笔按现状 used 收尾（tok 归下一笔第一笔），无 U 连写。
            # 这是单笔字在无 U 字界跨入下一字、以及无 U 换笔的唯一转移，
            # 与按笔推进的前向 DP 一一对应，不产生重复 assign。
            if 1 <= used <= MAX_LEN and fi + 1 < total_strokes:
                nxt_di, nxt_si, nxt_ch = flat[fi + 1]
                q = (
                    1.0 - self.u_internal_p(ch, si)
                    if nxt_di == di
                    else 1.0 - self.u_boundary_p
                )
                for lp, rest in go(pos + 1, fi + 1, 1):
                    fac = (
                        _log(self.len_p(ch, si, used))
                        + _log(q)
                        + _log(self.emit_p(nxt_ch, nxt_si, tok))
                    )
                    out.append(
                        (lp + fac, ((nxt_di, nxt_si),) + rest)
                    )
            return out

        raw = go(0, 0, 0)
        go.cache_clear()
        seen = {}
        for lp, assign in raw:
            if assign not in seen:
                seen[assign] = lp
        paths = []
        for assign, lp in seen.items():
            stroke_len = {}
            for a in assign:
                if a is not None:
                    stroke_len[a] = stroke_len.get(a, 0) + 1
            if any(not (1 <= l <= MAX_LEN) for l in stroke_len.values()):
                continue
            spans = {}
            for ti, a in enumerate(assign):
                if a is None:
                    continue
                d = a[0]
                lo, hi = spans.get(d, (ti, ti))
                spans[d] = (min(lo, ti), max(hi, ti))
            paths.append(
                {"logp": lp, "assign": assign, "spans": spans}
            )
        return paths

    # ---- 行推断 ------------------------------------------------------------
    def infer_line(self, tokens: list, candidates: list[str]) -> dict:
        """返回每个候选金额的后验概率（同金额多种切法先相加，再在候选集内归一）。"""
        amounts = {}
        for amount in candidates:
            if len(amount) > 3:
                continue
            paths = self.enumerate_paths(tokens, amount)
            if not paths:
                continue
            log_total = math.log(sum(math.exp(p["logp"]) for p in paths))
            amounts[amount] = {"log_prob": log_total, "paths": paths}
        if amounts:
            lse = max(
                e["log_prob"] + len(a) * math.log(DIGIT_PRIOR)
                for a, e in amounts.items()
            )
            denom = lse + math.log(
                sum(
                    math.exp(
                        e["log_prob"] + len(a) * math.log(DIGIT_PRIOR) - lse
                    )
                    for a, e in amounts.items()
                )
            )
            for a, e in amounts.items():
                e["prob"] = math.exp(
                    e["log_prob"] + len(a) * math.log(DIGIT_PRIOR) - denom
                )
        return amounts

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.params, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Model":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# 训练：只有金额标签的约束 EM
# --------------------------------------------------------------------------
def initial_params() -> dict:
    dir_p, len_p, u_int = {}, {}, {}
    for d, homes in TEMPLATES.items():
        dir_p[d], len_p[d], u_int[d] = [], [], []
        for s, home in enumerate(homes):
            row = [0.05] * 8
            row[home] += 4.0
            z = sum(row)
            dir_p[d].append([v / z for v in row])
            len_p[d].append([1 / len(LENGTHS)] * len(LENGTHS))
            if s < len(homes) - 1:
                u_int[d].append(0.5)
    return {
        "dir": dir_p,
        "len": len_p,
        "u_internal": u_int,
        "u_boundary": 0.8,
    }


def _path_soft_counts(line: dict, paths: list[dict]) -> dict:
    lp_max = max(p["logp"] for p in paths)
    raw_w = [math.exp(p["logp"] - lp_max) for p in paths]
    z = sum(raw_w)
    weights = [w / z for w in raw_w]
    tokens = line["tokens"]
    amount = line["amount"]
    c = {
        "dir": {d: [[0.0] * 8 for _ in TEMPLATES[d]] for d in TEMPLATES},
        "len": {
            d: [[0.0] * len(LENGTHS) for _ in TEMPLATES[d]] for d in TEMPLATES
        },
        "ui_on": {d: [0.0] * max(0, len(TEMPLATES[d]) - 1) for d in TEMPLATES},
        "ui_off": {d: [0.0] * max(0, len(TEMPLATES[d]) - 1) for d in TEMPLATES},
        "ub_on": 0.0,
        "ub_off": 0.0,
    }
    for p, w in zip(paths, weights):
        assign = p["assign"]
        for ti, a in enumerate(assign):
            if a is not None:
                di, si = a
                c["dir"][amount[di]][si][tokens[ti]] += w
        stroke_len = {}
        for ti, a in enumerate(assign):
            if a is not None:
                stroke_len[a] = stroke_len.get(a, 0) + 1
        for (di, si), l in stroke_len.items():
            c["len"][amount[di]][si][l - 1] += w
        for di, ch in enumerate(amount):
            for b in range(len(TEMPLATES[ch]) - 1):
                has_u = _boundary_has_u(assign, tokens, di, b)
                if has_u:
                    c["ui_on"][ch][b] += w
                else:
                    c["ui_off"][ch][b] += w
        for j in range(len(amount) - 1):
            gap = range(p["spans"][j][1] + 1, p["spans"][j + 1][0])
            if any(tokens[t] == UP for t in gap):
                c["ub_on"] += w
            else:
                c["ub_off"] += w
    return c


def _boundary_has_u(assign, tokens, di: int, stroke_b: int) -> bool:
    last_t = first_t = None
    for ti, a in enumerate(assign):
        if a is None:
            continue
        if a == (di, stroke_b):
            last_t = ti
        if a == (di, stroke_b + 1) and first_t is None:
            first_t = ti
    if last_t is None or first_t is None:
        return False
    return any(tokens[t] == UP for t in range(last_t + 1, first_t))


def _merge(global_c: dict, local: dict) -> None:
    for d in TEMPLATES:
        for s in range(len(TEMPLATES[d])):
            for k in range(8):
                global_c["dir"][d][s][k] += local["dir"][d][s][k]
            for li in range(len(LENGTHS)):
                global_c["len"][d][s][li] += local["len"][d][s][li]
        for b in range(len(TEMPLATES[d]) - 1):
            global_c["ui_on"][d][b] += local["ui_on"][d][b]
            global_c["ui_off"][d][b] += local["ui_off"][d][b]
    global_c["ub_on"] += local["ub_on"]
    global_c["ub_off"] += local["ub_off"]


def train(train_lines: list[dict], iterations: int = 25) -> Model:
    params = initial_params()
    for _ in range(iterations):
        c = {
            "dir": {d: [[0.0] * 8 for _ in TEMPLATES[d]] for d in TEMPLATES},
            "len": {
                d: [[0.0] * len(LENGTHS) for _ in TEMPLATES[d]]
                for d in TEMPLATES
            },
            "ui_on": {d: [0.0] * max(0, len(TEMPLATES[d]) - 1) for d in TEMPLATES},
            "ui_off": {d: [0.0] * max(0, len(TEMPLATES[d]) - 1) for d in TEMPLATES},
            "ub_on": 0.0,
            "ub_off": 0.0,
        }
        for line in train_lines:
            model = Model(params)
            paths = model.enumerate_paths(line["tokens"], line["amount"])
            if paths:
                _merge(c, _path_soft_counts(line, paths))

        new = initial_params()
        for d, homes in TEMPLATES.items():
            for s in range(len(homes)):
                row = [c["dir"][d][s][k] + 0.2 for k in range(8)]
                z = sum(row)
                new["dir"][d][s] = [v / z for v in row]
                lrow = [c["len"][d][s][li] + 0.2 for li in range(len(LENGTHS))]
                z = sum(lrow)
                new["len"][d][s] = [v / z for v in lrow]
                if s < len(homes) - 1:
                    on, off = c["ui_on"][d][s], c["ui_off"][d][s]
                    new["u_internal"][d][s] = (on + 1.0) / (on + off + 2.0)
        new["u_boundary"] = (c["ub_on"] + 1.0) / (c["ub_on"] + c["ub_off"] + 2.0)
        params = new
    return Model(params)
