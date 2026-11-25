"""按选定设备的可行操作最小化期望损失。

共同裁切规则：
- 每条布最多剪去一个连续区段（区段可为空，即不剪）；
- 留样格（retain）与已裁断（blocked）格不可再剪，区段必须整体避开它们；
- 每格损失 loss_j：剪坏格或留好格（漏剪坏格）各损失 loss_j，正确处理为 0；
  因此给定后验 P(格 j 坏)，剪去 j 的期望损失 = loss_j*(1-P)，保留 j = loss_j*P。

便携头（默认路线）：
- 可扫两个单格；第二个扫哪个可依据第一次结果决定；
- 第一次按启动费 startup + 距初始头位 head0 的移动费 move*|c-head0| 付费；
- 第二次只付移动费 move*|c2-c1|（启动只计一次）。

桥式头（对照路线）：
- 开机前锁定至多两格的连续窗口，只走一趟，窗口内均看清；
- 只收一次开机费 bridge_fee，无逐格移动费。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import condition, marginals

EPS = 1e-12


def feasible_segments(n: int,
                      retain: Sequence[int],
                      blocked: Sequence[int]) -> List[Optional[Tuple[int, int]]]:
    """返回全部可行连续区段，None 表示不剪。区段端点为闭区间。"""
    forbidden = set(retain) | set(blocked)
    segs: List[Optional[Tuple[int, int]]] = [None]
    for a in range(n):
        for b in range(a, n):
            if any(j in forbidden for j in range(a, b + 1)):
                continue
            segs.append((a, b))
    return segs


def segment_cells(seg: Optional[Tuple[int, int]]) -> List[int]:
    if seg is None:
        return []
    return list(range(seg[0], seg[1] + 1))


def expected_cut_loss(post: Dict[str, float],
                      seg: Optional[Tuple[int, int]],
                      losses: Sequence[float]) -> float:
    """给定后验，剪某区段的期望损失（未覆盖格全部保留）。"""
    marg = marginals(post)
    cut = set(segment_cells(seg))
    total = 0.0
    for j, p_bad in enumerate(marg):
        if j in cut:
            total += losses[j] * (1.0 - p_bad)  # 误剪好格
        else:
            total += losses[j] * p_bad          # 漏剪坏格
    return total


def best_cut(post: Dict[str, float],
             losses: Sequence[float],
             retain: Sequence[int],
             blocked: Sequence[int]) -> Dict[str, Any]:
    """在可行区段里取期望损失最小者；并列时选更短、更靠左的区段。"""
    n = len(losses)
    best: Optional[Dict[str, Any]] = None
    for seg in feasible_segments(n, retain, blocked):
        val = expected_cut_loss(post, seg, losses)
        key = (val, len(segment_cells(seg)),
               n + 1 if seg is None else seg[0],
               n + 1 if seg is None else seg[1])
        if best is None or key < best["_key"]:
            best = {"segment": seg, "expected_loss": val, "_key": key}
    assert best is not None
    best.pop("_key")
    return best


def value_with_precision(post: Dict[str, float],
                         observed: Dict[int, int],
                         losses: Sequence[float],
                         retain: Sequence[int],
                         blocked: Sequence[int]) -> float:
    """已知若干格精检真值后的最优裁切期望损失。"""
    p = post
    for cell, bit in observed.items():
        p = condition(p, cell, bit)
    return best_cut(p, losses, retain, blocked)["expected_loss"]


def expected_value(post: Dict[str, float],
                   observed: Dict[int, int],
                   next_cell: int,
                   losses: Sequence[float],
                   retain: Sequence[int],
                   blocked: Sequence[int]) -> float:
    """E[最优裁切损失 | 已有精检，再扫 next_cell]，对读数好坏求期望。"""
    base = post
    for cell, bit in observed.items():
        base = condition(base, cell, bit)
    marg = marginals(base)
    total = 0.0
    for bit in (0, 1):
        prob = marg[next_cell] if bit == 1 else 1.0 - marg[next_cell]
        total += prob * value_with_precision(
            post, {**observed, next_cell: bit}, losses, retain, blocked)
    return total


# ----------------------------- 便携头 -----------------------------

def _scan_cost_portable(c1: int, c2: Optional[int], head0: int,
                        startup: float, move: float) -> float:
    fee = startup + move * abs(c1 - head0)
    if c2 is not None:
        fee += move * abs(c2 - c1)
    return fee


def portable_plan(post: Dict[str, float],
                  losses: Sequence[float],
                  retain: Sequence[int],
                  blocked: Sequence[int],
                  head0: int,
                  startup: float,
                  move: float) -> Dict[str, Any]:
    """枚举便携头全部可行策略，返回期望总成本最小的一份。

    策略三类：不扫；只扫一格；扫第一格后按结果自适应决定第二格。
    并列时按 不扫 < 只扫 < 两格扫 的次序选，保证可复跑且不无谓付费。
    """
    n = len(losses)
    scan_allowed = [j for j in range(n) if j not in set(retain) | set(blocked)]

    no_scan = best_cut(post, losses, retain, blocked)
    plans: List[Dict[str, Any]] = [{
        "device": "portable", "kind": "no_scan",
        "expected_cost": no_scan["expected_loss"],
        "cut_if_no_scan": no_scan["segment"],
    }]

    # 只扫一格
    for c1 in scan_allowed:
        fee = _scan_cost_portable(c1, None, head0, startup, move)
        val = expected_value(post, {}, c1, losses, retain, blocked)
        plans.append({
            "device": "portable", "kind": "one_scan", "cell1": c1,
            "scan_fee": fee, "expected_cut_loss": val,
            "expected_cost": fee + val,
        })

    # 两格扫：第二格按第一次结果选择（含“结果出来后不再扫第二格”的分支）
    for c1 in scan_allowed:
        for c2 in scan_allowed:
            if c2 == c1:
                continue
            first_fee = startup + move * abs(c1 - head0)
            marg = marginals(post)
            exp_val = 0.0
            branches: Dict[str, Any] = {}
            for bit in (0, 1):
                prob = marg[c1] if bit == 1 else 1.0 - marg[c1]
                stop_val = value_with_precision(
                    post, {c1: bit}, losses, retain, blocked)
                go_val = expected_value(
                    post, {c1: bit}, c2, losses, retain, blocked)
                extra_fee = move * abs(c2 - c1)
                if go_val + extra_fee < stop_val - EPS:
                    branch_val = go_val + extra_fee
                    decision = "scan_second"
                else:
                    branch_val = stop_val
                    decision = "stop"
                branches[f"first_{bit}"] = {
                    "probability": prob,
                    "decision": decision,
                    "expected_cut_loss_after_branch": branch_val,
                }
                exp_val += prob * branch_val
            plans.append({
                "device": "portable", "kind": "two_scan",
                "cell1": c1, "cell2": c2,
                "scan_fee_total_if_both": _scan_cost_portable(
                    c1, c2, head0, startup, move),
                "expected_cost": first_fee + exp_val,
                "branches": branches,
            })

    rank = {"no_scan": 0, "one_scan": 1, "two_scan": 2}
    plans.sort(key=lambda p: (p["expected_cost"], rank[p["kind"]],
                              p.get("cell1", -1), p.get("cell2", -1)))
    return plans[0]


# ----------------------------- 桥式头 -----------------------------

def bridge_windows(n: int, blocked: Sequence[int]) -> List[Tuple[int, int]]:
    """至多两格的连续窗口（窗口不得覆盖已裁断格；留样格允许精检，本就不剪）。"""
    blocked_set = set(blocked)
    windows: List[Tuple[int, int]] = []
    for a in range(n):
        for b in (a, min(a + 1, n - 1)):
            if any(j in blocked_set for j in range(a, b + 1)):
                continue
            windows.append((a, b))
    return windows


def bridge_plan(post: Dict[str, float],
                losses: Sequence[float],
                retain: Sequence[int],
                blocked: Sequence[int],
                bridge_fee: float) -> Dict[str, Any]:
    """枚举桥式头全部窗口，开机费 + 窗口读数后期望裁切损失，取最小。"""
    no_scan = best_cut(post, losses, retain, blocked)
    plans: List[Dict[str, Any]] = [{
        "device": "bridge", "kind": "no_scan",
        "expected_cost": no_scan["expected_loss"],
        "cut_if_no_scan": no_scan["segment"],
    }]
    for win in bridge_windows(len(losses), blocked):
        val = 0.0
        marg = marginals(post)
        # 联合枚举窗口内各格真值（至多 2^2=4 种）
        cells = list(range(win[0], win[1] + 1))
        for bits in range(1 << len(cells)):
            assignment = {c: (bits >> (len(cells) - 1 - k)) & 1
                          for k, c in enumerate(cells)}
            cond = post
            prob = 1.0
            # 用逐格条件化计算该联合读数的概率
            for c in cells:
                p_c = marginals(cond)[c]
                prob *= p_c if assignment[c] == 1 else 1.0 - p_c
                cond = condition(cond, c, assignment[c])
            val += prob * value_with_precision(
                post, assignment, losses, retain, blocked)
        plans.append({
            "device": "bridge", "kind": "one_pass",
            "window": list(win),
            "bridge_fee": bridge_fee,
            "expected_cut_loss": val,
            "expected_cost": bridge_fee + val,
        })
    rank = {"no_scan": 0, "one_pass": 1}
    plans.sort(key=lambda p: (p["expected_cost"], rank[p["kind"]],
                              p.get("window", [-1])[0]))
    return plans[0]


def make_plan(device: str, post: Dict[str, float], order: Dict[str, Any],
              blocked: Sequence[int]) -> Dict[str, Any]:
    losses = order["losses"]
    retain = order.get("retain", [])
    if device == "portable":
        return portable_plan(post, losses, retain, blocked,
                             order["head0"], order["startup_fee"],
                             order["move_fee"])
    if device == "bridge":
        return bridge_plan(post, losses, retain, blocked,
                           order["bridge_fee"])
    raise ValueError(f"未知设备 {device}")


def realized_cut_loss(truth: str, seg: Optional[Tuple[int, int]],
                      losses: Sequence[float]) -> float:
    """单据结清后用精检真图核算实际剪口损失（演示与测试用）。"""
    cut = set(segment_cells(seg))
    total = 0.0
    for j, c in enumerate(truth):
        bad = c == "1"
        if j in cut and not bad:
            total += losses[j]  # 误剪好格
        elif j not in cut and bad:
            total += losses[j]  # 漏剪坏格
    return total
