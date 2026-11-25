"""端到端工作流：把模型、单据、读数与档案串起来。

正常下单入口 place_order 的时序严格对应题目要求：
1. 用低清读数算整幅图后验，按选定设备枚举可行操作，下单时只决定值不值得扫；
2. 收回该台实际读数（便携头可能自适应扫到第二格）后，才开出具体剪口；
3. 准备复扫、收读数、已裁断三类事件全部记入布号档案。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .common import condition, marginals, posterior
from .order import (blocked_cells, issue_cut, prepare_order, record_reading,
                    save_ledger)
from .policy import best_cut, make_plan, segment_cells


def posterior_from_readings(model: Dict[str, Any],
                            order: Dict[str, Any]) -> Dict[str, float]:
    readings = order.get("readings", [])
    return posterior(model["prior"], model.get("emission"), readings)


def _portable_scan(plan: Dict[str, Any],
                   precision_available: Dict[int, int],
                   order: Dict[str, Any]) -> Tuple[Dict[int, int], float, List[Dict[str, Any]]]:
    """按便携头策略实际复扫，第二格依第一次结果决定。"""
    head0, startup, move = order["head0"], order["startup_fee"], order["move_fee"]
    steps: List[Dict[str, Any]] = []
    if plan["kind"] == "no_scan":
        return {}, 0.0, steps

    c1 = plan["cell1"]
    if c1 not in precision_available:
        raise ValueError(f"便携头要扫第 {c1} 格，但该台没有返回此格读数")
    bit1 = precision_available[c1]
    fee1 = startup + move * abs(c1 - head0)
    steps.append({"cell": c1, "bit": bit1, "fee": fee1, "leg": "first"})
    scanned = {c1: bit1}

    if plan["kind"] == "two_scan":
        branch = plan["branches"][f"first_{bit1}"]
        if branch["decision"] == "scan_second":
            c2 = plan["cell2"]
            if c2 not in precision_available:
                raise ValueError(f"便携头自适应要扫第 {c2} 格，但该台没有返回此格读数")
            bit2 = precision_available[c2]
            fee2 = move * abs(c2 - c1)
            steps.append({"cell": c2, "bit": bit2, "fee": fee2, "leg": "second"})
            scanned[c2] = bit2
    return scanned, sum(s["fee"] for s in steps), steps


def _bridge_scan(plan: Dict[str, Any],
                 precision_available: Dict[int, int],
                 order: Dict[str, Any]) -> Tuple[Dict[int, int], float, List[Dict[str, Any]]]:
    if plan["kind"] == "no_scan":
        return {}, 0.0, []
    cells = list(plan["window"])
    for c in cells:
        if c not in precision_available:
            raise ValueError(f"桥式头窗口 {cells} 含未返回读数的第 {c} 格")
    scanned = {c: precision_available[c] for c in cells}
    steps = [{"cell": c, "bit": scanned[c], "fee": 0.0, "leg": "one_pass"}
             for c in cells]
    return scanned, order["bridge_fee"], steps


def place_order(model: Dict[str, Any], order: Dict[str, Any],
                precision_available: Dict[int, int],
                ledger: Dict[str, Any]) -> Dict[str, Any]:
    """正常下单入口。precision_available 是该台能返回的单格精检读数。"""
    fabric_id = order["fabric_id"]
    blocked = blocked_cells(ledger, fabric_id)
    post0 = posterior_from_readings(model, order)
    plan = make_plan(order["device"], post0, order, blocked)
    prepare_order(ledger, order, plan)

    if order["device"] == "portable":
        scanned, fee, steps = _portable_scan(plan, precision_available, order)
    else:
        scanned, fee, steps = _bridge_scan(plan, precision_available, order)

    if scanned:
        record_reading(ledger, fabric_id, order["order_id"], scanned, fee)

    post = post0
    for cell, bit in scanned.items():
        post = condition(post, cell, bit)
    cut = best_cut(post, order["losses"], order.get("retain", []), blocked)
    issue_cut(ledger, fabric_id, order["order_id"],
              cut["segment"], post, order, scan_fee=fee)

    return {
        "order_id": order["order_id"],
        "fabric_id": fabric_id,
        "device": order["device"],
        "plan_at_order_time": plan,
        "scan_steps": steps,
        "scan_fee": fee,
        "scanned_cells": scanned,
        "posterior_marginals_before": marginals(post0),
        "posterior_marginals_after": marginals(post),
        "cut_segment": cut["segment"],
        "cut_cells": segment_cells(cut["segment"]),
        "expected_cut_loss": cut["expected_loss"],
        "expected_total_cost": fee + cut["expected_loss"],
        "blocked_cells_after": blocked_cells(ledger, fabric_id),
    }
