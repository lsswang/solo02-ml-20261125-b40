"""单据与布号下的状态档案。

每张单据描述一条布的裁切任务：每格损失、留样格、初始头位、设备费率。
档案（ledger）按布号记录三类事件，重新开单时据此判断哪一段已不能再剪：
- prepared  准备复扫（下单时锁定设备/策略，尚未拿到读数）
- reading   收回该台实际读数（精检结果）
- cut       已裁断，被剪区段进入 blocked，后续单据必须避开
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import dump_json, load_json
from .policy import best_cut, expected_cut_loss, segment_cells


def new_ledger(path: Optional[str] = None) -> Dict[str, Any]:
    ledger: Dict[str, Any] = {"fabrics": {}}
    if path:
        try:
            ledger = load_json(path)
        except FileNotFoundError:
            pass
    return ledger


def _fabric_record(ledger: Dict[str, Any], fabric_id: str) -> Dict[str, Any]:
    return ledger["fabrics"].setdefault(fabric_id, {
        "events": [], "blocked_cells": [], "retain_cells": [],
        "current_order": None,
    })


def blocked_cells(ledger: Dict[str, Any], fabric_id: str) -> List[int]:
    return list(_fabric_record(ledger, fabric_id)["blocked_cells"])


def prepare_order(ledger: Dict[str, Any], order: Dict[str, Any],
                  plan: Dict[str, Any], ts: Optional[float] = None) -> Dict[str, Any]:
    """下单：记录准备复扫事件，存下选定策略；此时只决定值不值得扫。"""
    rec = _fabric_record(ledger, order["fabric_id"])
    rec["retain_cells"] = list(order.get("retain", []))
    event = {
        "type": "prepared",
        "ts": ts if ts is not None else time.time(),
        "order_id": order["order_id"],
        "device": plan["device"],
        "plan": plan,
        "order": order,
    }
    rec["events"].append(event)
    rec["current_order"] = order["order_id"]
    return event


def record_reading(ledger: Dict[str, Any], fabric_id: str, order_id: str,
                   precision: Dict[int, int], scan_fee: float,
                   ts: Optional[float] = None) -> Dict[str, Any]:
    """收回该台实际读数（格号 -> 0好/1坏），并记录实付扫描费。"""
    rec = _fabric_record(ledger, fabric_id)
    event = {
        "type": "reading",
        "ts": ts if ts is not None else time.time(),
        "order_id": order_id,
        "precision": {str(k): v for k, v in precision.items()},
        "scan_fee": scan_fee,
    }
    rec["events"].append(event)
    return event


def issue_cut(ledger: Dict[str, Any], fabric_id: str, order_id: str,
              segment: Optional[Tuple[int, int]], post: Dict[str, float],
              order: Dict[str, Any], scan_fee: float = 0.0,
              ts: Optional[float] = None) -> Dict[str, Any]:
    """开出具体剪口：剪口在拿到实际读数之后才能确定，并更新 blocked。"""
    rec = _fabric_record(ledger, fabric_id)
    blocked = list(rec["blocked_cells"])
    forbidden = set(blocked) | set(order.get("retain", []))
    cells = segment_cells(segment)
    conflict = [j for j in cells if j in forbidden]
    if conflict:
        raise ValueError(f"剪口 {segment} 触及留样/已裁断格 {conflict}，禁止开单")
    exp = expected_cut_loss(post, segment, order["losses"])
    event = {
        "type": "cut",
        "ts": ts if ts is not None else time.time(),
        "order_id": order_id,
        "segment": None if segment is None else list(segment),
        "cells": cells,
        "expected_cut_loss_at_cut": exp,
        "scan_fee": scan_fee,
        "expected_total_cost": scan_fee + exp,
    }
    rec["events"].append(event)
    rec["blocked_cells"] = sorted(set(blocked) | set(cells))
    rec["current_order"] = None
    return event


def settle(ledger: Dict[str, Any], fabric_id: str, truth: str,
           order: Dict[str, Any], ts: Optional[float] = None) -> Dict[str, Any]:
    """结清：用精检真图对最近一次剪口核算实际损失。"""
    from .policy import realized_cut_loss
    rec = _fabric_record(ledger, fabric_id)
    last_cut = next(e for e in reversed(rec["events"]) if e["type"] == "cut")
    seg_raw = last_cut["segment"]
    seg = None if seg_raw is None else tuple(seg_raw)
    actual = realized_cut_loss(truth, seg, order["losses"])
    event = {
        "type": "settled",
        "ts": ts if ts is not None else time.time(),
        "order_id": last_cut["order_id"],
        "truth": truth,
        "actual_cut_loss": actual,
    }
    rec["events"].append(event)
    return event


def save_ledger(ledger: Dict[str, Any], path: str) -> None:
    dump_json(ledger, path)


def history(ledger: Dict[str, Any], fabric_id: str) -> List[Dict[str, Any]]:
    return list(_fabric_record(ledger, fabric_id)["events"])
