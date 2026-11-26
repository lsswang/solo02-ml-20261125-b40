from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .model import BlendModel
from .utils import (
    append_jsonl,
    composition_to_string,
    ensure_dir,
    parse_composition,
    read_json,
    read_jsonl,
    total_parts,
    write_json,
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "parts_per_cup": 4,
    "gate_threshold": 0.80,
    "trial_accuracy": 0.90,
    "good_cup_reward": 2.0,
    "bad_cup_penalty": 8.0,
    "grams_per_part": 15.0,
    "default_bucket_capacity_cups": 2,
    "currency_unit": "synthetic score",
}


def state_key(roast_id: str, formula_id: str) -> str:
    return f"{roast_id}||{formula_id}"


def expected_score(p: float, cups: int, cfg: Dict[str, Any]) -> float:
    return cups * (p * float(cfg["good_cup_reward"]) - (1.0 - p) * float(cfg["bad_cup_penalty"]))


def bayes_update(prior: float, result: str, accuracy: float) -> Dict[str, float]:
    a = float(accuracy)
    if not 0.5 < a < 1.0:
        raise ValueError("trial accuracy should be in (0.5, 1) for a useful symmetric signal")
    r = result.strip().lower()
    if r in {"like", "similar", "像", "是", "positive", "pass"}:
        likelihood_true = a
        likelihood_false = 1.0 - a
        canonical = "like"
    elif r in {"unlike", "different", "不像", "否", "negative", "fail"}:
        likelihood_true = 1.0 - a
        likelihood_false = a
        canonical = "unlike"
    else:
        raise ValueError("trial result must be like/similar/像 or unlike/different/不像")
    num = prior * likelihood_true
    den = prior * likelihood_true + (1.0 - prior) * likelihood_false
    posterior = num / den
    # The sample uses exact 0.5/0.9/0.1 arithmetic; rounding only cleans JSON.
    posterior = round(posterior, 12)
    return {
        "reading": canonical,
        "prior": prior,
        "likelihood_if_qualified": likelihood_true,
        "likelihood_if_unqualified": likelihood_false,
        "posterior": posterior,
    }


class ShopError(RuntimeError):
    pass


class ShopService:
    """Append-only local shop ledger.

    The JSONL ledger is authoritative.  ``state.json`` is only a replay cache.
    There are deliberately no edit/delete methods for accepted orders,
    successful weighings, or consumed beans.
    """

    def __init__(self, state_dir: str, model: Optional[BlendModel] = None) -> None:
        self.state_dir = state_dir
        ensure_dir(state_dir)
        self.ledger_path = os.path.join(state_dir, "ledger.jsonl")
        self.snapshot_path = os.path.join(state_dir, "state.json")
        self.model = model
        self.state: Dict[str, Any] = self._empty_state()
        self._replay()

    def _empty_state(self) -> Dict[str, Any]:
        return {
            "initialized": False,
            "event_count": 0,
            "config": dict(DEFAULT_CONFIG),
            "batch_meta": {},
            "loose_stock_parts": {},
            "buckets": {},
            "orders": {},
            "trial_states": {},
            "shifts": {},
            "roast_state_overrides": {},
            "weighings": [],
            "events": [],
        }

    def _replay(self) -> None:
        events = read_jsonl(self.ledger_path)
        if not events and os.path.exists(self.snapshot_path):
            # Snapshot is a convenience; normally ledger exists after init.
            self.state = read_json(self.snapshot_path)
            return
        for event in events:
            self._apply(event, store_event=False)
        self.state["event_count"] = len(events)
        if events:
            self.state["initialized"] = True

    def _next_event_id(self) -> str:
        self.state["event_count"] += 1
        return f"E{self.state['event_count']:04d}"

    def _append(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.state.get("initialized") and event_type != "shop_initialized":
            raise ShopError("shop is not initialized")
        event_id = self._next_event_id()
        event = {"event_id": event_id, "type": event_type, **payload}
        append_jsonl(self.ledger_path, event)
        self._apply(event, store_event=True)
        write_json(self.snapshot_path, self.state)
        return event

    def _apply(self, event: Dict[str, Any], *, store_event: bool) -> None:
        et = event["type"]
        if et == "shop_initialized":
            self.state["initialized"] = True
            self.state["config"] = {**DEFAULT_CONFIG, **event.get("config", {})}
            self.state["batch_meta"] = event.get("batch_meta", {})
            self.state["loose_stock_parts"] = dict(event.get("loose_stock_parts", {}))
            self.state["roast_state_overrides"] = dict(event.get("roast_state_overrides", {}))
            for b in event.get("empty_buckets", []):
                bid = b["bucket_id"]
                self.state["buckets"][bid] = {
                    "bucket_id": bid,
                    "status": "empty",
                    "capacity_cups": int(b.get("capacity_cups", self.state["config"]["default_bucket_capacity_cups"])),
                    "cups_total": 0,
                    "remaining_cups": 0,
                    "reserved_cups": 0,
                    "roast_id": None,
                    "formula_id": None,
                    "composition": None,
                    "dose_parts_per_cup": None,
                    "weighing_id": None,
                }
            for b in event.get("weighed_buckets", []):
                bid = b["bucket_id"]
                comp = parse_composition(b["composition"])
                cap = int(b["capacity_cups"])
                remaining = int(b.get("remaining_cups", cap))
                self.state["buckets"][bid] = {
                    "bucket_id": bid,
                    "status": "active" if remaining > 0 else "empty",
                    "capacity_cups": cap,
                    "cups_total": int(b.get("cups_total", cap)),
                    "remaining_cups": remaining,
                    "reserved_cups": int(b.get("reserved_cups", 0)),
                    "roast_id": b["roast_id"],
                    "formula_id": b["formula_id"],
                    "composition": comp,
                    "dose_parts_per_cup": {k: int(v) for k, v in comp.items()},
                    "weighing_id": b.get("weighing_id", "preweighed-input"),
                }
        elif et == "stock_withdrawn":
            b = event["batch_id"]
            self.state["loose_stock_parts"][b] = int(self.state["loose_stock_parts"].get(b, 0)) - int(event["parts"])
        elif et == "bucket_opened":
            self.state["loose_stock_parts"] = dict(event["stock_after"])
            self.state["buckets"][event["bucket_id"]].update(
                {
                    "status": "active",
                    "cups_total": int(event["cups"]),
                    "remaining_cups": int(event["cups"]),
                    "reserved_cups": 0,
                    "roast_id": event["roast_id"],
                    "formula_id": event["formula_id"],
                    "composition": dict(event["composition"]),
                    "dose_parts_per_cup": dict(event["composition"]),
                    "weighing_id": event["weighing_id"],
                }
            )
            self.state["weighings"].append(event["weighing"])
        elif et == "order_received":
            self.state["orders"][event["order_id"]] = {
                "order_id": event["order_id"],
                "cups": int(event["cups"]),
                "roast_id": event["roast_id"],
                "formula_id": event["formula_id"],
                "composition": dict(event["composition"]),
                "mode": event["mode"],
                "shift_id": event.get("shift_id"),
                "status": "received",
                "probability_before": event.get("probability_before"),
                "decision": None,
                "source": None,
                "events": [],
            }
        elif et == "trial_prepared":
            o = self.state["orders"][event["order_id"]]
            o["status"] = "awaiting_reading"
            o["trial_weighing_id"] = event["weighing_id"]
            o["events"].append(event["event_id"])
            self.state["weighings"].append(event["weighing"])
            self.state["loose_stock_parts"] = dict(event["stock_after"])
            shift = event["shift_id"]
            self.state["shifts"][shift] = {
                "trial_used": True,
                "order_id": event["order_id"],
                "weighing_id": event["weighing_id"],
            }
        elif et == "trial_recorded":
            key = state_key(event["roast_id"], event["formula_id"])
            ts = self.state["trial_states"].setdefault(
                key, {"readings": [], "probability": event["update"]["posterior"]}
            )
            ts["probability"] = event["update"]["posterior"]
            ts["readings"].append(event["update"])
            o = self.state["orders"][event["order_id"]]
            o["probability_after_trial"] = event["update"]["posterior"]
            o["events"].append(event["event_id"])
        elif et == "order_accepted":
            o = self.state["orders"][event["order_id"]]
            o["status"] = "accepted"
            o["decision"] = "accept"
            o["source"] = event["source"]
            o["probability_at_decision"] = event["probability"]
            o["expected_score"] = event.get("expected_score")
            o["reason"] = event.get("reason")
            o["events"].append(event["event_id"])
            if event["source"]["kind"] == "bucket":
                bid = event["source"]["bucket_id"]
                self.state["buckets"][bid]["reserved_cups"] += int(o["cups"])
        elif et == "order_rejected":
            o = self.state["orders"][event["order_id"]]
            o["status"] = "rejected"
            o["decision"] = "reject"
            o["probability_at_decision"] = event.get("probability")
            o["reason"] = event["reason"]
            o["events"].append(event["event_id"])
        elif et == "direct_weigh_succeeded":
            self.state["loose_stock_parts"] = dict(event["stock_after"])
            self.state["weighings"].append(event["weighing"])
            o = self.state["orders"][event["order_id"]]
            o.setdefault("direct_weighing_ids", []).append(event["weighing_id"])
            o["events"].append(event["event_id"])
        elif et == "cup_dispensed":
            o = self.state["orders"][event["order_id"]]
            o["status"] = "served"
            o["served_cups"] = int(event.get("served_cups", o["cups"]))
            o["events"].append(event["event_id"])
            if event.get("bucket_id"):
                b = self.state["buckets"][event["bucket_id"]]
                n = int(event.get("served_cups", o["cups"]))
                b["remaining_cups"] -= n
                b["reserved_cups"] = max(0, b["reserved_cups"] - n)
                if b["remaining_cups"] <= 0:
                    b["status"] = "empty"
        elif et == "stocktake_unfulfillable":
            for oid in event["order_ids"]:
                o = self.state["orders"].get(oid)
                if o and o["status"] in {"accepted", "received", "awaiting_reading"}:
                    o["status"] = "unfulfillable_after_stocktake"
                    o["stocktake_reason"] = event["reason"]
        if store_event:
            self.state["events"].append(event)

    @classmethod
    def initialize(
        cls,
        state_dir: str,
        inventory: Dict[str, Any] | str,
        *,
        model: Optional[BlendModel] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        reset: bool = False,
    ) -> "ShopService":
        if reset and os.path.exists(state_dir):
            shutil.rmtree(state_dir)
        shop = cls(state_dir, model=model)
        if shop.state["initialized"]:
            if model is not None:
                shop.model = model
            return shop
        if isinstance(inventory, str):
            inv = read_json(inventory)
        else:
            inv = inventory
        cfg = {**DEFAULT_CONFIG, **(inv.get("config") or {}), **(config_overrides or {})}
        batch_meta = {b["batch_id"]: b for b in inv.get("batches", [])}
        overrides = {}
        for rs in inv.get("roast_states", []):
            fid = rs.get("formula_id") or composition_to_string(parse_composition(rs["composition"]))
            overrides[state_key(rs["roast_id"], fid)] = rs
        shop._append(
            "shop_initialized",
            {
                "config": cfg,
                "batch_meta": batch_meta,
                "loose_stock_parts": inv.get("loose_stock_parts", {}),
                "empty_buckets": inv.get("empty_buckets", []),
                "weighed_buckets": inv.get("weighed_buckets", []),
                "roast_state_overrides": overrides,
            },
        )
        shop.model = model
        return shop

    # ---------- probabilities and materials ----------

    def probability_for(self, roast_id: str, formula_id: str, comp: Dict[str, int]) -> Tuple[float, str]:
        key = state_key(roast_id, formula_id)
        if key in self.state["trial_states"]:
            return float(self.state["trial_states"][key]["probability"]), "trial_posterior"
        if key in self.state["roast_state_overrides"]:
            rs = self.state["roast_state_overrides"][key]
            return float(rs["probability"]), rs.get("source", "calibrated_roast_state")
        if self.model is not None:
            try:
                return float(self.model.predict_proba(comp)), "mixture_model"
            except KeyError as exc:
                raise ShopError(
                    "inventory contains a batch absent from the fitted model; "
                    "rerun the pipeline on matching sensory records. " + str(exc)
                ) from exc
        raise ShopError("no probability source: load a model or supply roast_states")

    def direct_committed_parts(self) -> Dict[str, int]:
        """Parts promised by accepted direct-scale orders not yet served."""
        committed: Dict[str, int] = {}
        for o in self.state["orders"].values():
            if o.get("status") != "accepted":
                continue
            if (o.get("source") or {}).get("kind") != "direct_scale":
                continue
            for b, parts in o["composition"].items():
                committed[b] = committed.get(b, 0) + int(parts) * int(o["cups"])
        return committed

    def available_loose_parts(self) -> Dict[str, int]:
        """Raw bin stock minus unserved direct-order commitments."""
        available = {b: int(v) for b, v in self.state["loose_stock_parts"].items()}
        for b, parts in self.direct_committed_parts().items():
            available[b] = available.get(b, 0) - parts
        return available

    @staticmethod
    def max_cups_from_part_map(part_map: Dict[str, int], comp: Dict[str, int]) -> int:
        out: Optional[int] = None
        for b, parts in comp.items():
            have = int(part_map.get(b, 0))
            if have < 0:
                have = 0
            cups = have // int(parts)
            out = cups if out is None else min(out, cups)
        return int(out or 0)

    def max_loose_cups(self, comp: Dict[str, int]) -> int:
        # New promises must not consume parts already promised to direct orders.
        return self.max_cups_from_part_map(self.available_loose_parts(), comp)

    def max_raw_loose_cups(self, comp: Dict[str, int]) -> int:
        return self.max_cups_from_part_map(self.state["loose_stock_parts"], comp)

    def matching_buckets(self, roast_id: str, formula_id: str, comp: Dict[str, int]) -> List[Dict[str, Any]]:
        ckey = tuple(sorted(comp.items()))
        out = []
        for b in self.state["buckets"].values():
            if b["status"] != "active" or b["remaining_cups"] <= 0:
                continue
            if b["roast_id"] != roast_id or b["formula_id"] != formula_id:
                # Exact composition is still accepted if formula ids differ by
                # naming convention, but roast must match shared state.
                if tuple(sorted((b.get("composition") or {}).items())) != ckey:
                    continue
            if tuple(sorted((b.get("composition") or {}).items())) != ckey:
                continue
            out.append(b)
        return out

    def material_summary(self, roast_id: str, formula_id: str, comp: Dict[str, int]) -> Dict[str, Any]:
        matching = self.matching_buckets(roast_id, formula_id, comp)
        bucket_remaining = sum(int(b["remaining_cups"]) for b in matching)
        bucket_available = sum(max(0, int(b["remaining_cups"]) - int(b["reserved_cups"])) for b in matching)
        loose_max = self.max_loose_cups(comp)
        raw_loose_max = self.max_raw_loose_cups(comp)
        return {
            "matching_bucket_ids": [b["bucket_id"] for b in matching],
            "matching_bucket_remaining_cups": bucket_remaining,
            "matching_bucket_unreserved_cups": bucket_available,
            "max_cups_from_loose_parts": loose_max,
            "max_cups_from_raw_loose_parts": raw_loose_max,
            "direct_committed_parts": self.direct_committed_parts(),
            "available_loose_parts": self.available_loose_parts(),
            "max_additional_cups_from_loose_parts": loose_max,
        }

    def _weigh_lines(self, comp: Dict[str, int], cups: int) -> List[Dict[str, Any]]:
        grams_per_part = float(self.state["config"]["grams_per_part"])
        lines = []
        for batch, parts in sorted(comp.items()):
            total_parts = int(parts) * int(cups)
            meta = self.state["batch_meta"].get(batch, {})
            lines.append(
                {
                    "batch_id": batch,
                    "bean_name": meta.get("bean_name", meta.get("display_name", batch)),
                    "parts_per_cup": int(parts),
                    "cups": int(cups),
                    "weighed_parts": total_parts,
                    "grams_per_part": grams_per_part,
                    "weighed_grams": total_parts * grams_per_part,
                }
            )
        return lines

    def _make_weighing(
        self, kind: str, comp: Dict[str, int], cups: int, *, bucket_id: Optional[str] = None
    ) -> Dict[str, Any]:
        seq = len(self.state["weighings"]) + 1
        wid = f"W-{seq:03d}"
        lines = self._weigh_lines(comp, cups)
        return {
            "weighing_id": wid,
            "status": "success",
            "kind": kind,
            "bucket_id": bucket_id,
            "cups": int(cups),
            "scale_lines": lines,
            "total_parts": sum(x["weighed_parts"] for x in lines),
            "total_grams": sum(x["weighed_grams"] for x in lines),
            "verification": "simulated scale returned stable positive dose for every line",
        }

    def _stock_after_deduct(self, comp: Dict[str, int], cups: int) -> Dict[str, int]:
        after = dict(self.state["loose_stock_parts"])
        for b, parts in comp.items():
            need = int(parts) * int(cups)
            if int(after.get(b, 0)) < need:
                raise ShopError(f"insufficient {b}: need {need}, have {after.get(b, 0)}")
            after[b] = int(after[b]) - need
        return after

    # ---------- commands ----------

    def withdraw_stock(self, batch_id: str, parts: int, reason: str) -> Dict[str, Any]:
        parts = int(parts)
        if parts <= 0:
            raise ShopError("withdrawal parts must be positive")
        have = int(self.state["loose_stock_parts"].get(batch_id, 0))
        if have < parts:
            raise ShopError(f"cannot withdraw {parts} {batch_id}; only {have}")
        event = self._append(
            "stock_withdrawn",
            {"batch_id": batch_id, "parts": parts, "reason": reason},
        )
        # A stocktake/withdrawal can expose that accepted but unserved orders
        # can no longer be covered.  We never rewrite their recipes/history.
        affected = []
        for oid, o in self.state["orders"].items():
            if o["status"] != "accepted":
                continue
            src = o.get("source") or {}
            if src.get("kind") == "bucket":
                b = self.state["buckets"][src["bucket_id"]]
                if b["remaining_cups"] < o["cups"]:
                    affected.append(oid)
            else:
                try:
                    for bid, p in o["composition"].items():
                        if int(self.state["loose_stock_parts"].get(bid, 0)) < int(p) * o["cups"]:
                            raise ShopError
                except ShopError:
                    affected.append(oid)
        if affected:
            self._append(
                "stocktake_unfulfillable",
                {"order_ids": affected, "reason": f"withdrawal {parts} {batch_id}: {reason}"},
            )
        return event

    def receive_order(
        self,
        order_id: str,
        *,
        cups: int = 1,
        composition: str | Dict[str, Any],
        roast_id: str,
        formula_id: Optional[str] = None,
        mode: str = "bucket",
        shift_id: Optional[str] = None,
        require_trial: bool = False,
    ) -> Dict[str, Any]:
        if order_id in self.state["orders"]:
            raise ShopError(f"order {order_id} already exists; orders are immutable")
        comp = parse_composition(composition)
        expected_parts = int(self.state["config"]["parts_per_cup"])
        if total_parts(comp) != expected_parts:
            raise ShopError(f"quarter recipe must total {expected_parts} parts, got {total_parts(comp)}")
        fid = formula_id or composition_to_string(comp)
        shift = shift_id or self.state["config"].get("default_shift_id", "SHIFT-1")
        p, psource = self.probability_for(roast_id, fid, comp)
        self._append(
            "order_received",
            {
                "order_id": order_id,
                "cups": int(cups),
                "roast_id": roast_id,
                "formula_id": fid,
                "composition": comp,
                "mode": mode,
                "shift_id": shift,
                "probability_before": p,
                "probability_source": psource,
            },
        )

        cfg = self.state["config"]
        threshold = float(cfg["gate_threshold"])
        mat = self.material_summary(roast_id, fid, comp)
        result: Dict[str, Any] = {"order_id": order_id, "probability": p, "probability_source": psource, "material": mat}

        if mode == "trial":
            return self._receive_trial_order(order_id, cups, comp, roast_id, fid, shift, p, threshold, require_trial, result)
        if mode != "bucket":
            raise ShopError(f"unknown mode {mode!r}; use 'bucket' or 'trial'")
        return self._receive_bucket_order(order_id, cups, comp, roast_id, fid, p, threshold, result)

    def _receive_bucket_order(
        self, oid: str, cups: int, comp: Dict[str, int], roast: str, fid: str,
        p: float, threshold: float, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        score_order = expected_score(p, cups, self.state["config"])
        score_per_cup = expected_score(p, 1, self.state["config"])
        mat = self.material_summary(roast, fid, comp)
        max_fulfillable = mat["matching_bucket_unreserved_cups"] + mat["max_cups_from_loose_parts"]
        result.update(
            {
                "expected_score_per_cup": score_per_cup,
                "expected_score_order": score_order,
                "max_fulfillable_cups": max_fulfillable,
                "loss_decision_rule": (
                    f"accept/open only if p >= gate {threshold:.2f}; "
                    f"cost rule p >= penalty/(reward+penalty)="
                    f"{float(self.state['config']['bad_cup_penalty']) / (float(self.state['config']['good_cup_reward']) + float(self.state['config']['bad_cup_penalty'])):.2f}"
                ),
            }
        )

        # Use an already weighed matching bucket first.
        for b in self.matching_buckets(roast, fid, comp):
            available = max(0, int(b["remaining_cups"]) - int(b["reserved_cups"]))
            if available >= cups and p >= threshold:
                event = self._append(
                    "order_accepted",
                    {
                        "order_id": oid,
                        "probability": p,
                        "expected_score": score_order,
                        "reason": f"matching bucket {b['bucket_id']} has {available} unreserved cups",
                        "source": {"kind": "bucket", "bucket_id": b["bucket_id"]},
                    },
                )
                result["decision"] = "accepted"
                result["source"] = event["source"]
                return result
            if available >= cups and p < threshold:
                self._append(
                    "order_rejected",
                    {"order_id": oid, "probability": p, "reason": f"quality gate failed: {p:.3f} < {threshold:.2f}; leave bucket/materials"},
                )
                result["decision"] = "rejected_quality"
                return result

        # Quality gate before occupying a new whole bucket.
        if p < threshold:
            self._append(
                "order_rejected",
                {
                    "order_id": oid,
                    "probability": p,
                    "reason": f"quality gate failed: {p:.3f} < {threshold:.2f}; do not open bucket, leave beans",
                },
            )
            result["decision"] = "rejected_quality"
            return result

        # Find an empty physical bucket whose entire pre-weighed batch fits.
        candidates = []
        for b in self.state["buckets"].values():
            if b["status"] != "empty":
                continue
            cap = int(b["capacity_cups"])
            if cap < cups:
                continue
            # Preserve material already promised to accepted direct orders.
            if self.max_cups_from_part_map(self.available_loose_parts(), comp) < cap:
                continue
            candidates.append((cap, b["bucket_id"]))
        if not candidates:
            loose_max = self.max_loose_cups(comp)
            self._append(
                "order_rejected",
                {
                    "order_id": oid,
                    "probability": p,
                    "reason": f"insufficient material to open a whole bucket; max loose cups={loose_max}",
                },
            )
            result["decision"] = "rejected_material"
            return result

        cap, bid = sorted(candidates)[0]
        stock_after = self._stock_after_deduct(comp, cap)
        weighing = self._make_weighing("whole_bucket", comp, cap, bucket_id=bid)
        self._append(
            "bucket_opened",
            {
                "bucket_id": bid,
                "roast_id": roast,
                "formula_id": fid,
                "composition": comp,
                "cups": cap,
                "stock_after": stock_after,
                "weighing_id": weighing["weighing_id"],
                "weighing": weighing,
                "expected_score_full_bucket": expected_score(p, cap, self.state["config"]),
            },
        )
        self._append(
            "order_accepted",
            {
                "order_id": oid,
                "probability": p,
                "expected_score": score_order,
                "reason": f"opened {bid}: {cap} cups share roast/formula state; p={p:.3f}",
                "source": {"kind": "bucket", "bucket_id": bid},
            },
        )
        result["decision"] = "accepted_and_bucket_opened"
        result["bucket_id"] = bid
        result["weighing"] = weighing
        return result

    def _receive_trial_order(
        self, oid: str, cups: int, comp: Dict[str, int], roast: str, fid: str,
        shift: str, p: float, threshold: float, require_trial: bool, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Existing high posterior/calibrated state can be accepted without
        # consuming the shift's only trial cup.
        if not require_trial and p >= threshold:
            if self.max_cups_from_part_map(self.available_loose_parts(), comp) < cups:
                self._append(
                    "order_rejected",
                    {"order_id": oid, "probability": p, "reason": "insufficient uncommitted material for order"},
                )
                result["decision"] = "rejected_material"
                return result
            self._append(
                "order_accepted",
                {
                    "order_id": oid,
                    "probability": p,
                    "expected_score": expected_score(p, cups, self.state["config"]),
                    "reason": "already calibrated above gate",
                    "source": {"kind": "direct_scale"},
                },
            )
            result["decision"] = "accepted_direct"
            return result

        shift_state = self.state["shifts"].get(shift)
        if shift_state and shift_state.get("trial_used"):
            self._append(
                "order_rejected",
                {
                    "order_id": oid,
                    "probability": p,
                    "reason": f"shift {shift} already used its only trial cup",
                },
            )
            result["decision"] = "rejected_no_trial_cup"
            return result

        # The trial cup consumes one cup's dose even after a bad reading.
        # To be able to accept this order after a "like", uncommitted stock
        # must cover both that trial cup and all ordered cups.
        if self.max_cups_from_part_map(self.available_loose_parts(), comp) < 1 + int(cups):
            self._append(
                "order_rejected",
                {"order_id": oid, "probability": p,
                 "reason": "insufficient uncommitted material for trial cup plus possible accepted order"},
            )
            result["decision"] = "rejected_material"
            return result
        try:
            stock_after = self._stock_after_deduct(comp, 1)
        except ShopError:
            self._append(
                "order_rejected",
                {"order_id": oid, "probability": p, "reason": "insufficient material to weigh trial cup"},
            )
            result["decision"] = "rejected_material"
            return result
        weighing = self._make_weighing("single_trial_cup", comp, 1)
        self._append(
            "trial_prepared",
            {
                "order_id": oid,
                "shift_id": shift,
                "weighing_id": weighing["weighing_id"],
                "weighing": weighing,
                "stock_after": stock_after,
            },
        )
        result["decision"] = "awaiting_trial_reading"
        result["trial_weighing"] = weighing
        result["instruction"] = "Run record-trial with --result like or unlike."
        return result

    def record_trial(self, order_id: str, result: str) -> Dict[str, Any]:
        o = self.state["orders"].get(order_id)
        if not o:
            raise ShopError(f"unknown order {order_id}")
        if o["status"] != "awaiting_reading":
            raise ShopError(f"order {order_id} is {o['status']}, not awaiting_reading")
        roast, fid, comp = o["roast_id"], o["formula_id"], o["composition"]
        key = state_key(roast, fid)
        prior = float(self.state["trial_states"].get(key, {}).get("probability", o.get("probability_before", 0.5)))
        upd = bayes_update(prior, result, float(self.state["config"]["trial_accuracy"]))
        self._append(
            "trial_recorded",
            {
                "order_id": order_id,
                "roast_id": roast,
                "formula_id": fid,
                "update": upd,
            },
        )
        p = upd["posterior"]
        threshold = float(self.state["config"]["gate_threshold"])
        cfg = self.state["config"]
        out = {"order_id": order_id, **upd, "threshold": threshold}

        if p >= threshold:
            if self.max_cups_from_part_map(self.available_loose_parts(), comp) < int(o["cups"]):
                self._append(
                    "order_rejected",
                    {"order_id": order_id, "probability": p,
                     "reason": "trial was good but not enough uncommitted material remains for the paid cup(s)"},
                )
                out["decision"] = "rejected_material"
                return out
            self._append(
                "order_accepted",
                {
                    "order_id": order_id,
                    "probability": p,
                    "expected_score": expected_score(p, int(o["cups"]), cfg),
                    "reason": f"trial reading {upd['reading']} updated posterior to {p:.3f} >= {threshold:.2f}",
                    "source": {"kind": "direct_scale"},
                },
            )
            out["decision"] = "accepted"
        else:
            self._append(
                "order_rejected",
                {
                    "order_id": order_id,
                    "probability": p,
                    "reason": f"trial reading {upd['reading']} updated posterior to {p:.3f} < {threshold:.2f}; reject and keep remaining beans",
                },
            )
            out["decision"] = "rejected_quality"
        return out

    def serve_order(self, order_id: str) -> Dict[str, Any]:
        o = self.state["orders"].get(order_id)
        if not o:
            raise ShopError(f"unknown order {order_id}")
        if o["status"] == "served":
            raise ShopError(f"order {order_id} was already served; duplicate dispensing is forbidden")
        if o["status"] != "accepted":
            raise ShopError(f"cannot serve order {order_id} with status {o['status']}")
        cups = int(o["cups"])
        source = o.get("source") or {}
        if source.get("kind") == "bucket":
            bid = source["bucket_id"]
            b = self.state["buckets"][bid]
            if int(b["remaining_cups"]) < cups:
                raise ShopError(f"bucket {bid} has only {b['remaining_cups']} cups")
            self._append("cup_dispensed", {"order_id": order_id, "bucket_id": bid, "served_cups": cups})
            return {"order_id": order_id, "served_cups": cups, "source": source}

        stock_after = self._stock_after_deduct(o["composition"], cups)
        weighing = self._make_weighing("direct_order", o["composition"], cups)
        self._append(
            "direct_weigh_succeeded",
            {
                "order_id": order_id,
                "weighing_id": weighing["weighing_id"],
                "weighing": weighing,
                "stock_after": stock_after,
            },
        )
        self._append("cup_dispensed", {"order_id": order_id, "bucket_id": None, "served_cups": cups})
        return {"order_id": order_id, "served_cups": cups, "weighing": weighing, "stock_after": stock_after}

    def scan_order(self, order_id: str) -> Dict[str, Any]:
        o = self.state["orders"].get(order_id)
        if not o:
            raise ShopError(f"unknown order {order_id}")
        comp = o["composition"]
        actual: List[Dict[str, Any]] = []
        bucket_info = None
        source = o.get("source") or {}
        if source.get("kind") == "bucket":
            bid = source["bucket_id"]
            b = self.state["buckets"][bid]
            bucket_info = {k: b.get(k) for k in ("bucket_id", "capacity_cups", "remaining_cups", "reserved_cups", "status")}
            if b.get("weighing_id"):
                w = next((w for w in self.state["weighings"] if w["weighing_id"] == b["weighing_id"]), None)
                if w:
                    actual.append(w)
        for wid in o.get("direct_weighing_ids", []):
            w = next((w for w in self.state["weighings"] if w["weighing_id"] == wid), None)
            if w:
                actual.append(w)
        if o.get("trial_weighing_id"):
            w = next((w for w in self.state["weighings"] if w["weighing_id"] == o["trial_weighing_id"]), None)
            if w:
                actual.append(w)

        # Remaining active half-buckets and open orders that share their state.
        refill = []
        for b in self.state["buckets"].values():
            if b["status"] != "active" or b["remaining_cups"] <= 0:
                continue
            candidates = []
            for oid2, other in self.state["orders"].items():
                if oid2 == order_id:
                    continue
                if other["status"] not in {"received", "accepted", "awaiting_reading"}:
                    continue
                if other["roast_id"] == b["roast_id"] and tuple(sorted(other["composition"].items())) == tuple(sorted(b["composition"].items())):
                    candidates.append({"order_id": oid2, "status": other["status"], "cups": other["cups"]})
            fraction = b["remaining_cups"] / b["capacity_cups"] if b["capacity_cups"] else None
            refill.append(
                {
                    "bucket_id": b["bucket_id"],
                    "remaining_cups": b["remaining_cups"],
                    "capacity_cups": b["capacity_cups"],
                    "remaining_fraction_of_bucket": fraction,
                    "is_half_bucket": abs((fraction or 0) - 0.5) < 1e-9,
                    "can_refill_orders": candidates,
                }
            )
        return {
            "scanned_order_id": order_id,
            "status": o["status"],
            "roast_id": o["roast_id"],
            "formula_id": o["formula_id"],
            "requested_composition": comp,
            "order_dose_lines": self._weigh_lines(comp, int(o["cups"])),
            "display_recipe": composition_to_string(comp),
            "probability_at_decision": o.get("probability_at_decision", o.get("probability_before")),
            "decision": o.get("decision"),
            "source": source,
            "actual_weighings": actual,
            "bucket": bucket_info,
            "remaining_buckets_and_refills": refill,
            "immutable_ledger": self.ledger_path,
        }

    def stock_report(self) -> Dict[str, Any]:
        total_loose_parts = sum(int(v) for v in self.state["loose_stock_parts"].values())
        buckets = []
        for b in self.state["buckets"].values():
            if b["status"] == "empty":
                buckets.append({"bucket_id": b["bucket_id"], "status": "empty", "capacity_cups": b["capacity_cups"]})
            else:
                buckets.append(
                    {
                        "bucket_id": b["bucket_id"],
                        "status": b["status"],
                        "roast_id": b["roast_id"],
                        "formula_id": b["formula_id"],
                        "composition": b["composition"],
                        "remaining_cups": b["remaining_cups"],
                        "reserved_cups": b["reserved_cups"],
                        "unreserved_cups": max(0, b["remaining_cups"] - b["reserved_cups"]),
                    }
                )
        committed_direct = self.direct_committed_parts()
        available_loose = self.available_loose_parts()
        return {
            "loose_stock_parts": self.state["loose_stock_parts"],
            "direct_committed_parts": committed_direct,
            "available_loose_parts": available_loose,
            "total_loose_parts": total_loose_parts,
            "buckets": buckets,
            "orders": {oid: {"status": o["status"], "cups": o["cups"]} for oid, o in self.state["orders"].items()},
        }
