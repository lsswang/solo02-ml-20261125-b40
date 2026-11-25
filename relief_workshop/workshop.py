from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ioformats import write_json
from .planning import (
    contingency_holes,
    evaluate_request,
    build_start_candidates,
    cut_positions_for_plan,
)


class WorkshopStore:
    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.orders_dir = self.dir / "orders"
        self.orders_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "ticket_index.json"
        if not self.index_path.exists():
            write_json(self.index_path, {})

    def _state_path(self, order_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in order_id)
        return self.orders_dir / f"{safe}.json"

    def load_index(self) -> dict[str, str]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def save_index(self, index: dict[str, str]) -> None:
        write_json(self.index_path, index)

    def save_state(self, state: dict) -> Path:
        path = self._state_path(state["order_id"])
        write_json(path, state)
        index = self.load_index()
        index[state["order_id"]] = state["order_id"]
        for ticket in state.get("ticket_ids", []):
            index[ticket] = state["order_id"]
        self.save_index(index)
        return path

    def resolve_parent_id(self, query_id: str) -> str:
        if self._state_path(query_id).exists():
            return query_id
        index = self.load_index()
        if query_id in index:
            return index[query_id]
        raise FileNotFoundError(f"no work order or ticket known as {query_id}")

    def load_state(self, order_id: str) -> dict:
        parent = self.resolve_parent_id(order_id)
        return json.loads(self._state_path(parent).read_text(encoding="utf-8"))

    def start(self, edge_doc: dict, order: dict, edge_source: str | None = None) -> dict:
        order_id = str(order["order_id"])
        report = build_start_candidates(edge_doc, order)
        state: dict[str, Any] = {
            "order_id": order_id,
            "rows": int(order["rows"]),
            "cols": int(order["cols"]),
            "edge_source": edge_source,
            "edge_doc": edge_doc,
            "order": order,
            "ticket_ids": [order_id],
            "cuts": [],
            "rejections": [],
            "accepted_extra_holes": [],
            "history": [],
        }

        if not report["feasible"]:
            state.update({
                "status": "no_feasible_solution",
                "method": None,
                "seam_column": None,
                "last_full_heights": None,
            })
            state["history"].append({
                "work_order_id": order_id,
                "action": "start_attempt",
                "status": "no_feasible_solution",
                "reason": report.get("solver_reason"),
            })
            path = self.save_state(state)
            report["state_path"] = str(path)
            report["order_id"] = order_id
            return report

        selected = report["selected"]
        cuts = selected["cuts"]
        state.update({
            "status": "locked_whole" if selected["method"] == "whole" else "seam_open",
            "method": selected["method"],
            "seam_column": selected["seam_column"],
            "last_full_heights": selected["heights"],
            "cuts": cuts,
            "selected_summary": {
                "method": selected["method"],
                "seam_column": selected["seam_column"],
                "score": selected["score"],
                "expected_rejection_loss": selected["risk"]["expected_rejection_loss"],
                "extra_station_required": selected.get("extra_station", False),
                "station_requirement": selected.get("station_requirement"),
                "selection_reason": report["selection_reason"],
            },
            "start_report": _json_safe_report(report),
        })
        state["history"].append({
            "work_order_id": order_id,
            "action": "start",
            "status": state["status"],
            "method": selected["method"],
            "seam_column": selected["seam_column"],
            "signed_cut_count": len(cuts),
            "signed_cuts": cuts,
            "optimal_score": report["optimal_score"],
            "expected_rejection_loss": selected["risk"]["expected_rejection_loss"],
        })
        path = self.save_state(state)
        report["state_path"] = str(path)
        return report

    def supplement(self, parent_or_ticket_id: str, ticket: dict,
                  finalize_seam: bool = True) -> dict:
        query_id = parent_or_ticket_id
        parent_id = self.resolve_parent_id(query_id)
        state = json.loads(self._state_path(parent_id).read_text(encoding="utf-8"))
        ticket_id = str(ticket.get("ticket_id") or ticket.get("order_id") or f"T-{len(state['history']) + 1}")
        if ticket_id not in state["ticket_ids"]:
            state["ticket_ids"].append(ticket_id)

        rows, cols = state["rows"], state["cols"]
        new_holes = contingency_holes(ticket, rows)
        known_holes = list(state.get("accepted_extra_holes", [])) + new_holes

        if state["status"] == "no_feasible_solution":
            response = {
                "ticket_id": ticket_id, "parent_order_id": parent_id,
                "accepted": False, "reason": "original_plate_has_no_feasible_solution",
                "cuts_retained": state["cuts"],
            }
            state["rejections"].append(response)
            state["history"].append({"work_order_id": ticket_id, "action": "supplement", **response})
            self.save_state(state)
            return response

        ev = evaluate_request(state["edge_doc"], state["order"], state["cuts"], known_holes)
        response: dict[str, Any] = {
            "ticket_id": ticket_id,
            "parent_order_id": parent_id,
            "accepted": ev["accepted"],
            "reason": ev["reason"],
            "conflicts": ev.get("conflicts", []),
            "score": ev.get("score"),
        }

        if not ev["accepted"]:
            response["cuts_retained"] = state["cuts"]
            state["rejections"].append(response)
            state["history"].append({
                "work_order_id": ticket_id, "action": "supplement",
                "status": "rejected", **response,
            })
            self.save_state(state)
            return response

        state["accepted_extra_holes"] = known_holes
        state["last_full_heights"] = ev["heights"]
        signed_cuts: list[dict] = []

        if state["method"] == "segment" and state["status"] == "seam_open":
            if finalize_seam:
                signed_cuts = cut_positions_for_plan(
                    tuple(ev["heights"]), rows, cols, "segment",
                    state["seam_column"], ticket_id, "seam"
                )
                for cut in signed_cuts:
                    cut["parent_order_id"] = parent_id
                state["cuts"].extend(signed_cuts)
                state["status"] = "completed"
            else:
                response["note"] = "hole recorded as compatible; seam intentionally left open"
        elif state["method"] == "whole":
            response["note"] = "whole plate was already locked; requested heights match actual cuts"

        response["signed_cuts"] = signed_cuts
        response["height_map"] = ev.get("feasible_plan")
        state["history"].append({
            "work_order_id": ticket_id,
            "parent_order_id": parent_id,
            "action": "supplement",
            "status": "accepted",
            "finalized": bool(signed_cuts) or state["method"] == "whole",
            "signed_cuts": signed_cuts,
            "score": ev.get("score"),
            "holes": new_holes,
        })
        self.save_state(state)
        response["all_cuts"] = state["cuts"]
        return response

    def cuts_report(self, query_id: str) -> dict:
        parent_id = self.resolve_parent_id(query_id)
        state = json.loads(self._state_path(parent_id).read_text(encoding="utf-8"))
        if query_id == parent_id:
            ticket_cuts = state["cuts"]
        else:
            ticket_cuts = [c for c in state["cuts"] if c.get("work_order_id") == query_id]
        ticket_rejections = [
            r for r in state.get("rejections", [])
            if r.get("ticket_id") == query_id or query_id == parent_id
        ]
        if query_id == parent_id:
            history = state.get("history", [])
        else:
            history = [h for h in state.get("history", [])
                       if h.get("work_order_id") == query_id]
        queried_cuts = state["cuts"] if query_id == parent_id else ticket_cuts
        return {
            "queried_work_order": query_id,
            "parent_order_id": parent_id,
            "status": state["status"],
            "method": state["method"],
            "seam_column": state.get("seam_column"),
            "station_requirement": (state.get("selected_summary") or {}).get("station_requirement"),
            "cut_count": len(queried_cuts),
            "board_cut_count": len(state["cuts"]),
            "cuts": queried_cuts,
            "cuts_for_queried_ticket": ticket_cuts,
            "ticket_history": history,
            "rejections": ticket_rejections,
            "latest_height_map": _heights_to_map(state.get("last_full_heights"),
                                                 state["rows"], state["cols"]),
        }


def _heights_to_map(heights, rows: int, cols: int):
    if not heights:
        return None
    return [[heights[r * cols + c] for c in range(cols)] for r in range(rows)]


def _json_safe_report(report: dict) -> dict:
    # The report currently contains only JSON-safe ints/floats/lists/dicts.
    return json.loads(json.dumps(report, allow_nan=False))
