from __future__ import annotations

from itertools import product
import math
from typing import Any

from .grid import HEIGHTS, index_to_coord
from .ioformats import build_problem, order_constraints
from .solver import SolveResult, brute_force_optima, solve_map

SCORE_TOL = 1e-8


def parse_probability(value: Any) -> float:
    """Accept 0.75, ``"3/4"`` and ``"75%"`` in addition to JSON numbers."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.endswith("%"):
        return float(text[:-1].strip()) / 100.0
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator.strip()) / float(denominator.strip())
    return float(text)


def contingency_holes(contingency: dict, rows: int) -> list[dict]:
    holes = [dict(h) for h in contingency.get("holes", [])]
    if "column" in contingency:
        col = int(contingency["column"])
        height = int(contingency["height"])
        for r in range(1, rows + 1):
            holes.append({"row": r, "col": col, "height": height})
    if "row" in contingency and "height" in contingency and "col" not in contingency:
        # A whole-row lock, useful for transposed stamps.
        row = int(contingency["row"])
        height = int(contingency["height"])
        for c in range(1, int(contingency.get("cols") or 999) + 1):
            holes.append({"row": row, "col": c, "height": height})
    return holes


def cut_constraints(cuts: list[dict]) -> list[dict]:
    return [{"row": int(x["row"]), "col": int(x["col"]), "height": int(x["height"]),
             "source": "already_cut"} for x in cuts]


def find_cut_conflicts(cuts: list[dict], holes: list[dict], cols: int) -> list[dict]:
    cut_by_vertex = {}
    for cut in cuts:
        v = (int(cut["row"]) - 1) * cols + (int(cut["col"]) - 1)
        cut_by_vertex[v] = cut
    conflicts = []
    for hole in holes:
        v = (int(hole["row"]) - 1) * cols + (int(hole["col"]) - 1)
        if v in cut_by_vertex and "height" in hole:
            old = int(cut_by_vertex[v]["height"])
            new = int(hole["height"])
            if old != new:
                conflicts.append({
                    "row": int(hole["row"]), "col": int(hole["col"]),
                    "already_cut_height": old, "requested_height": new,
                    "cut_id": cut_by_vertex[v].get("cut_id"),
                    "work_order_id": cut_by_vertex[v].get("work_order_id"),
                })
    return conflicts


def solve_with_known(edge_doc: dict, order: dict, cuts: list[dict] | None = None,
                     extra_constraints: list[dict] | None = None) -> tuple[SolveResult, Any]:
    constraints = order_constraints(order)
    constraints.extend(cut_constraints(cuts or []))
    constraints.extend(extra_constraints or [])
    problem, applied = build_problem(edge_doc, order.get("gauge"), constraints)
    return solve_map(problem), applied


def evaluate_request(edge_doc: dict, order: dict, cuts: list[dict],
                     holes: list[dict]) -> dict:
    """Check a formal/probabilistic hole request against physical cuts and scan."""
    cols = int(edge_doc["cols"])
    conflicts = find_cut_conflicts(cuts, holes, cols)
    if conflicts:
        return {"accepted": False, "reason": "cut_conflict", "conflicts": conflicts,
                "feasible_plan": None}

    result, _ = solve_with_known(edge_doc, order, cuts, holes)
    if not result.feasible:
        return {"accepted": False, "reason": "no_feasible_full_plate",
                "conflicts": [], "solver_reason": result.reason, "feasible_plan": None}
    return {
        "accepted": True,
        "reason": "compatible_and_feasible",
        "conflicts": [],
        "score": result.score,
        "heights": list(result.heights) if result.heights is not None else None,
        "feasible_plan": height_map(result.heights, int(edge_doc["rows"]), cols),
    }


def height_map(heights: tuple[int, ...] | list[int] | None, rows: int, cols: int) -> list[list[int]] | None:
    if heights is None:
        return None
    return [[int(heights[r * cols + c]) for c in range(cols)] for r in range(rows)]


def cut_positions_for_plan(heights: tuple[int, ...], rows: int, cols: int,
                           method: str, seam_column: int | None,
                           work_order_id: str, phase: str) -> list[dict]:
    cuts = []
    for v, h in enumerate(heights):
        r0, c0 = divmod(v, cols)
        row, col = r0 + 1, c0 + 1
        is_seam = method == "segment" and seam_column == col
        if phase == "first" and is_seam:
            continue
        if phase == "seam" and not is_seam:
            continue
        cuts.append({
            "cut_id": f"C-{row}-{col}-{phase}",
            "row": row,
            "col": col,
            "vertex_index": v,
            "height": int(h),
            "phase": phase,
            "work_order_id": work_order_id,
        })
    return cuts


def expected_contingency_loss(edge_doc: dict, order: dict, cuts: list[dict]) -> dict:
    items = []
    total = 0.0
    rows = int(order["rows"])
    for cont in order.get("contingencies", []):
        holes = contingency_holes(cont, rows)
        ev = evaluate_request(edge_doc, order, cuts, holes)
        p = parse_probability(cont.get("probability", 0.0))
        loss = float(cont.get("rejection_loss", 0.0))
        expected_reject = p * loss if not ev["accepted"] else 0.0
        total += expected_reject
        items.append({
            "contingency_id": cont.get("contingency_id", cont.get("ticket_id", "?")),
            "label": cont.get("label", ""),
            "probability": p,
            "rejection_loss": loss,
            "holes": holes,
            "accepted": ev["accepted"],
            "reason": ev["reason"],
            "conflicts": ev.get("conflicts", []),
            "expected_rejection_loss": round(expected_reject, 12),
            "possible_height_map": ev.get("feasible_plan"),
        })
    return {"expected_rejection_loss": total, "items": items}


def seam_legal_margin(edge_doc: dict, order: dict, cuts: list[dict],
                      seam_column: int, optimal_score: float | None = None) -> dict:
    """Enumerate legal heights of the one open seam column.

    All non-seam vertices have already been cut.  A tuple is legal when the
    resulting complete 0..3 plate satisfies gauge, holes and edge probabilities.
    Optimal tuples additionally retain the pre-cut MAP score.
    """
    rows, cols = int(edge_doc["rows"]), int(edge_doc["cols"])
    result, _ = solve_with_known(edge_doc, order, cuts)
    if not result.feasible or result.heights is None:
        return {"seam_column": seam_column, "legal": False, "tuples": []}

    problem, _ = build_problem(
        edge_doc, order.get("gauge"),
        order_constraints(order) + cut_constraints(cuts)
    )
    seam_vertices = [(r - 1) * cols + (seam_column - 1) for r in range(1, rows + 1)]
    cut_values = {int(c["vertex_index"]): int(c["height"]) for c in cuts
                  if "vertex_index" in c}
    legal_tuples: list[dict] = []
    optimal_tuples: list[list[int]] = []
    allowed_by_row: list[set[int]] = [set() for _ in range(rows)]

    for tup in product(HEIGHTS, repeat=rows):
        h = [0] * problem.n_vertices
        for v, value in cut_values.items():
            h[v] = value
        for i, v in enumerate(seam_vertices):
            h[v] = tup[i]
        score = problem.score(h)
        if not math.isfinite(score):
            continue
        legal_tuples.append({"heights_top_to_bottom": list(tup), "score": score})
        for i, value in enumerate(tup):
            allowed_by_row[i].add(value)
        if optimal_score is not None and abs(score - optimal_score) <= SCORE_TOL:
            optimal_tuples.append(list(tup))

    return {
        "seam_column": seam_column,
        "legal": bool(legal_tuples),
        "legal_tuple_count": len(legal_tuples),
        "allowed_heights_by_row": [sorted(x) for x in allowed_by_row],
        "optimal_tuples": optimal_tuples,
        "tuples": legal_tuples,
    }


def assignment_request_feasible(problem, rows: int, cols: int, method: str,
                               seam_column: int | None,
                               h: tuple[int, ...], holes: list[dict]) -> bool:
    """Fast tie-break test for one already-optimal commitment.

    For whole milling every requested point must equal h.  For a segment seam,
    cut columns must equal h while the single open column is enumerated.
    """
    seam_vertices: set[int] = set()
    if method == "segment" and seam_column is not None:
        seam_vertices = {r * cols + (seam_column - 1) for r in range(rows)}

    seam_exact: dict[int, int] = {}
    seam_allowed: dict[int, set[int]] = {}
    for hole in holes:
        v = (int(hole["row"]) - 1) * cols + (int(hole["col"]) - 1)
        if v in seam_vertices:
            if "height" in hole and hole["height"] is not None:
                value = int(hole["height"])
                if v in seam_exact and seam_exact[v] != value:
                    return False
                seam_exact[v] = value
            if hole.get("allowed_heights") is not None:
                values = {int(x) for x in hole["allowed_heights"]}
                seam_allowed[v] = values if v not in seam_allowed else seam_allowed[v] & values
                if not seam_allowed[v]:
                    return False
        else:
            if "height" in hole and hole["height"] is not None and h[v] != int(hole["height"]):
                return False
            if hole.get("allowed_heights") is not None and h[v] not in {int(x) for x in hole["allowed_heights"]}:
                return False

    if method == "whole":
        return True

    seam_list = sorted(seam_vertices)
    for tup in product(HEIGHTS, repeat=len(seam_list)):
        g = list(h)
        ok = True
        for v, value in zip(seam_list, tup):
            if v in seam_exact and seam_exact[v] != value:
                ok = False
                break
            if v in seam_allowed and value not in seam_allowed[v]:
                ok = False
                break
            g[v] = value
        if ok and math.isfinite(problem.score(g)):
            return True
    return False


def expected_loss_for_optimal_h(problem, order: dict, method: str,
                                seam_column: int | None, h: tuple[int, ...]) -> float:
    rows, cols = problem.rows, problem.cols
    total = 0.0
    for cont in order.get("contingencies", []):
        holes = contingency_holes(cont, rows)
        accepted = assignment_request_feasible(
            problem, rows, cols, method, seam_column, h, holes
        )
        if not accepted:
            total += parse_probability(cont.get("probability", 0.0)) * float(cont.get("rejection_loss", 0.0))
    return total


def target_deltas(edge_doc: dict, heights: tuple[int, ...] | list[int]) -> dict[str, int]:
    cols = int(edge_doc["cols"])
    base = int(edge_doc.get("coordinate_base", 1))
    out = {}
    for e in edge_doc["edges"]:
        u = (int(e["u_row"]) - base) * cols + (int(e["u_col"]) - base)
        v = (int(e["v_row"]) - base) * cols + (int(e["v_col"]) - base)
        out[e["edge_id"]] = int(heights[v] - heights[u])
    return out


def build_start_candidates(edge_doc: dict, order: dict) -> dict[str, Any]:
    rows, cols = int(edge_doc["rows"]), int(edge_doc["cols"])
    if int(order.get("rows", rows)) != rows or int(order.get("cols", cols)) != cols:
        raise ValueError("order rows/cols do not match the edge-probability scan")
    constraints = order_constraints(order)
    base_problem, _ = build_problem(edge_doc, order.get("gauge"), constraints)
    base_result = solve_map(base_problem)
    if not base_result.feasible:
        return {
            "feasible": False,
            "solver_reason": base_result.reason,
            "base_score": base_result.score,
            "candidates": [],
        }

    best_score = float(base_result.score)
    # For workshop-sized tiny stamps, enumerate every tied MAP assignment so
    # the expected-loss tie-break is exact even when several plates tie.
    if base_problem.n_vertices <= 8:
        optimal_heights = brute_force_optima(base_problem)
    else:
        optimal_heights = [base_result.heights]
    policy = order.get("method_policy", "any")
    seam_columns = [int(x) for x in order.get("seam_columns", [])]
    # De-duplicate and validate.
    seam_columns = sorted(set(seam_columns))
    for s in seam_columns:
        if not 1 <= s <= cols:
            raise ValueError(f"seam column {s} is outside 1..{cols}")

    candidates: list[dict] = []

    def add_candidate(method: str, seam: int | None) -> None:
        best_choice = None
        # Every h here is a globally optimal full plate.  Choose the actual
        # first-cut commitment with the smallest expected future rejection loss.
        for h in optimal_heights:
            if h is None:
                continue
            loss = expected_loss_for_optimal_h(
                base_problem, order, method, seam, h
            )
            key = (loss, h)
            if best_choice is None or key < best_choice[0]:
                best_choice = (key, h)
        if best_choice is None:
            return
        h = best_choice[1]
        if method == "whole":
            cuts = cut_positions_for_plan(h, rows, cols, method, None,
                                          str(order["order_id"]), "whole")
        else:
            cuts = cut_positions_for_plan(h, rows, cols, method, seam,
                                          str(order["order_id"]), "first")
        # Rich report with per-contingency conflicts/possible maps is built once.
        risk = expected_contingency_loss(edge_doc, order, cuts)
        candidates.append({
            "method": method,
            "seam_column": seam,
            "score": float(base_result.score),
            "score_tied": abs(float(base_result.score) - best_score) <= SCORE_TOL,
            "tied_optimal_assignment_count": len(optimal_heights),
            "heights": list(h),
            "target_deltas": target_deltas(edge_doc, h),
            "cuts": cuts,
            "risk": risk,
            "height_map": height_map(h, rows, cols),
            "extra_station": method == "segment",
            "station_requirement": (
                None if method == "whole" else {
                    "station_id": f"SEAM-COL-{seam}",
                    "kind": "open_seam",
                    "seam_column": seam,
                    "extra_station_count": 1,
                }
            ),
        })

    if policy in ("any", "whole"):
        add_candidate("whole", None)
    if policy in ("any", "segment"):
        for s in seam_columns:
            add_candidate("segment", s)

    tied = [c for c in candidates if c["score_tied"]]
    if not tied:
        # Defensive: every candidate currently uses the same global MAP.
        tied = candidates
    min_risk = min(c["risk"]["expected_rejection_loss"] for c in tied)
    risk_tied = [c for c in tied
                 if abs(c["risk"]["expected_rejection_loss"] - min_risk) <= 1e-10]
    # Deterministic final tie-break: avoid the extra seam station if risk ties,
    # otherwise use the leftmost allowed seam.
    risk_tied.sort(key=lambda c: (0 if c["method"] == "whole" else 1,
                                  c["seam_column"] if c["seam_column"] is not None else 0))
    selected = risk_tied[0]
    selected["extra_station_required"] = selected["method"] == "segment"
    if selected["method"] == "segment":
        selected["legal_margin"] = seam_legal_margin(
            edge_doc, order, selected["cuts"], selected["seam_column"], best_score
        )
    else:
        selected["legal_margin"] = None
    return {
        "feasible": True,
        "optimal_score": best_score,
        "optimal_height_map": height_map(base_result.heights, rows, cols),
        "cycle_errors": _cycle_errors(edge_doc, base_result.heights),
        "candidates": candidates,
        "selected": selected,
        "selection_reason": _selection_reason(tied, selected, min_risk),
    }


def _cycle_errors(edge_doc: dict, heights: tuple[int, ...] | None) -> list[dict]:
    if heights is None:
        return []
    # Reconstruct only for cycle validation.
    problem, _ = build_problem(edge_doc, {}, [])
    return problem.cycle_checks(heights)


def _selection_reason(candidates: list[dict], selected: dict, min_risk: float) -> str:
    methods = ", ".join(
        f"{c['method']}" + (f":col{c['seam_column']}" if c["seam_column"] else "")
        + f"={c['risk']['expected_rejection_loss']:g}"
        for c in candidates
    )
    return (f"All start plans share the optimal score; expected rejection losses are "
            f"[{methods}]. Selected {selected['method']}"
            + (f" seam column {selected['seam_column']}" if selected["seam_column"] else "")
            + f" with minimum expected loss {min_risk:g}.")
