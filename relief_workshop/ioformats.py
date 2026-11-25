from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .grid import DELTAS, Edge, GridProblem, coord_to_index
from .model import GaussianNBModel


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def predict_scan(scan_path: str | Path, model: GaussianNBModel) -> dict:
    rows = read_csv(scan_path)
    if not rows:
        raise ValueError("scan CSV has no rows")
    rows_n = int(rows[0]["rows"])
    cols_n = int(rows[0]["cols"])
    edges = []
    for row in rows:
        x = [float(row[f"ch{i}"]) for i in range(4)]
        probs = model.probabilities(x)
        edges.append({
            "edge_id": row["edge_id"],
            "u_row": int(row["u_row"]),
            "u_col": int(row["u_col"]),
            "v_row": int(row["v_row"]),
            "v_col": int(row["v_col"]),
            "probabilities": {str(d): probs[d] for d in DELTAS},
        })
    return {
        "format": "relief-edge-probabilities-v1",
        "rows": rows_n,
        "cols": cols_n,
        "coordinate_base": 1,
        "probability_kind": "raw_calibrated",
        "edges": edges,
    }


def _as_int_set(value: Any) -> set[int] | None:
    if value is None:
        return None
    return {int(x) for x in value}


def gauge_allowed_deltas(edge_obj: dict, gauge: dict | None) -> set[int]:
    gauge = gauge or {}
    allowed = set(DELTAS)

    global_allowed = _as_int_set(gauge.get("allowed_deltas"))
    if global_allowed is not None:
        allowed &= global_allowed
    global_forbidden = _as_int_set(gauge.get("forbidden_deltas"))
    if global_forbidden:
        allowed -= global_forbidden

    ur, uc = int(edge_obj["u_row"]), int(edge_obj["u_col"])
    vr, vc = int(edge_obj["v_row"]), int(edge_obj["v_col"])
    horizontal = ur == vr and uc != vc
    vertical = uc == vc and ur != vr

    ori = "horizontal" if horizontal else "vertical" if vertical else "directed"
    ori_allowed = _as_int_set(gauge.get(f"{ori}_allowed"))
    if ori_allowed is not None:
        allowed &= ori_allowed
    ori_forbidden = _as_int_set(gauge.get(f"{ori}_forbidden"))
    if ori_forbidden:
        allowed -= ori_forbidden

    eid = edge_obj["edge_id"]
    edge_allowed = (gauge.get("edge_allowed") or {}).get(eid)
    if edge_allowed is not None:
        allowed &= {int(x) for x in edge_allowed}
    edge_forbidden = (gauge.get("edge_forbidden") or {}).get(eid)
    if edge_forbidden:
        allowed -= {int(x) for x in edge_forbidden}

    # A calibrated zero is itself a hard prohibition.
    probs = edge_obj["probabilities"]
    for d in DELTAS:
        if float(probs[str(d)]) <= 0.0:
            allowed.discard(d)
    return allowed


def effective_edge_table(edge_doc: dict, gauge: dict | None) -> list[dict]:
    """Apply stop-gauge exclusions and renormalize the surviving probabilities."""
    table = []
    for e in edge_doc["edges"]:
        raw = [float(e["probabilities"][str(d)]) for d in DELTAS]
        allowed_set = gauge_allowed_deltas(e, gauge)
        allowed_mask = [d in allowed_set for d in DELTAS]
        mass = sum(raw[i] for i, ok in enumerate(allowed_mask) if ok)
        eff_prob = [0.0] * 5
        logw = [math.inf * -1] * 5
        conditional_logw = [math.inf * -1] * 5
        if mass > 0:
            for i, ok in enumerate(allowed_mask):
                if ok and raw[i] > 0.0:
                    eff_prob[i] = raw[i] / mass
                    # MAP score uses the original calibrated log probabilities;
                    # the deterministic gauge is a hard feasibility mask.
                    logw[i] = math.log(raw[i])
                    conditional_logw[i] = math.log(eff_prob[i])
        table.append({
            "edge_id": e["edge_id"],
            "u_row": int(e["u_row"]), "u_col": int(e["u_col"]),
            "v_row": int(e["v_row"]), "v_col": int(e["v_col"]),
            "raw_prob": raw,
            "allowed_mask": allowed_mask,
            "effective_prob": eff_prob,
            "log_weight": logw,
            "conditional_log_weight": conditional_logw,
        })
    return table


def build_problem(edge_doc: dict, gauge: dict | None = None,
                  constraints: list[dict] | None = None) -> tuple[GridProblem, list[dict]]:
    rows = int(edge_doc["rows"])
    cols = int(edge_doc["cols"])
    base = int(edge_doc.get("coordinate_base", 1))
    table = effective_edge_table(edge_doc, gauge)
    edges: list[Edge] = []
    for t in table:
        u = coord_to_index(t["u_row"], t["u_col"], cols, base)
        v = coord_to_index(t["v_row"], t["v_col"], cols, base)
        edges.append(Edge(t["edge_id"], u, v, t["log_weight"], t["raw_prob"], t["allowed_mask"]))

    problem = GridProblem(rows=rows, cols=cols, edges=edges)
    applied = apply_constraints(problem, constraints or [], base)
    return problem, applied


def apply_constraints(problem: GridProblem, constraints: list[dict], base: int = 1) -> list[dict]:
    applied = []
    for con in constraints:
        r = int(con["row"])
        c = int(con["col"])
        v = coord_to_index(r, c, problem.cols, base)
        before = problem.unary_mask[v]
        if "height" in con and con["height"] is not None:
            h = int(con["height"])
            problem.unary_mask[v] &= 1 << h
            spec = {"row": r, "col": c, "height": h}
        elif "allowed_heights" in con and con["allowed_heights"] is not None:
            hs = [int(x) for x in con["allowed_heights"]]
            mask = 0
            for h in hs:
                mask |= 1 << h
            problem.unary_mask[v] &= mask
            spec = {"row": r, "col": c, "allowed_heights": hs}
        else:
            continue
        spec["vertex_index"] = v
        spec["became_empty"] = problem.unary_mask[v] == 0
        spec["removed_previously_allowed"] = before != problem.unary_mask[v]
        applied.append(spec)
    return applied


def order_constraints(order: dict) -> list[dict]:
    """Normalize holes, column locks and column allowed-height maps."""
    out: list[dict] = []
    for hole in order.get("holes", []):
        out.append(dict(hole))
    for col_lock in order.get("fixed_columns", []):
        col = int(col_lock["col"])
        h = int(col_lock["height"])
        for r in range(1, int(order["rows"]) + 1):
            out.append({"row": r, "col": col, "height": h, "source": "fixed_column"})
    for col_allowed in order.get("column_allowed", []):
        col = int(col_allowed["col"])
        hs = col_allowed["allowed_heights"]
        for r in range(1, int(order["rows"]) + 1):
            out.append({"row": r, "col": col, "allowed_heights": hs, "source": "column_allowed"})
    return out
