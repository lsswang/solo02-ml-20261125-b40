from __future__ import annotations

from .grid import DELTAS
from .ioformats import effective_edge_table


def local_argmax_report(edge_doc: dict, gauge: dict | None = None) -> dict:
    """Report per-edge modal slopes and elementary-loop closure failures.

    The modal slope of each edge is not necessarily a realizable height field.
    A non-zero loop sum is exactly such a contradiction.
    """
    rows, cols = int(edge_doc["rows"]), int(edge_doc["cols"])
    table = effective_edge_table(edge_doc, gauge)
    by_id = {t["edge_id"]: t for t in table}
    edges = []
    deltas: dict[str, int] = {}
    for t in table:
        raw_best_i = max(range(5), key=lambda i: t["raw_prob"][i])
        eff_best_i = max(range(5), key=lambda i: t["effective_prob"][i])
        raw_d = DELTAS[raw_best_i]
        eff_d = DELTAS[eff_best_i] if any(t["allowed_mask"]) else None
        deltas[t["edge_id"]] = eff_d if eff_d is not None else raw_d
        edges.append({
            "edge_id": t["edge_id"],
            "raw_modal_delta": raw_d,
            "gauge_constrained_modal_delta": eff_d,
            "raw_probabilities": {str(DELTAS[i]): t["raw_prob"][i] for i in range(5)},
        })

    loops = []
    all_close = True
    for r in range(1, rows):
        for c in range(1, cols):
            top = f"H-r{r}-c{c}"
            bottom = f"H-r{r + 1}-c{c}"
            left = f"V-r{r}-c{c}"
            right = f"V-r{r}-c{c + 1}"
            if not all(x in deltas for x in (top, bottom, left, right)):
                continue
            s = deltas[top] + deltas[right] - deltas[bottom] - deltas[left]
            if s != 0:
                all_close = False
            loops.append({"row": r, "col": c, "clockwise_sum": s, "closes": s == 0})

    return {
        "uses_gauge_constrained_modes": gauge is not None,
        "all_elementary_loops_close": all_close,
        "loop_checks": loops,
        "edges": edges,
    }
