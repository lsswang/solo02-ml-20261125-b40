from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .ioformats import build_problem, load_json, order_constraints, write_json
from .model import train_gaussian_nb, mean_log_loss, GaussianNBModel
from .solver import brute_force_best, solve_map
from .synthetic import generate_project
from .diagnostics import local_argmax_report
from .workshop import WorkshopStore


def _print_json(obj: Any, out: str | Path | None = None) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
    print(text)


def cmd_generate(args: argparse.Namespace) -> int:
    paths = generate_project(args.out, seed=args.seed,
                             scan_rows=args.rows, scan_cols=args.cols)
    _print_json({k: str(v) for k, v in paths.items()})
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    import csv
    with Path(args.train).open(newline="", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    model = train_gaussian_nb(train_rows, var_floor=args.var_floor,
                              label_smoothing=args.label_smoothing)
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    report: dict[str, Any] = {
        "n_train": len(train_rows),
        "model": str(model_path),
        "classes": [-2, -1, 0, 1, 2],
    }
    if args.holdout:
        with Path(args.holdout).open(newline="", encoding="utf-8") as f:
            holdout_rows = list(csv.DictReader(f))
        loss, acc = mean_log_loss(model, holdout_rows)
        report.update({
            "holdout": args.holdout,
            "holdout_coupon_id": holdout_rows[0].get("coupon_id") if holdout_rows else None,
            "n_holdout_edges": len(holdout_rows),
            "holdout_mean_log_loss_nats": loss,
            "holdout_total_log_loss_nats": loss * len(holdout_rows),
            "holdout_accuracy": acc,
        })
    if args.report:
        write_json(args.report, report)
    _print_json(report)
    return 0


def cmd_scan_edges(args: argparse.Namespace) -> int:
    model = GaussianNBModel.from_dict(load_json(args.model))
    from .ioformats import predict_scan
    doc = predict_scan(args.scan, model)
    write_json(args.out, doc)
    print(f"wrote {len(doc['edges'])} directed edge probability rows to {args.out}", file=sys.stderr)
    return 0


def _margin_compact(margin: dict | None) -> dict | None:
    if not margin:
        return None
    return {
        "seam_column": margin.get("seam_column"),
        "legal": margin.get("legal"),
        "legal_tuple_count": margin.get("legal_tuple_count"),
        "allowed_heights_by_row": margin.get("allowed_heights_by_row"),
        "optimal_tuples": margin.get("optimal_tuples"),
    }


def _read_holes_csv(path: str | Path) -> list[dict]:
    from .ioformats import read_csv
    holes = []
    for row in read_csv(path):
        hole = {"row": int(row["row"]), "col": int(row["col"])}
        if row.get("height"):
            hole["height"] = int(row["height"])
        elif row.get("allowed_heights"):
            sep = "|" if "|" in row["allowed_heights"] else ","
            hole["allowed_heights"] = [int(x) for x in row["allowed_heights"].split(sep) if x]
        holes.append(hole)
    return holes


def cmd_start(args: argparse.Namespace) -> int:
    edge_doc = load_json(args.edges)
    if args.order:
        order = load_json(args.order)
    else:
        order = {
            "order_id": args.order_id or f"WO-AUTO-{edge_doc['rows']}X{edge_doc['cols']}",
            "rows": int(edge_doc["rows"]),
            "cols": int(edge_doc["cols"]),
            "holes": [],
            "gauge": {},
            "method_policy": "whole",
            "seam_columns": [],
            "contingencies": [],
        }
    if args.order_id:
        order["order_id"] = args.order_id
    if args.holes:
        order["holes"] = order.get("holes", []) + _read_holes_csv(args.holes)
    store = WorkshopStore(args.state_dir)
    report = store.start(edge_doc, order, edge_source=str(args.edges))
    if args.out:
        write_json(args.out, report)
    selected = report.get("selected")
    compact = {
        "order_id": order.get("order_id"),
        "feasible": report["feasible"],
        "state_dir": args.state_dir,
        "out_report": str(args.out) if args.out else None,
    }
    if report["feasible"]:
        compact.update({
            "optimal_score": report["optimal_score"],
            "optimal_height_map": report["optimal_height_map"],
            "selected_target_deltas": selected.get("target_deltas"),
            "selected_method": selected["method"],
            "selected_seam_column": selected["seam_column"],
            "selected_expected_rejection_loss": selected["risk"]["expected_rejection_loss"],
            "first_cut_count": len(selected["cuts"]),
            "first_cuts": selected["cuts"],
            "extra_station_required": selected.get("extra_station_required", False),
            "station_requirement": selected.get("station_requirement"),
            "legal_margin": _margin_compact(selected.get("legal_margin")),
            "selection_reason": report["selection_reason"],
        })
    else:
        compact["reason"] = report.get("solver_reason", "no feasible solution")
    _print_json(compact)
    return 0 if report["feasible"] else 3


def _ticket_from_args(args: argparse.Namespace) -> dict:
    if args.ticket:
        ticket = load_json(args.ticket)
    else:
        ticket = {"ticket_id": args.ticket_id or "CLI-TICKET", "holes": []}
    for spec in args.hole or []:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError("--hole expects row:col:height, e.g. 1:2:2")
        r, c, h = map(int, parts)
        ticket.setdefault("holes", []).append({"row": r, "col": c, "height": h})
    if args.ticket_id:
        ticket["ticket_id"] = args.ticket_id
    return ticket


def cmd_supplement(args: argparse.Namespace) -> int:
    ticket = _ticket_from_args(args)
    store = WorkshopStore(args.state_dir)
    response = store.supplement(args.order_id, ticket, finalize_seam=not args.keep_seam)
    if args.out:
        write_json(args.out, response)
    _print_json(response)
    return 0 if response.get("accepted") else 4


def cmd_cuts(args: argparse.Namespace) -> int:
    store = WorkshopStore(args.state_dir)
    report = store.cuts_report(args.order_id)
    _print_json(report)
    return 0


def cmd_local_check(args: argparse.Namespace) -> int:
    edge_doc = load_json(args.edges)
    gauge = None
    if args.order:
        gauge = load_json(args.order).get("gauge")
    report = local_argmax_report(edge_doc, gauge)
    _print_json(report, args.out)
    return 0 if report["all_elementary_loops_close"] else 6


def cmd_verify(args: argparse.Namespace) -> int:
    edge_doc = load_json(args.edges)
    order = load_json(args.order) if args.order else {"rows": edge_doc["rows"], "cols": edge_doc["cols"]}
    problem, _ = build_problem(edge_doc, order.get("gauge"), order_constraints(order))
    dp = solve_map(problem)
    report: dict[str, Any] = {
        "vertices": problem.n_vertices,
        "dp": {"feasible": dp.feasible, "score": dp.score,
               "heights": list(dp.heights) if dp.heights else None},
    }
    if problem.n_vertices <= args.enumerate_limit:
        bf = brute_force_best(problem)
        report["enumeration"] = {
            "feasible": bf.feasible, "score": bf.score,
            "heights": list(bf.heights) if bf.heights else None,
        }
        if dp.feasible and bf.feasible:
            report["scores_match"] = abs(dp.score - bf.score) < 1e-8
            report["assignments_match_or_both_optimal"] = abs(dp.score - bf.score) < 1e-8
        else:
            report["scores_match"] = dp.feasible == bf.feasible
    else:
        report["enumeration_skipped"] = (
            f"{problem.n_vertices} vertices > --enumerate-limit {args.enumerate_limit}"
        )
    if args.out:
        write_json(args.out, report)
    _print_json(report)
    return 0 if report.get("scores_match", True) else 5


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="relief-workshop", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate small synthetic materials")
    g.add_argument("--out", default="data/generated")
    g.add_argument("--seed", type=int, default=15)
    g.add_argument("--rows", type=int, default=2)
    g.add_argument("--cols", type=int, default=4)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("calibrate", help="learn slope probabilities from labeled four-channel coupons")
    c.add_argument("--train", required=True)
    c.add_argument("--holdout")
    c.add_argument("--model", required=True)
    c.add_argument("--report")
    c.add_argument("--var-floor", type=float, default=0.12)
    c.add_argument("--label-smoothing", type=float, default=0.004)
    c.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("scan-edges", help="predict complete -2..2 probabilities for every directed edge")
    s.add_argument("--scan", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_scan_edges)

    st = sub.add_parser("start", help="sign the first actual cuts after exact MAP planning")
    st.add_argument("--edges", required=True)
    st.add_argument("--order", help="work-order JSON; omitted means no holes/gauge and whole milling")
    st.add_argument("--order-id", help="override or provide the work-order number")
    st.add_argument("--holes", help="optional workshop clamping-hole CSV to merge into the order")
    st.add_argument("--state-dir", default="out/state")
    st.add_argument("--out")
    st.set_defaults(func=cmd_start)

    su = sub.add_parser("supplement", help="evaluate and sign a formal hole supplement")
    su.add_argument("--state-dir", default="out/state")
    su.add_argument("--order-id", required=True, help="parent job id or known ticket id")
    su.add_argument("--ticket")
    su.add_argument("--ticket-id")
    su.add_argument("--hole", action="append")
    su.add_argument("--keep-seam", action="store_true")
    su.add_argument("--out")
    su.set_defaults(func=cmd_supplement)

    cu = sub.add_parser("cuts", help="query actual tool positions by work/ticket order number")
    cu.add_argument("--state-dir", default="out/state")
    cu.add_argument("--order-id", required=True)
    cu.set_defaults(func=cmd_cuts)

    lc = sub.add_parser("local-check", help="show modal edge slopes and loop-closure contradictions")
    lc.add_argument("--edges", required=True)
    lc.add_argument("--order")
    lc.add_argument("--out")
    lc.set_defaults(func=cmd_local_check)

    v = sub.add_parser("verify", help="DP score plus independent small-grid enumeration")
    v.add_argument("--edges", required=True)
    v.add_argument("--order")
    v.add_argument("--enumerate-limit", type=int, default=8)
    v.add_argument("--out")
    v.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
