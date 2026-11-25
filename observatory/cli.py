from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .calibration import calibrate as run_calibration
from .calibration import write_json
from .querying import load_json, run_query_file
from .reservations import (
    book_reservation,
    load_reservation,
    record_first,
    record_second,
    resolve_reservation_path,
    save_reservation,
)
from .scheduling import (
    Application,
    filter_ephemerides,
    load_shifts,
    plan_fixed,
    plan_voucher,
)
from .synthetic import generate_sample_tree


def _print(title: str, value: Any) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _cmd_generate(args: argparse.Namespace) -> int:
    root = generate_sample_tree(args.out)
    print(f"generated reproducible synthetic input tree in {root.resolve()}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibration import CalibrationInput

    cal_input = CalibrationInput.load(args.input)
    result = run_calibration(cal_input, z=args.z)
    out = write_json(result, args.out)
    print(f"calibration fit written to {out}")
    print("central responses:")
    for star, resp in result["responses"].items():
        print(f"  {star}: ap1={resp['ap1']:.4f}, ap2={resp['ap2']:.4f}")
    print(f"95% response columns overlap: {result['columns_overlap_within_95']}")
    print(f"whole-level holdout reports: {len(result['holdout_validation'])} brightness levels")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    result = run_query_file(args.calibration, args.observation, args.out, args.uncertainty)
    if args.out:
        print(f"query result written to {Path(args.out).resolve()}")
        print(f"supported stars: {result['supported_stars']}")
        for star in result["supported_stars"]:
            print(f"  {star} depth interval: {result['stars'][star]['depth_interval_text']}")
    else:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        print()
    return 0


def _load_planning_inputs(args: argparse.Namespace):
    app = Application.load(args.application)
    shifts = load_shifts(args.shifts, app.horizon)
    query = load_json(args.query) if getattr(args, "query", None) else None
    if query is not None:
        supported = query.get("supported_stars", [])
        filter_source = f"query {args.query}"
    elif getattr(args, "stars", None):
        supported = [s.strip() for s in args.stars.split(",") if s.strip()]
        query = {"supported_stars": supported}
        filter_source = "--stars"
    else:
        supported = None
        filter_source = "no star filter"
    candidates, excluded = filter_ephemerides(app.ephemerides, supported, app.known_clear_nights)
    return app, shifts, query, supported, filter_source, candidates, excluded


def _planning_document(args: argparse.Namespace):
    app, shifts, query, supported, filter_source, candidates, excluded = _load_planning_inputs(args)
    doc: dict[str, Any] = {
        "schema": "observatory-plan-v1",
        "order_id": app.order_id,
        "horizon_nights": app.horizon,
        "star_filter_source": filter_source,
        "supported_stars": supported,
        "known_clear_nights": sorted(app.known_clear_nights),
        "candidate_count": len(candidates),
        "candidate_ephemerides": [e.to_dict() for e in candidates],
        "excluded": excluded,
        "shift_table": [s.to_dict() for s in sorted(shifts.values(), key=lambda s: s.night)],
    }
    method = args.method
    allow_skip = bool(getattr(args, "allow_skip_redemption", False)) or bool(
        app.raw.get("allow_skip_redemption", False)
    )
    if method in {"fixed", "both"}:
        doc["fixed_buyout"] = plan_fixed(candidates, shifts)
    if method in {"voucher", "both"}:
        doc["flexible_voucher"] = plan_voucher(candidates, shifts, app.voucher_fee, allow_skip)
    return app, shifts, query, candidates, excluded, doc


def _cmd_plan(args: argparse.Namespace) -> int:
    _, _, _, _, _, doc = _planning_document(args)
    if args.out:
        write_json(doc, args.out)
        print(f"plan written to {Path(args.out).resolve()}")
    print(f"compatible candidates after photometry/clear records: {doc['candidate_count']}")
    if "fixed_buyout" in doc and doc["fixed_buyout"].get("feasible"):
        p = doc["fixed_buyout"]
        print(f"fixed buyout best: nights={p['nights']} worst_remaining={p['worst_remaining']} worst_cost={p['worst_cost']}")
    if "flexible_voucher" in doc and doc["flexible_voucher"].get("feasible"):
        p = doc["flexible_voucher"]
        print(
            "voucher best: "
            f"first_night={p['first_night']} listed={p['voucher_listed_nights']} "
            f"worst_remaining={p['worst_remaining']} worst_cost={p['worst_cost']:.3g}"
        )
    return 0


def _cmd_book(args: argparse.Namespace) -> int:
    app, shifts, query, candidates, excluded, doc = _planning_document(
        argparse.Namespace(**{**vars(args), "method": "voucher"})
    )
    plan = doc["flexible_voucher"]
    if not plan.get("feasible"):
        raise RuntimeError(plan.get("reason", "voucher planning is infeasible"))

    reservation_id = args.reservation_id or f"RSV-{app.order_id}"
    voucher_id = args.voucher_id or f"VCH-{app.order_id}"
    query_summary = None
    if query:
        supported = query.get("supported_stars", [])
        query_summary = {
            "calibration_file": query.get("calibration_file"),
            "observation_file": query.get("observation_file"),
            "observation_id": query.get("observation_id"),
            "supported_stars": supported,
            "drops": query.get("drops"),
            "error_bounds": query.get("error_bounds"),
            "response_uncertainty": query.get("response_uncertainty"),
            "depth_intervals": {
                star: query.get("stars", {}).get(star, {}).get("depth_interval")
                for star in supported
            },
        }
    reservation, path = book_reservation(
        order_id=app.order_id,
        reservation_id=reservation_id,
        voucher_id=voucher_id,
        plan=plan,
        compatible_ephemerides=candidates,
        excluded=excluded,
        shifts_snapshot=doc["shift_table"],
        query_summary=query_summary,
        registry_path=args.registry,
        state_dir=args.state_dir,
        force=args.force,
    )
    if args.out:
        write_json(reservation, args.out)
    print(f"reservation number: {reservation_id}")
    print(f"voucher number: {voucher_id}")
    print(f"reservation file: {path.resolve()}")
    print(
        f"booked basis: first night {plan['first_night']}; "
        f"voucher may list nights {plan['voucher_listed_nights']}; "
        f"worst remaining={plan['worst_remaining']}, worst cost={plan['worst_cost']:.3g}"
    )
    print("successor command:")
    print(
        f"  python -m observatory fill-first --state-dir {args.state_dir} "
        f"--reservation-id {reservation_id} --result dim|bright"
    )
    return 0


def _cmd_fill_first(args: argparse.Namespace) -> int:
    path = resolve_reservation_path(
        args.reservation, args.reservation_id, args.state_dir, args.registry
    )
    reservation = load_reservation(path)
    record_first(reservation, args.result)
    save_reservation(reservation, path)
    action = reservation["action_for_successor"]
    print(f"reservation {reservation['reservation_id']}: first result bit={reservation['first_result']}")
    print(f"actual voucher night to watch: {action['watch_night']}")
    print("still-compatible ephemerides:")
    for e in action["still_compatible_ephemerides"]:
        print(f"  {e['id']} star={e['star']} pattern={e['pattern']} transit={e['transit_time']}")
    if args.out:
        write_json(reservation, args.out)
    return 0


def _cmd_fill_second(args: argparse.Namespace) -> int:
    path = resolve_reservation_path(
        args.reservation, args.reservation_id, args.state_dir, args.registry
    )
    reservation = load_reservation(path)
    record_second(reservation, args.result)
    save_reservation(reservation, path)
    print(f"reservation {reservation['reservation_id']}: second result bit={reservation['second_result']}")
    print(f"final compatible ephemeris ids: {reservation['final_compatible_ids']}")
    for e in reservation.get("final_compatible_ephemerides", []):
        print(f"  {e['id']} star={e['star']} pattern={e['pattern']} transit={e['transit_time']}")
    if args.out:
        write_json(reservation, args.out)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    path = resolve_reservation_path(
        args.reservation, args.reservation_id, args.state_dir, args.registry
    )
    reservation = load_reservation(path)
    summary = {
        "reservation_id": reservation["reservation_id"],
        "order_id": reservation["order_id"],
        "voucher_id": reservation["voucher_id"],
        "status": reservation["status"],
        "first_result": reservation.get("first_result"),
        "actual_redeem_night": reservation.get("actual_redeem_night"),
        "compatible_after_first_ids": [
            e["id"] for e in reservation.get("compatible_after_first", [])
        ],
        "second_result": reservation.get("second_result"),
        "final_compatible_ids": reservation.get("final_compatible_ids"),
        "booked_worst_remaining": reservation["plan"].get("worst_remaining"),
        "voucher_listed_nights": reservation["plan"].get("voucher_listed_nights"),
    }
    _print("reservation", summary)
    if reservation.get("actual_redeem_night") is not None and reservation["status"] == "first_recorded":
        print(f"Next action: redeem voucher for night {reservation['actual_redeem_night']}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="observatory",
        description="Offline two-star photometry, point queries, and minimax voucher scheduling.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate-sample", help="generate small reproducible inputs")
    p_gen.add_argument("--out", default="sample_inputs")
    p_gen.set_defaults(func=_cmd_generate)

    p_cal = sub.add_parser("calibrate", help="fit non-negative two-aperture/two-star responses")
    p_cal.add_argument("--input", required=True, help="calibration directory with backgrounds.csv and calibration.csv")
    p_cal.add_argument("--out", required=True)
    p_cal.add_argument("--z", type=float, default=1.96)
    p_cal.set_defaults(func=_cmd_calibrate)

    p_q = sub.add_parser("query", help="point-query which single star dimming explains a drop")
    p_q.add_argument("--calibration", required=True)
    p_q.add_argument("--observation", required=True)
    p_q.add_argument("--out")
    p_q.add_argument(
        "--uncertainty",
        choices=["none", "se95", "validation"],
        help="override observation file response uncertainty mode",
    )
    p_q.set_defaults(func=_cmd_query)

    def add_planning_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--application", required=True)
        p.add_argument("--shifts", required=True)
        src = p.add_mutually_exclusive_group()
        src.add_argument("--query", help="query result JSON; supported_stars filter ephemeris star names")
        src.add_argument("--stars", help="comma-separated supported star names when no query JSON is supplied")
        p.add_argument(
            "--allow-skip-redemption",
            action="store_true",
            help="allow cost-optimal no-redemption when no voucher night can improve the worst count",
        )

    p_plan = sub.add_parser("plan", help="read-only minimax plan")
    add_planning_args(p_plan)
    p_plan.add_argument("--method", choices=["fixed", "voucher", "both"], default="both")
    p_plan.add_argument("--out")
    p_plan.set_defaults(func=_cmd_plan)

    p_book = sub.add_parser("book", help="book the selected flexible-voucher method and issue a reservation number")
    add_planning_args(p_book)
    p_book.add_argument("--reservation-id")
    p_book.add_argument("--voucher-id")
    p_book.add_argument("--state-dir", default="out/state")
    p_book.add_argument("--registry", default="out/state/voucher_registry.json")
    p_book.add_argument("--out")
    p_book.add_argument("--force", action="store_true", help="replace a local reservation id")
    p_book.set_defaults(func=_cmd_book)

    p_first = sub.add_parser("fill-first", help="backfill the first night's binary result")
    p_first.add_argument("--reservation")
    p_first.add_argument("--reservation-id")
    p_first.add_argument("--state-dir", default="out/state")
    p_first.add_argument("--registry", default="out/state/voucher_registry.json")
    p_first.add_argument("--result", required=True, help="dim/1 or bright/0")
    p_first.add_argument("--out")
    p_first.set_defaults(func=_cmd_fill_first)

    p_second = sub.add_parser("fill-second", help="backfill the redeemed night's binary result")
    p_second.add_argument("--reservation")
    p_second.add_argument("--reservation-id")
    p_second.add_argument("--state-dir", default="out/state")
    p_second.add_argument("--registry", default="out/state/voucher_registry.json")
    p_second.add_argument("--result", required=True)
    p_second.add_argument("--out")
    p_second.set_defaults(func=_cmd_fill_second)

    p_show = sub.add_parser("show", help="show reservation status")
    p_show.add_argument("--reservation")
    p_show.add_argument("--reservation-id")
    p_show.add_argument("--state-dir", default="out/state")
    p_show.add_argument("--registry", default="out/state/voucher_registry.json")
    p_show.set_defaults(func=_cmd_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
