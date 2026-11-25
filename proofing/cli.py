"""Command-line interface for the proofing program.

Sub-commands
------------
``gen-data``   write synthetic ramp CSVs + hidden truth into a directory
``gen-jobs``   write the example job JSON files into a directory
``fit``        fit responses from ramp CSVs and write ``response.json``
``plan``       choose a recipe (route 1, direct output) and write a plan
``show``       re-print / diagnose a saved plan and evaluation
``recheck``    feed back an actual press offset / reading and re-check the sheet

Everything uses only the standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

from . import jobs as jobs_mod
from . import synthetic
from .jobs import load_job
from .model import CHANNELS, Response, fit_all
from .planner import plan_direct


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read_ramps(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for k in ("coverage", "cu", "cv"):
                if k in row:
                    row[k] = float(row[k])
            for k in ("d0", "d1", "d2"):
                if k in row:
                    row[k] = float(row[k])
            rows.append(row)
    return rows


def _load_responses(data_dir: str) -> Dict[str, Response]:
    single = _read_ramps(os.path.join(data_dir, "ramps_single.csv"))
    dual = _read_ramps(os.path.join(data_dir, "ramps_dual.csv"))
    responses, _reports = fit_all(single, dual)
    return responses


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# global chromatic check: intermediate ramps must match in every channel
# --------------------------------------------------------------------------- #
def _check_chromatic(resp: Response, tol: float = 0.02) -> dict:
    """The wide-block target is achromatic (all channels equal).  Verify the
    learned response really produces near-equal channels across intermediate
    coverage and both orders (within ``tol``)."""
    worst = {"gap": 0.0, "order": None, "cu": None, "cv": None}
    n = 0
    for order in ("BA", "AB"):
        for cu in (0.2, 0.4, 0.6, 0.8, 1.0):
            for cv in (0.2, 0.4, 0.6, 0.8, 1.0):
                d = [resp.dual_density(order, cu, cv, ch) for ch in CHANNELS]
                gap = max(d) - min(d)
                n += 1
                if gap > worst["gap"]:
                    worst = {"gap": gap, "order": order, "cu": cu, "cv": cv}
    return {"achromatic": worst["gap"] <= tol, "max_gap": worst["gap"],
            "tol": tol, "worst": worst, "points": n}


# --------------------------------------------------------------------------- #
# sub-commands
# --------------------------------------------------------------------------- #
def cmd_gen_data(args) -> int:
    if args.spec:
        with open(args.spec, encoding="utf-8") as fh:
            spec = json.load(fh)
        truth = synthetic.build_truth_from_spec(spec)
    else:
        truth = synthetic.build_truth()
    paths = synthetic.write_dataset(args.outdir, truth)
    print("wrote:")
    for k, v in paths.items():
        print(f"  {k:6s} {v}")
    return 0


def cmd_gen_jobs(args) -> int:
    paths = jobs_mod.write_example_jobs(args.outdir)
    print("wrote jobs:")
    for p in paths:
        print(f"  {p}")
    return 0


def cmd_fit(args) -> int:
    single = _read_ramps(os.path.join(args.data, "ramps_single.csv"))
    dual = _read_ramps(os.path.join(args.data, "ramps_dual.csv"))
    responses, reports = fit_all(single, dual)
    out = {"responses": {p: r.to_dict() for p, r in responses.items()},
           "holdout": reports,
           "chromatic": {p: _check_chromatic(r) for p, r in responses.items()}}
    _save_json(args.out, out)
    print(f"fitted {len(responses)} paper(s) -> {args.out}")
    for p, rep in reports.items():
        s, d = rep["single"], rep["dual"]
        ch = out["chromatic"][p]
        print(f"  {p}: single holdout max|err|={s['max_abs']:.4f}, "
              f"dual holdout max|err|={d['max_abs']:.4f}, achromatic={ch['achromatic']}")
    return 0


def cmd_plan(args) -> int:
    responses = _load_responses(args.data)
    job = load_job(args.job)
    if job.paper not in responses:
        print(f"error: no learned response for paper {job.paper!r}", file=sys.stderr)
        return 2
    resp = responses[job.paper]
    plan = plan_direct(job, resp)
    result = {"job": job.name, "paper": job.paper, "route": plan.route,
              "plan": plan.to_dict()}
    _save_json(args.out, result)

    if plan.feasible:
        print(f"job={job.name} route={plan.route} recipe={plan.recipe} "
              f"cost={plan.cost:.1f} feasible=True")
    else:
        print(f"job={job.name} route={plan.route} feasible=False "
              f"(would be scrapped)")
        ev = plan.evidence
        print("  reason:", ev.get("reason"))
        # show one concrete failing offset per considered recipe
        for rn, fails in ev.get("failures", {}).items():
            if fails:
                f0 = fails[0]
                print(f"  {rn}: first failing offset {f0['offset']} -> {f0['kind']}"
                      + (f" at cell {f0.get('worst_cell')}" if f0.get("kind") == "out_of_tol" else ""))
    return 0


def cmd_show(args) -> int:
    with open(args.plan, encoding="utf-8") as fh:
        result = json.load(fh)
    plan = result["plan"]
    print(f"job={result['job']} paper={result['paper']} route={plan['route']}")
    print(f"recipe={plan['recipe']} cost={plan['cost']} feasible={plan['feasible']}")
    for name, chk in plan.get("checks", {}).items():
        print(f"  check {name}: passes={chk['passes']} scrap={chk['scrap']} "
              f"worst_relief={chk['worst_relief']:.5f} "
              f"worst_cell={chk['worst_cell']} worst_ch={chk['worst_channel']}")
    return 0


def cmd_report(args) -> int:
    """Print a per-offset compliance table for one recipe (verification aid)."""
    from .render import evaluate
    responses = _load_responses(args.data)
    job = load_job(args.job)
    resp = responses[job.paper]
    ev = evaluate(job, resp, args.recipe)
    print(f"job={job.name} paper={job.paper} recipe={args.recipe}")
    print(f"  passes={ev.passes} scrap={ev.scrap} equal_channel_gap="
          f"{ev.equal_channel_gap:.4f} (equal_channels={job.equal_channels})")
    for off in ev.offsets:
        if off.scrap_cells:
            tag = f"SCRAP ({len(off.scrap_cells)} cells >=3 layers)"
        elif off.worst_relief <= 1e-12:
            tag = "PASS"
        else:
            tag = f"FAIL relief={off.worst_relief:+.4f}"
        print(f"  offset={off.offset!s:28s} {tag:24s} "
              f"worst_cell={off.worst_cell} worst_ch={off.worst_channel}")
    return 0 if ev.passes else 1


def cmd_handoff(args) -> int:
    """Hand the selected paper's learned response to the platemaker.

    Prints the fitted parameters, the full-coverage recipe densities, and the
    held-out *intermediate coverage* validation (the part of the ramp that was
    never used for fitting)."""
    from .model import ORDERS
    single = _read_ramps(os.path.join(args.data, "ramps_single.csv"))
    dual = _read_ramps(os.path.join(args.data, "ramps_dual.csv"))
    responses, reports = fit_all(single, dual)
    if args.paper not in responses:
        print(f"error: unknown paper {args.paper!r}; have {sorted(responses)}",
              file=sys.stderr)
        return 2
    resp = responses[args.paper]
    rep = reports[args.paper]

    print(f"# Response handoff for paper {args.paper}")
    print("\n## Full-coverage densities (per channel)")
    for name, dens in (("forward (BA, A presses B)", resp.full_dual("BA")),
                       ("reverse (AB, B presses A)", resp.full_dual("AB")),
                       ("spot (S)", resp.full_single("S"))):
        print(f"  {name:28s} " + "  ".join(f"{d:.4f}" for d in dens))

    print("\n## Learned parameters")
    for ink in ("A", "B", "S"):
        gains = " ".join(f"{resp.g[ink][ch]:.4f}" for ch in CHANNELS)
        solids = " ".join(f"{resp.s[ink][ch]:.4f}" for ch in CHANNELS)
        print(f"  {ink}: gain=[{gains}]  solid=[{solids}]")
    for order in ORDERS:
        ks = " ".join(f"{resp.k[order][ch]:.4f}" for ch in CHANNELS)
        print(f"  k[{order}]=[{ks}]")

    print("\n## Held-out intermediate-coverage validation")
    print(f"  single ramps: n={rep['single']['n']}  rmse={rep['single']['rmse']:.5f}  "
          f"max|err|={rep['single']['max_abs']:.5f}")
    print(f"  dual ramps  : n={rep['dual']['n']}  rmse={rep['dual']['rmse']:.5f}  "
          f"max|err|={rep['dual']['max_abs']:.5f}")

    print("\n## Dual-ink intermediate sweep (forward, cu=cv, channel 0)")
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        print(f"  coverage {c:.2f} -> D={resp.dual_density('BA', c, c, 0):.4f}")
    if args.json:
        out = {"paper": args.paper,
               "forward": resp.full_dual("BA"),
               "reverse": resp.full_dual("AB"),
               "spot": resp.full_single("S"),
               "params": resp.to_dict(),
               "holdout": rep}
        _save_json(args.json, out)
        print(f"\nwrote {args.json}")
    return 0


def cmd_recheck(args) -> int:
    """After the plan is saved the pressman enters the *actual* offset and
    the program re-checks the worst pixel and total cost."""
    from .render import render_sheet
    responses = _load_responses(args.data)
    job = load_job(args.job)
    resp = responses[job.paper]
    with open(args.plan, encoding="utf-8") as fh:
        result = json.load(fh)
    plan = result["plan"]
    if not plan["feasible"]:
        print("plan is infeasible; nothing to recheck")
        return 0
    recipe = job.recipe(plan["recipe"])
    offset = json.loads(args.offset)
    full = {ink: int(offset.get(ink, 0)) for ink in recipe.inks}
    res = render_sheet(job, resp, recipe, full)
    ok = (not res.scrap_cells) and res.worst_relief <= 1e-12
    print(f"recheck recipe={plan['recipe']} offset={full}")
    print(f"  scrap_cells={res.scrap_cells}")
    print(f"  worst_relief={res.worst_relief:.5f} worst_cell={res.worst_cell} "
          f"worst_ch={res.worst_channel}")
    print(f"  sheet_ok={ok} total_cost={plan['cost']}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="proofing", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-data", help="write synthetic ramp data")
    g.add_argument("--outdir", default="data")
    g.add_argument("--spec", default=None,
                   help="optional JSON paper spec (same format as PAPER_SPEC)")
    g.set_defaults(func=cmd_gen_data)

    g = sub.add_parser("gen-jobs", help="write example job specs")
    g.add_argument("--outdir", default="jobs")
    g.set_defaults(func=cmd_gen_jobs)

    f = sub.add_parser("fit", help="fit responses from ramps")
    f.add_argument("--data", default="data")
    f.add_argument("--out", default="artifacts/response.json")
    f.set_defaults(func=cmd_fit)

    pl = sub.add_parser("plan", help="choose a recipe (route 1)")
    pl.add_argument("--data", default="data")
    pl.add_argument("--job", required=True)
    pl.add_argument("--out", required=True)
    pl.set_defaults(func=cmd_plan)

    sh = sub.add_parser("show", help="show a saved plan")
    sh.add_argument("--plan", required=True)
    sh.set_defaults(func=cmd_show)

    rp = sub.add_parser("report", help="per-offset table for one recipe")
    rp.add_argument("--data", default="data")
    rp.add_argument("--job", required=True)
    rp.add_argument("--recipe", required=True)
    rp.set_defaults(func=cmd_report)

    ho = sub.add_parser("handoff", help="hand a paper's learned response to the platemaker")
    ho.add_argument("--data", default="data")
    ho.add_argument("--paper", required=True)
    ho.add_argument("--json", default=None)
    ho.set_defaults(func=cmd_handoff)

    rc = sub.add_parser("recheck", help="recheck with actual offset")
    rc.add_argument("--data", default="data")
    rc.add_argument("--job", required=True)
    rc.add_argument("--plan", required=True)
    rc.add_argument("--offset", required=True, help='JSON, e.g. {"A":0,"B":1}')
    rc.set_defaults(func=cmd_recheck)
    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
