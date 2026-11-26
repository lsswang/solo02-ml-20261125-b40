#!/usr/bin/env python3
"""Reproducible requirement checks for the coffee-blend delivery.

Run:  python3 verify.py

The script regenerates the small synthetic CSV files, reruns both pipelines,
and checks the numerical/state invariants called out in the puzzle statement.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict

from coffee_blend.extra_demos import preweighed_half_bucket_demo
from coffee_blend.pipeline import run_pipeline
from coffee_blend.shop import bayes_update, ShopService
from coffee_blend.synthetic import generate_votes
from coffee_blend.utils import read_json, write_json

ROOT = os.path.dirname(os.path.abspath(__file__))
GOOD_INPUT = os.path.join(ROOT, "input")
BAD_INPUT = os.path.join(ROOT, "input_bad_b")
GOOD_OUT = os.path.join(ROOT, "deliverables", "good")
BAD_OUT = os.path.join(ROOT, "deliverables", "bad_b")

CHECKS = []


def check(name: str, condition: bool, detail: Any = "") -> None:
    if not condition:
        raise AssertionError(f"FAILED: {name} :: {detail}")
    CHECKS.append(name)
    print(f"PASS  {name}")


def action(results: Dict[str, Any], cmd: str, **match: Any) -> Dict[str, Any]:
    for a in results["actions"]:
        if a.get("cmd") != cmd:
            continue
        if all(a.get("result", {}).get(k) == v for k, v in match.items()):
            return a
    # Fall back to order id matching.
    for a in results["actions"]:
        r = a.get("result") or {}
        if a.get("cmd") == cmd and all(r.get(k) == v for k, v in match.items()):
            return a
    raise KeyError((cmd, match))


def main() -> int:
    # Regenerate the shipped sample inputs so the check is self-contained.
    generate_votes(os.path.join(GOOD_INPUT, "sensory_votes.csv"), "good", 231125)
    generate_votes(os.path.join(BAD_INPUT, "sensory_votes.csv"), "bad_b", 231125)

    good = run_pipeline(GOOD_INPUT, GOOD_OUT)
    bad = run_pipeline(BAD_INPUT, BAD_OUT)
    preweighed_scan = preweighed_half_bucket_demo()

    eval_good = read_json(os.path.join(GOOD_OUT, "blind_evaluation.json"))
    split = eval_good["split"]
    check("whole-roast/formula grouped split has no violations", split["invariants_ok"] is True, split)
    check("all formula judge groups stay in one partition", split["same_formula_split_violations"] == [], split.get("same_formula_split_violations"))
    check("all tasting-round judge groups stay intact", split["same_tasting_event_group_violations"] == [])
    check("requested blind roast was held out", "R-BLIND-11" in split["blind_roasts"], split["blind_roasts"])
    check("re-roast connected component is also blind", "R-BLIND-12" in split["blind_roasts"])
    check("finite blind log loss", eval_good["metrics"]["blind_vote_log_loss"] > 0)
    check("model beats constant baseline on sample", eval_good["metrics"]["log_loss_improvement_vs_baseline"] > 0)

    rec = read_json(os.path.join(GOOD_OUT, "production_recommendations.json"))
    passing = rec["feasible_passing_recipes"]
    check("exactly one feasible passing recipe", len(passing) == 1, passing)
    ab = passing[0]
    check("passing recipe is JIA2+YI2", ab["formula_id"] == "JIA-2409:2|YI-2409:2", ab)
    check("half-half probability >= .80", ab["predicted_probability"] >= 0.80, ab["predicted_probability"])
    check("stock supports exactly two cups", ab["max_cups_from_loose_stock"] == 2)

    model = read_json(os.path.join(GOOD_OUT, "model.json"))
    coeff = model["coefficient_map"]
    check("positive JIA*YI synergy coefficient", coeff["pair::JIA-2409*YI-2409"] > 5, coeff)

    main_run = good["shop_runs"]["01_bucket_three_orders"]
    decisions = {a["result"].get("order_id"): a["result"].get("decision") for a in main_run["actions"] if a["cmd"] == "order"}
    check("O1 opens bucket", decisions.get("O1") == "accepted_and_bucket_opened", decisions)
    check("O2 accepted from same bucket", decisions.get("O2") == "accepted", decisions)
    check("O3 reports material shortage", decisions.get("O3") == "rejected_material", decisions)
    stock = read_json(os.path.join(GOOD_OUT, "shops/01_bucket_three_orders/stock_final.json"))
    check("A and B exhausted after two cups", stock["loose_stock_parts"]["JIA-2409"] == 0 and stock["loose_stock_parts"]["YI-2409"] == 0, stock)

    scan_o1 = read_json(os.path.join(GOOD_OUT, "shops/01_bucket_three_orders/scan_O1_half_bucket_refills_O2.json"))
    rb = scan_o1["remaining_buckets_and_refills"][0]
    check("scan identifies remaining half bucket", rb["is_half_bucket"] and rb["remaining_cups"] == 1, rb)
    check("half bucket maps to O2 refill", rb["can_refill_orders"] and rb["can_refill_orders"][0]["order_id"] == "O2", rb)
    check("scan shows actual successful weighing", len(scan_o1["actual_weighings"]) == 1 and scan_o1["actual_weighings"][0]["total_grams"] == 120.0)
    check("top-level weighing example exists", os.path.exists(os.path.join(GOOD_OUT, "actual_successful_weighing_example.json")))
    pre = preweighed_scan["remaining_buckets_and_refills"][0]
    check("input weighed half-bucket is parsed and matched to an order", pre["is_half_bucket"] and pre["can_refill_orders"], pre)

    wd_run = good["shop_runs"]["02_three_jia_withdrawal_invalidates_two_cups"]
    wd_order = [a for a in wd_run["actions"] if a["cmd"] == "order"][0]["result"]
    check("removing three JIA leaves one part", wd_run["stock_final"]["loose_stock_parts"]["JIA-2409"] == 1)
    check("two-cup commitment fails after JIA withdrawal", wd_order["decision"] == "rejected_material", wd_order)

    broken_stock = read_json(os.path.join(GOOD_OUT, "shops/02b_two_commitments_then_three_jia_stocktake/stock_commitments_broken.json"))
    check(
        "two previously accepted commitments are preserved but flagged unfulfillable",
        broken_stock["orders"].get("P1", {}).get("status") == "unfulfillable_after_stocktake"
        and broken_stock["orders"].get("P2", {}).get("status") == "unfulfillable_after_stocktake"
        and broken_stock["loose_stock_parts"]["JIA-2409"] == 1,
        broken_stock,
    )

    like = bayes_update(0.5, "like", 0.9)
    unlike = bayes_update(0.5, "unlike", 0.9)
    check("like reading gives 0.9 posterior", abs(like["posterior"] - 0.9) < 1e-12, like)
    check("unlike reading gives 0.1 posterior", abs(unlike["posterior"] - 0.1) < 1e-12, unlike)

    like_run = good["shop_runs"]["03_trial_same_order_like"]
    unlike_run = good["shop_runs"]["04_trial_same_order_unlike"]
    like_trial = [a for a in like_run["actions"] if a["cmd"] == "trial"][0]["result"]
    unlike_trial = [a for a in unlike_run["actions"] if a["cmd"] == "trial"][0]["result"]
    check("like branch accepts", like_trial["decision"] == "accepted" and abs(like_trial["posterior"] - 0.9) < 1e-12)
    check("unlike branch rejects", unlike_trial["decision"] == "rejected_quality" and abs(unlike_trial["posterior"] - 0.1) < 1e-12)
    like_stock = read_json(os.path.join(GOOD_OUT, "shops/03_trial_same_order_like/stock_like.json"))
    unlike_stock = read_json(os.path.join(GOOD_OUT, "shops/04_trial_same_order_unlike/stock_unlike.json"))
    check("like branch consumes trial plus sold cup", like_stock["loose_stock_parts"]["JIA-2409"] == 0 and like_stock["loose_stock_parts"]["YI-2409"] == 0)
    check("unlike branch keeps remaining beans but trial dose is consumed", unlike_stock["loose_stock_parts"]["JIA-2409"] == 2 and unlike_stock["loose_stock_parts"]["YI-2409"] == 2)

    cal_run = good["shop_runs"]["05_reliably_calibrated_whole_bucket_signoff"]
    cal_order = [a for a in cal_run["actions"] if a["cmd"] == "order"][0]["result"]
    check("calibrated p=.92 directly signs bucket", cal_order["decision"] == "accepted_and_bucket_opened" and abs(cal_order["probability"] - 0.92) < 1e-12)

    bad_rec = read_json(os.path.join(BAD_OUT, "production_recommendations.json"))
    bad_ab = [r for r in bad_rec["all_recipes"] if r["formula_id"] == "JIA-2409:2|YI-2418:2"][0]
    check("new B half-half is below gate", bad_ab["predicted_probability"] < 0.8, bad_ab)
    check("bad input has no passing feasible recipe", bad_rec["feasible_passing_recipes"] == [])
    bad_order = [a for a in bad["shop_runs"]["bad_b_first_order_must_change_or_refuse"]["actions"] if a["cmd"] == "order"][0]["result"]
    check("bad-B first order refused on quality", bad_order["decision"] == "rejected_quality", bad_order)
    bad_stock = read_json(os.path.join(BAD_OUT, "shops/bad_b_first_order_must_change_or_refuse/stock_unchanged_after_refusal.json"))
    check("refusal leaves quantities unchanged", bad_stock["loose_stock_parts"]["YI-2418"] == 4, bad_stock)

    # Reopen the event ledger and ensure duplicate serving cannot rewrite usage.
    state_dir = os.path.join(GOOD_OUT, "shops/01_bucket_three_orders")
    shop = ShopService(state_dir)
    ledger_lines_before = len(open(os.path.join(state_dir, "ledger.jsonl"), encoding="utf-8").readlines())
    duplicate_blocked = False
    try:
        shop.serve_order("O1")
    except Exception:
        duplicate_blocked = True
    ledger_lines_after = len(open(os.path.join(state_dir, "ledger.jsonl"), encoding="utf-8").readlines())
    check("duplicate serve is blocked and appends no event", duplicate_blocked and ledger_lines_before == ledger_lines_after)

    # Same-format input substitution in a clean directory.
    with tempfile.TemporaryDirectory(prefix="coffee_swap_") as tmp:
        in_dir = os.path.join(tmp, "input")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir)
        generate_votes(os.path.join(in_dir, "sensory_votes.csv"), "good", 424242)
        write_json(os.path.join(in_dir, "inventory.json"), {
            "batches": [
                {"batch_id": "JIA-2409", "bean_name": "甲"},
                {"batch_id": "YI-2409", "bean_name": "乙"},
                {"batch_id": "BING-2409", "bean_name": "丙"},
            ],
            "loose_stock_parts": {"JIA-2409": 4, "YI-2409": 4, "BING-2409": 4},
            "empty_buckets": [{"bucket_id": "KEG-1", "capacity_cups": 2}, {"bucket_id": "KEG-2", "capacity_cups": 2}],
            "weighed_buckets": [],
            "roast_states": [],
        })
        swapped = run_pipeline(in_dir, out_dir)
        check("swapped same-format input produces blind report", swapped["split_invariants_ok"] is True)
        check("swapped input produces model deliverable", os.path.exists(os.path.join(out_dir, "model.json")))

    print(f"\nALL {len(CHECKS)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
