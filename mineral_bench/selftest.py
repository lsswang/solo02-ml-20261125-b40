"""Standard-library assertions for the required Bayesian and booth behavior."""

from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Tuple

from .core import (
    alive_mask_from_evidence,
    load_json,
    posterior_from_mask,
    validate_and_build,
)
from .generate import (
    canonical_tray,
    half_b_transmits,
    minimal_two_sides,
    perfect_transmission,
    upside_down_sealed,
)
from .learning import learn_orientations
from .pipeline import run_pipeline
from .planner import WorkbenchPlanner
from .workbench import checkout, create_case, flip, observe, status


def _model(doc: Dict[str, Any]):
    weights, _ = learn_orientations(doc)
    return validate_and_build(doc, weights), weights


def _posterior_for(doc: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Fraction]:
    model, _ = _model(doc)
    mask = alive_mask_from_evidence(model, evidence)
    post, _, total = posterior_from_mask(model, mask)
    assert total > 0
    return post


def assert_equal_fraction(actual: Fraction, expected: Fraction, label: str) -> None:
    assert actual == expected, f"{label}: expected {expected}, got {actual}"


def test_textbook_posteriors() -> None:
    # 1. Perfect transmission: either bright or dark signs out a material.
    post = _posterior_for(perfect_transmission(), {"trans": 1})
    assert_equal_fraction(post["A"], Fraction(1), "perfect bright A")
    assert_equal_fraction(post["B"], Fraction(0), "perfect bright B")
    post = _posterior_for(perfect_transmission(), {"trans": 0})
    assert_equal_fraction(post["B"], Fraction(1), "perfect dark B")

    # 2. Half of B orientations also transmit: bright gives A = 2/3.
    post = _posterior_for(half_b_transmits(), {"trans": 1})
    assert_equal_fraction(post["A"], Fraction(2, 3), "half-B bright A")
    assert post["A"] < Fraction(9, 10)
    post = _posterior_for(half_b_transmits(), {"trans": 0})
    assert_equal_fraction(post["B"], Fraction(1), "half-B dark B")

    # 3. Sealed transmission and one identical remaining reading: still 1/2.
    post = _posterior_for(upside_down_sealed(), {"rim": 1})
    assert_equal_fraction(post["A"], Fraction(1, 2), "sealed tray A")
    assert_equal_fraction(post["B"], Fraction(1, 2), "sealed tray B")


def test_minimal_optimal_rate() -> None:
    model, _ = _model(minimal_two_sides())
    planner = WorkbenchPlanner(model)

    row0 = planner.plan_summary(budget=0)
    row1 = planner.plan_summary(budget=1)
    row2 = planner.plan_summary(budget=2)
    row3 = planner.plan_summary(budget=3)

    assert row0["best_recognition_probability"] == 0.0
    assert row1["best_recognition_probability"] == 0.25
    assert row2["best_recognition_probability"] == 0.25
    assert row3["best_recognition_probability"] == 1.0
    assert row3["minimum_expected_cost_at_best_rate"] == 2.5

    # At equal 0.25 recognition, front-first costs 1 expected; back-first
    # pays the second-instrument setup fee and costs 2.
    curve = {r["budget"]: r for r in planner.budget_curve(max_budget=4)}
    assert curve[2]["expected_cost"] == 1.0
    assert curve[3]["expected_cost"] == 2.5


def test_persistent_end_to_end_and_exit_status() -> None:
    doc = minimal_two_sides()
    _, weights = _model(doc)
    with tempfile.TemporaryDirectory(prefix="mineral-selftest-") as td:
        create_case(td, "EXIT-7", doc, weights)

        initial = status(td, "EXIT-7")
        assert initial["cost"]["spent"] == 0
        assert initial["judgment"]["state"] == "undecided"
        assert len(initial["actions"]["available_now"]) == 1

        # Cannot check out before evidence.
        try:
            checkout(td, "EXIT-7")
        except RuntimeError:
            pass
        else:
            raise AssertionError("early checkout must be refused")

        observe(td, "EXIT-7", "front_signal", "bright")
        st = status(td, "EXIT-7")
        assert abs(st["material_posterior"]["A"] - 2 / 3) < 1e-9
        assert st["judgment"]["state"] == "undecided"

        # Exiting and constructing a fresh status call is equivalent to coming
        # back by slice ID; the JSON file is the persistent audience receipt.
        reloaded = status(td, "EXIT-7")
        assert reloaded["evidence"] == {"front_signal": 1}
        assert reloaded["cost"]["spent"] == 1.0

        flip(td, "EXIT-7")
        # Front-only action is no longer physically available.
        try:
            observe(td, "EXIT-7", "front_signal", 1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("front action should be invalid after flip")

        observe(td, "EXIT-7", "back_signal", "bright")
        st = checkout(td, "EXIT-7")
        final = status(td, "EXIT-7")
        assert final["status"] == "signed_out"
        assert final["judgment"]["material"] == "A"
        assert final["cost"]["spent"] == 3.0
        assert final["cost"]["remaining"] == 0.0
        assert Path(td, "cases", "EXIT-7.json").exists()


def test_pipeline_outputs_for_another_input(tmp: Path) -> None:
    # The canonical generated tray stands in for a replacement same-format
    # museum input supplied by a verifier.
    input_path = tmp / "another_tray.json"
    from .core import dump_json

    dump_json(input_path, canonical_tray())
    out = tmp / "deliverables"
    result = run_pipeline(input_path, out)
    assert (out / "learned_model.json").exists()
    assert (out / "validation_report.json").exists()
    assert (out / "workbench_policy.json").exists()
    assert result["validation"]["checked_readings"] > 0
    assert result["evaluation"]["exact_bayesian"]["best_recognition_probability"] > 0


def scenario_report() -> Dict[str, Any]:
    """Machine-readable values for the three textbook checks and minimal tray."""
    from .core import fnum

    def post(doc, evidence):
        p = _posterior_for(doc, evidence)
        return {m: {"exact": str(v), "value": fnum(v)} for m, v in p.items()}

    minimal_model, minimal_weights = _model(minimal_two_sides())
    planner = WorkbenchPlanner(minimal_model)
    curve = planner.budget_curve(max_budget=3)
    return {
        "perfect_transmission": {
            "bright_posterior": post(perfect_transmission(), {"trans": 1}),
            "dark_posterior": post(perfect_transmission(), {"trans": 0}),
            "expected_conclusion": "both bright and dark branches are checkout-eligible",
        },
        "half_b_transmits": {
            "bright_posterior": post(half_b_transmits(), {"trans": 1}),
            "dark_posterior": post(half_b_transmits(), {"trans": 0}),
            "expected_conclusion": "bright gives A=2/3 and remains undecided; dark can check out B",
        },
        "upside_down_sealed": {
            "rim_bright_posterior": post(upside_down_sealed(), {"rim": 1}),
            "expected_conclusion": "identical remaining reading leaves A/B at 1/2",
        },
        "minimal_two_sides": {
            "budget_curve": curve,
            "demo_path": [
                "register DEMO-001",
                "front_signal=bright -> A=2/3, undecided, spent=1",
                "flip -> back face, front action unavailable",
                "back_signal=bright -> A=1, spent=3 including setup fee=1",
                "checkout A",
            ],
        },
    }


def run_selftest(verbose: bool = True) -> bool:
    tests = [
        ("textbook posteriors", test_textbook_posteriors),
        ("minimal optimal rate", test_minimal_optimal_rate),
        ("persistent end-to-end status", test_persistent_end_to_end_and_exit_status),
    ]
    try:
        for name, fn in tests:
            fn()
            if verbose:
                print(f"PASS {name}")
        with tempfile.TemporaryDirectory(prefix="mineral-pipeline-") as td:
            test_pipeline_outputs_for_another_input(Path(td))
        if verbose:
            print("PASS replacement-input pipeline")
        print("selftest: all checks passed")
        return True
    except Exception:
        import traceback

        traceback.print_exc()
        print("selftest: FAILED", file=__import__("sys").stderr)
        return False
