"""Persistent ID-addressable guide workbench cases."""

from __future__ import annotations

import datetime as _dt
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import (
    InputFormatError,
    alive_mask_from_evidence,
    binary_reading,
    dump_json,
    load_json,
    money,
    posterior_from_mask,
    validate_and_build,
)
from .planner import WorkbenchPlanner


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def case_path(state_dir: str | Path, case_id: str) -> Path:
    if not case_id or any(ch in case_id for ch in "/\\") or case_id in {".", ".."}:
        raise InputFormatError(f"unsafe case id {case_id!r}")
    root = Path(state_dir)
    return root / "cases" / f"{case_id}.json"


def load_case(state_dir: str | Path, case_id: str) -> Dict[str, Any]:
    return load_json(case_path(state_dir, case_id))


def save_case(state_dir: str | Path, case: Dict[str, Any]) -> Path:
    path = case_path(state_dir, case["case_id"])
    dump_json(path, case)
    return path


def model_for_case(case: Dict[str, Any]):
    data = case["input_snapshot"]
    weights = case.get("learned_weights") or data.get("learned_orientation_weights")
    return validate_and_build(data, weights)


def create_case(
    state_dir: str | Path,
    case_id: str,
    input_data: Dict[str, Any],
    learned_weights: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    truth: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    path = case_path(state_dir, case_id)
    if path.exists():
        raise FileExistsError(f"case {case_id!r} already exists at {path}")
    model = validate_and_build(input_data, learned_weights)
    paid = sorted(model.initial_paid)
    case: Dict[str, Any] = {
        "schema_version": "mineral-workbench-case/v1",
        "case_id": case_id,
        "created_at": _now(),
        "updated_at": _now(),
        "input_snapshot": input_data,
        "learned_weights": learned_weights
        or input_data.get("learned_orientation_weights"),
        "evidence": {},
        "used_tests": [],
        "side": model.initial_side,
        "flips_remaining": model.max_flips,
        "paid_instruments": paid,
        "spent_fraction": "0",
        "spent": 0.0,
        "budget": float(model.budget),
        "status": "open",
        "signed_material": None,
        "signoff_probability": None,
        "signed_at": None,
        "history": [{"at": _now(), "event": "register", "detail": "new slice registered"}],
    }
    if truth:
        # Simulation answer key.  status() never copies this into its output.
        case["simulation_truth"] = truth
    save_case(state_dir, case)
    return case


def _recommendation(case: Dict[str, Any]):
    model = model_for_case(case)
    planner = WorkbenchPlanner(model)
    spent = Fraction(case["spent_fraction"])
    rec = planner.recommend(
        case["evidence"],
        side=case["side"],
        flips_remaining=case["flips_remaining"],
        used_tests=case["used_tests"],
        paid_instruments=case["paid_instruments"],
        spent=spent,
        budget=model.budget,
    )
    return model, planner, rec


def observe(
    state_dir: str | Path,
    case_id: str,
    test_id: str,
    result: Any,
    *,
    auto_checkout: bool = False,
) -> Dict[str, Any]:
    case = load_case(state_dir, case_id)
    if case["status"] == "signed_out":
        raise RuntimeError(f"case {case_id} is already signed out as {case['signed_material']}")

    model = model_for_case(case)
    tid = str(test_id)
    test = model.active_test_by_id(tid)
    y = binary_reading(result)

    if tid in case["used_tests"]:
        raise RuntimeError(f"test {tid} has already been performed on {case_id}")

    if test.side != "either" and test.side != case["side"]:
        raise RuntimeError(
            f"test {tid} is only available on the {test.side} face; current face is {case['side']}. "
            "Flip or choose another action."
        )

    spent = Fraction(case["spent_fraction"])
    paid_mask = 0
    for iid in case["paid_instruments"]:
        paid_mask |= model.instrument_bit(iid)
    setup_fee = Fraction(0) if paid_mask & model.instrument_bit(test.instrument) else model.instruments[test.instrument].setup_fee
    charge = test.cost + setup_fee
    if spent + charge > model.budget + Fraction("1e-12"):
        raise RuntimeError(
            f"action {tid} costs {float(charge)} but only {float(model.budget - spent)} "
            f"of budget remains (budget={float(model.budget)}, spent={float(spent)})"
        )

    case["spent_fraction"] = str(spent + charge)
    case["spent"] = float(spent + charge)
    case["evidence"][tid] = y
    case["used_tests"].append(tid)
    if setup_fee > 0 and test.instrument not in case["paid_instruments"]:
        case["paid_instruments"].append(test.instrument)
    case["updated_at"] = _now()
    case["history"].append(
        {
            "at": case["updated_at"],
            "event": "test",
            "test_id": tid,
            "result": "bright" if y else "dark",
            "test_cost": float(test.cost),
            "setup_fee": float(setup_fee),
            "charge": float(charge),
            "instrument": test.instrument,
        }
    )

    if auto_checkout:
        # Save first so checkout sees the updated evidence even if it refuses.
        save_case(state_dir, case)
        try:
            checkout(state_dir, case_id)
        except RuntimeError:
            pass
        case = load_case(state_dir, case_id)
    else:
        save_case(state_dir, case)
    return case


def flip(state_dir: str | Path, case_id: str) -> Dict[str, Any]:
    case = load_case(state_dir, case_id)
    if case["status"] == "signed_out":
        raise RuntimeError(f"case {case_id} is already signed out")
    model = model_for_case(case)
    if int(case["flips_remaining"]) <= 0:
        raise RuntimeError("no flips remain for this slice")
    spent = Fraction(case["spent_fraction"])
    charge = model.flip_cost
    if spent + charge > model.budget + Fraction("1e-12"):
        raise RuntimeError("not enough budget remains to flip")

    old_side = case["side"]
    case["side"] = "back" if old_side == "front" else "front"
    case["flips_remaining"] = int(case["flips_remaining"]) - 1
    case["spent_fraction"] = str(spent + charge)
    case["spent"] = float(spent + charge)
    case["updated_at"] = _now()
    case["history"].append(
        {
            "at": case["updated_at"],
            "event": "flip",
            "from_side": old_side,
            "to_side": case["side"],
            "charge": float(charge),
        }
    )
    save_case(state_dir, case)
    return case


def checkout(state_dir: str | Path, case_id: str, material_id: Optional[str] = None) -> Dict[str, Any]:
    case = load_case(state_dir, case_id)
    if case["status"] == "signed_out":
        return case
    model, _, rec = _recommendation(case)
    if not rec["consistent"]:
        raise RuntimeError("evidence is inconsistent with all known templates; cannot sign out")
    if not rec["checkout_eligible"]:
        raise RuntimeError(
            f"highest material probability is {rec['best_probability']:.6f}, "
            f"below required {float(model.threshold):.2f}"
        )
    mid = material_id or rec["best_material"]
    if mid != rec["best_material"]:
        p = rec["posterior"].get(mid, 0.0)
        if p < float(model.threshold):
            raise RuntimeError(f"{mid} posterior {p:.6f} is below threshold")
    case["status"] = "signed_out"
    case["signed_material"] = mid
    case["signoff_probability"] = rec["posterior"][mid]
    case["signed_at"] = _now()
    case["updated_at"] = case["signed_at"]
    case["history"].append(
        {
            "at": case["signed_at"],
            "event": "checkout",
            "material": mid,
            "probability": case["signoff_probability"],
            "spent": case["spent"],
        }
    )
    save_case(state_dir, case)
    return case


def _action_status(case: Dict[str, Any], model) -> Dict[str, Any]:
    spent = Fraction(case["spent_fraction"])
    remaining = model.budget - spent
    paid_mask = 0
    for iid in case["paid_instruments"]:
        paid_mask |= model.instrument_bit(iid)

    available_now = []
    on_other_face = []
    used = []
    for t in model.active_tests:
        entry = {
            "test_id": t.id,
            "name": t.name,
            "side": t.side,
            "instrument": t.instrument,
            "test_cost": float(t.cost),
            "setup_fee_now": float(
                Fraction(0)
                if paid_mask & model.instrument_bit(t.instrument)
                else model.instruments[t.instrument].setup_fee
            ),
            "total_charge_now": float(
                t.cost
                + (
                    Fraction(0)
                    if paid_mask & model.instrument_bit(t.instrument)
                    else model.instruments[t.instrument].setup_fee
                )
            ),
        }
        entry["affordable_now"] = money(entry["total_charge_now"]) <= remaining
        if t.id in case["used_tests"]:
            used.append(entry)
        elif t.side == "either" or t.side == case["side"]:
            available_now.append(entry)
        else:
            on_other_face.append(entry)

    lost_if_flip = []
    if int(case["flips_remaining"]) > 0:
        for t in available_now:
            spec = model.test_by_id(t["test_id"])
            if spec.side == case["side"]:  # face-specific, not "either"
                lost_if_flip.append(t["test_id"])

    sealed_or_unavailable = [
        {"test_id": t.id, "name": t.name, "sealed": t.sealed}
        for t in model.tests
        if not t.active
    ]

    return {
        "remaining_unused_test_count": len(available_now) + len(on_other_face),
        "currently_available_count": len(available_now),
        "available_now": available_now,
        "on_other_face_after_flip": on_other_face,
        "sealed_or_unavailable": sealed_or_unavailable,
        "used": used,
        "front_specific_actions_lost_by_next_flip": lost_if_flip,
        "can_flip": int(case["flips_remaining"]) > 0 and remaining >= model.flip_cost,
        "flips_remaining": int(case["flips_remaining"]),
    }


def status(state_dir: str | Path, case_id: str) -> Dict[str, Any]:
    case = load_case(state_dir, case_id)
    model, planner, rec = _recommendation(case)
    alive = alive_mask_from_evidence(model, case["evidence"])
    _, weights, total = posterior_from_mask(model, alive)
    joint = []
    for h in model.hypotheses:
        p = weights[h.index] / total if total > 0 and weights[h.index] > 0 else Fraction(0)
        prior_joint = h.joint_prior
        joint.append(
            {
                "material": h.material,
                "material_name": model.material_names[h.material],
                "orientation": h.orientation,
                "posterior_joint_probability": float(p),
                "prior_joint_probability": float(prior_joint),
                "still_possible": bool((alive >> h.index) & 1),
            }
        )

    actions = _action_status(case, model)
    recommended_next = rec["recommended"]
    if case["status"] == "signed_out":
        recommended_next = {
            "type": "already_signed_out",
            "instruction": (
                f"本片已签出 {case['signed_material']}；无需继续试验。"
                "可凭片号随时复查花费和判断。"
            ),
        }
    spent = Fraction(case["spent_fraction"])
    if case["status"] == "signed_out":
        judgment = {
            "state": "signed_out",
            "material": case["signed_material"],
            "material_name": model.material_names.get(case["signed_material"], case["signed_material"]),
            "probability_at_signoff": case["signoff_probability"],
            "threshold": float(model.threshold),
        }
    elif rec["checkout_eligible"]:
        judgment = {
            "state": "ready_to_checkout",
            "material": rec["best_material"],
            "material_name": model.material_names.get(rec["best_material"], rec["best_material"]),
            "probability": rec["best_probability"],
            "threshold": float(model.threshold),
        }
    elif not rec["consistent"]:
        judgment = {"state": "inconsistent_evidence", "message": rec["reason"]}
    else:
        judgment = {
            "state": "undecided",
            "best_material": rec["best_material"],
            "best_probability": rec["best_probability"],
            "threshold": float(model.threshold),
            "message": "尚未达到签出阈值；可继续做推荐动作或保留未定",
        }

    return {
        "case_id": case_id,
        "status": case["status"],
        "created_at": case["created_at"],
        "updated_at": case["updated_at"],
        "current_face": case["side"],
        "flips_remaining": case["flips_remaining"],
        "cost": {
            "spent": float(spent),
            "budget": float(model.budget),
            "remaining": float(model.budget - spent),
            "currency": model.currency,
            "paid_instruments": case["paid_instruments"],
        },
        "evidence": case["evidence"],
        "used_tests": case["used_tests"],
        "actions": actions,
        "material_posterior": rec["posterior"],
        "joint_material_orientation_posterior": joint,
        "judgment": judgment,
        "recommended_next": recommended_next,
        "optimal_recognition_probability_from_here": rec[
            "optimal_recognition_probability_from_here"
        ],
        "history": case["history"],
    }
