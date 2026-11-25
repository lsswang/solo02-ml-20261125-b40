"""End-to-end learning, validation, planning, and synthetic evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .core import dump_json, load_json, validate_and_build
from .learning import learn_orientations, validate_heldout
from .planner import WorkbenchPlanner


def _empirical_specimens(
    data: Mapping[str, Any],
    planner: WorkbenchPlanner,
    hypothesis_paths: List[Dict[str, Any]],
) -> Dict[str, Any]:
    specimens = data.get("new_specimens", [])
    path_by_key = {(p["material"], p["orientation"]): p for p in hypothesis_paths}
    rows = []
    signed = correct = 0
    total_spent = 0.0
    by_material: Dict[str, Dict[str, float | int]] = {}

    for sp in specimens:
        sid = str(sp.get("id", f"specimen-{len(rows)+1}"))
        m = str(sp.get("material", ""))
        o = str(sp.get("orientation", ""))
        path = path_by_key.get((m, o))
        if path is None:
            continue
        is_signed = path["terminal"] == "checkout"
        is_correct = bool(path["correct"])
        signed += int(is_signed)
        correct += int(is_correct)
        total_spent += float(path["spent"])
        rows.append(
            {
                "specimen_id": sid,
                "true_material": m,
                "true_orientation": o,
                "signed": is_signed,
                "called_material": path["called_material"],
                "correct": is_correct,
                "spent": path["spent"],
                "actions": path["actions"],
            }
        )
        stat = by_material.setdefault(
            m, {"n": 0, "signed": 0, "correct": 0, "spent": 0.0}
        )
        stat["n"] = int(stat["n"]) + 1
        stat["signed"] = int(stat["signed"]) + int(is_signed)
        stat["correct"] = int(stat["correct"]) + int(is_correct)
        stat["spent"] = float(stat["spent"]) + float(path["spent"])

    n = len(rows)
    return {
        "n": n,
        "signed_out": signed,
        "correct_signed_out": correct,
        "signed_rate": round(signed / n, 6) if n else None,
        "correct_rate_over_all": round(correct / n, 6) if n else None,
        "accuracy_among_signed": round(correct / signed, 6) if signed else None,
        "mean_spent": round(total_spent / n, 6) if n else None,
        "by_material": by_material,
        "specimens": rows,
    }


def run_pipeline(input_path: str | Path, out_dir: str | Path) -> Dict[str, Any]:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(input_path)
    weights, learning_info = learn_orientations(data)

    learned_model_doc = {
        "schema_version": data.get("schema_version"),
        "source_input": str(input_path),
        "orientation_weights": weights,
        "material_priors": {
            str(m["id"]): float(m.get("prior", m.get("probability", 1)))
            for m in data.get("materials", [])
        },
        "learning": learning_info,
    }
    dump_json(out_dir / "learned_model.json", learned_model_doc)

    model = validate_and_build(data, weights)
    validation = validate_heldout(data, model, weights)
    dump_json(out_dir / "validation_report.json", validation)

    planner = WorkbenchPlanner(model)
    summary = planner.plan_summary()
    curve = planner.budget_curve()
    table = planner.decision_table()
    policy_doc = {
        "schema_version": "mineral-workbench-policy/v1",
        "route": "adaptive_workbench_only",
        "source_input": str(input_path),
        "learned_orientation_weights": weights,
        "summary": {k: v for k, v in summary.items() if k != "hypothesis_paths"},
        "budget_curve": curve,
        "decision_table": table,
    }
    dump_json(out_dir / "workbench_policy.json", policy_doc)

    empirical = _empirical_specimens(data, planner, summary["hypothesis_paths"])
    evaluation = {
        "source_input": str(input_path),
        "exact_bayesian": {
            "best_recognition_probability": summary["best_recognition_probability"],
            "minimum_expected_cost_at_best_rate": summary[
                "minimum_expected_cost_at_best_rate"
            ],
            "expected_correct_recognition_probability": summary[
                "expected_correct_recognition_probability"
            ],
            "state_count": summary["state_count"],
        },
        "synthetic_new_specimens": empirical,
        "heldout_reading_checks": {
            "checked_readings": validation["checked_readings"],
            "mean_brier": validation["mean_brier"],
            "mean_log_loss": validation["mean_log_loss"],
            "reading_accuracy": validation["reading_accuracy"],
        },
        "hypothesis_paths": summary["hypothesis_paths"],
    }
    dump_json(out_dir / "evaluation_report.json", evaluation)

    resolved = dict(data)
    resolved["learned_orientation_weights"] = weights
    resolved["learning_info"] = learning_info
    dump_json(out_dir / "resolved_input.json", resolved)

    return {
        "learned_model": learned_model_doc,
        "validation": validation,
        "policy": policy_doc,
        "evaluation": evaluation,
        "resolved_input": out_dir / "resolved_input.json",
    }
