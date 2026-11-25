"""Learn within-material orientation distributions from partial labeled records."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .core import (
    InputFormatError,
    TrayModel,
    binary_reading,
    dump_json,
    fnum,
    load_json,
    money,
)


def _record_readings(record: Mapping[str, Any]) -> Dict[str, int]:
    raw = record.get("readings", record.get("observed", {}))
    if not isinstance(raw, dict):
        raise InputFormatError("record.readings must be an object")
    return {str(k): binary_reading(v) for k, v in raw.items()}


def learn_orientations(
    data: Mapping[str, Any],
    max_iter: int = 300,
    tolerance: float = 1e-11,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """EM for latent orientation templates under deterministic binary readings.

    Training records carry material labels but need not carry orientation
    labels.  A record with partial readings gives fractional posterior mass to
    every orientation template compatible with the observed entries.
    """
    materials_raw = data.get("materials", [])
    materials = [str(m["id"]) for m in materials_raw]
    templates = data.get("templates", [])
    by_material: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for tp in templates:
        by_material[str(tp.get("material", tp.get("material_id")))].append(tp)

    smoothing = float(data.get("orientation_smoothing", 0.5))
    if smoothing < 0:
        raise InputFormatError("orientation_smoothing must be nonnegative")

    theta: Dict[str, List[float]] = {}
    orientation_ids: Dict[str, List[str]] = {}
    for m in materials:
        tps = by_material.get(m, [])
        if not tps:
            raise InputFormatError(f"material {m} has no templates")
        orientation_ids[m] = [
            str(tp.get("orientation", tp.get("orientation_id", tp.get("id")))) for tp in tps
        ]
        init = []
        for j, tp in enumerate(tps):
            if "em_initial_weight" in tp:
                # Explicit initializer is useful for controlled experiments.
                init.append(max(1e-12, float(tp["em_initial_weight"])))
            else:
                # Deliberately do not treat the generator's "weight" truth as
                # known training input.  EM starts uniform and learns from the
                # partial material-labeled records.
                init.append(1.0)
        s = sum(init)
        theta[m] = [x / s for x in init]

    reference_weights: Dict[str, Dict[str, float]] = {}
    for m in materials:
        reference_weights[m] = {}
        for tp in by_material[m]:
            oid = str(tp.get("orientation", tp.get("orientation_id", tp.get("id"))))
            if tp.get("weight") is not None:
                reference_weights[m][oid] = float(tp["weight"])

    records = data.get("training_records", data.get("train_records", []))
    if not isinstance(records, list):
        raise InputFormatError("training_records must be a list")

    prepared: List[Tuple[str, Dict[str, int], Optional[str], str]] = []
    warnings: List[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            raise InputFormatError("each training record must be an object")
        rid = str(rec.get("id", f"train-{len(prepared)+1}"))
        m = str(rec.get("material", rec.get("material_id", "")))
        if m not in by_material:
            warnings.append(f"{rid}: unknown material {m!r}; skipped")
            continue
        readings = _record_readings(rec)
        known_orientation = rec.get("orientation", rec.get("orientation_id"))
        prepared.append((m, readings, str(known_orientation) if known_orientation else None, rid))

    iterations_used = 0
    last_log_likelihood: Optional[float] = None
    converged = False

    for iteration in range(1, max_iter + 1):
        iterations_used = iteration
        new_theta: Dict[str, List[float]] = {}
        log_likelihood = 0.0
        max_delta = 0.0

        for m in materials:
            tps = by_material[m]
            k_count = len(tps)
            counts = [0.0] * k_count
            compatible_any = False

            for rec_m, readings, known_o, rid in prepared:
                if rec_m != m:
                    continue
                compat: List[bool] = []
                for j, tp in enumerate(tps):
                    if known_o is not None:
                        oid = orientation_ids[m][j]
                        if known_o != oid:
                            compat.append(False)
                            continue
                    tp_readings = tp.get("readings", {})
                    ok = True
                    for test_id, y in readings.items():
                        if test_id not in tp_readings:
                            # Unknown test in a record cannot constrain.
                            continue
                        if binary_reading(tp_readings[test_id]) != y:
                            ok = False
                            break
                    compat.append(ok)

                denom = sum(theta[m][j] for j, ok in enumerate(compat) if ok)
                if denom <= 0:
                    warnings.append(f"{rid}: readings contradict every {m} template; skipped")
                    continue
                compatible_any = True
                rec_ll = math.log(max(denom, 1e-300))
                log_likelihood += rec_ll
                for j, ok in enumerate(compat):
                    if ok:
                        resp = theta[m][j] / denom
                        counts[j] += resp

            n_m = sum(1 for rec_m, _, _, _ in prepared if rec_m == m)
            denom_counts = sum(counts) + smoothing * k_count
            if denom_counts <= 0 or not compatible_any:
                updated = theta[m][:]
            else:
                updated = [(counts[j] + smoothing) / denom_counts for j in range(k_count)]
            new_theta[m] = updated
            max_delta = max(max_delta, max(abs(a - b) for a, b in zip(theta[m], updated)))

        theta = new_theta
        last_log_likelihood = log_likelihood
        if max_delta < tolerance:
            converged = True
            break

    weights_out: Dict[str, Dict[str, float]] = {}
    for m in materials:
        weights_out[m] = {oid: float(w) for oid, w in zip(orientation_ids[m], theta[m])}

    info = {
        "training_records": len(prepared),
        "iterations": iterations_used,
        "converged": converged,
        "tolerance": tolerance,
        "log_likelihood": last_log_likelihood,
        "smoothing": smoothing,
        "reference_template_weights_if_supplied": reference_weights,
        "warnings": sorted(set(warnings)),
    }
    return weights_out, info


def _predict_with_weights(
    model: TrayModel,
    weights_map: Mapping[str, Mapping[str, float]],
    evidence: Mapping[str, int],
    target_test: str,
) -> Tuple[float, Optional[str], float]:
    """Predict one target reading and material posterior from evidence."""
    from fractions import Fraction

    joint: List[float] = []
    for h in model.hypotheses:
        w = float(model.material_prior[h.material]) * float(
            weights_map[h.material][h.orientation]
        )
        joint.append(w)

    active_index = model.active_test_index
    for tid, y in evidence.items():
        if tid not in active_index:
            continue
        i = active_index[tid]
        for h in model.hypotheses:
            if h.outcomes[i] != y:
                joint[h.index] = 0.0

    total = sum(joint)
    if total <= 0:
        return float("nan"), None, 0.0
    target_i = active_index[target_test]
    p_bright = sum(joint[h.index] for h in model.hypotheses if h.outcomes[target_i] == 1) / total
    material_mass: Dict[str, float] = {}
    for h in model.hypotheses:
        material_mass[h.material] = material_mass.get(h.material, 0.0) + joint[h.index]
    best_m = max(material_mass, key=material_mass.get)
    return p_bright, best_m, material_mass[best_m] / total


def validate_heldout(
    data: Mapping[str, Any],
    model: TrayModel,
    weights_map: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    """Score readings deliberately excluded from each inference update."""
    records = data.get("heldout_records", data.get("holdout_records", []))
    if not isinstance(records, list):
        raise InputFormatError("heldout_records must be a list")

    checked = 0
    brier_sum = 0.0
    log_sum = 0.0
    correct_reading = 0
    material_correct = 0
    material_scored = 0
    per_test: Dict[str, Dict[str, float | int]] = defaultdict(
        lambda: {"n": 0, "brier": 0.0, "log_loss": 0.0, "accuracy": 0}
    )
    rows: List[Dict[str, Any]] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        rid = str(rec.get("id", f"heldout-{checked+1}"))
        true_material = rec.get("material", rec.get("material_id"))
        all_readings = _record_readings(rec)

        explicit_evidence = rec.get("evidence_readings", rec.get("evidence"))
        explicit_checks = rec.get("check_readings", rec.get("hidden_readings"))

        if isinstance(explicit_checks, dict):
            evidence = (
                _record_readings({"readings": explicit_evidence})
                if isinstance(explicit_evidence, dict)
                else {}
            )
            targets = _record_readings({"readings": explicit_checks})
            # If evidence is a list of test ids, interpret against readings.
            if isinstance(explicit_evidence, list):
                evidence = {
                    tid: all_readings[tid]
                    for tid in map(str, explicit_evidence)
                    if tid in all_readings
                }
        else:
            targets = dict(all_readings)
            evidence = None  # leave-one-out below

        for target, actual in targets.items():
            if target not in model.active_test_index:
                continue
            if evidence is None:
                ev = {k: v for k, v in all_readings.items() if k != target}
            else:
                ev = {k: v for k, v in evidence.items() if k != target}
            p, pred_m, pred_p = _predict_with_weights(model, weights_map, ev, target)
            if math.isnan(p):
                continue
            checked += 1
            brier = (p - actual) ** 2
            eps = 1e-12
            ll = -(actual * math.log(max(p, eps)) + (1 - actual) * math.log(max(1 - p, eps)))
            pred_reading = 1 if p >= 0.5 else 0
            reading_ok = int(pred_reading == actual)
            brier_sum += brier
            log_sum += ll
            correct_reading += reading_ok
            pt = per_test[target]
            pt["n"] = int(pt["n"]) + 1
            pt["brier"] = float(pt["brier"]) + brier
            pt["log_loss"] = float(pt["log_loss"]) + ll
            pt["accuracy"] = int(pt["accuracy"]) + reading_ok
            if true_material is not None and pred_m is not None:
                material_scored += 1
                material_correct += int(pred_m == str(true_material))
            rows.append(
                {
                    "record_id": rid,
                    "target_test": target,
                    "actual": "bright" if actual else "dark",
                    "predicted_bright_probability": round(p, 6),
                    "brier": round(brier, 6),
                    "log_loss": round(ll, 6),
                    "predicted_material": pred_m,
                    "material_posterior": round(pred_p, 6),
                    "true_material": true_material,
                }
            )

    per_test_out = {}
    for tid, st in per_test.items():
        n = int(st["n"])
        per_test_out[tid] = {
            "n": n,
            "brier": round(float(st["brier"]) / n, 6) if n else None,
            "log_loss": round(float(st["log_loss"]) / n, 6) if n else None,
            "reading_accuracy": round(int(st["accuracy"]) / n, 6) if n else None,
        }

    return {
        "checked_readings": checked,
        "records": len(records),
        "mean_brier": round(brier_sum / checked, 6) if checked else None,
        "mean_log_loss": round(log_sum / checked, 6) if checked else None,
        "reading_accuracy": round(correct_reading / checked, 6) if checked else None,
        "material_accuracy_on_evidence": (
            round(material_correct / material_scored, 6) if material_scored else None
        ),
        "per_test": per_test_out,
        "rows": rows,
    }


def write_learned_model(
    input_path: str,
    output_path: str,
) -> Dict[str, Any]:
    data = load_json(input_path)
    weights, info = learn_orientations(data)
    model = {
        "schema_version": data.get("schema_version", "mineral-optical-tray/v1"),
        "source_input": str(input_path),
        "orientation_weights": weights,
        "material_priors": {
            str(m["id"]): float(m.get("prior", m.get("probability", 1)))
            for m in data.get("materials", [])
        },
        "learning": info,
    }
    dump_json(output_path, model)
    return model
