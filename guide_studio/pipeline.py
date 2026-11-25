from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .audio import m3u_for_replay, render_plan_audio
from .bundle import load_bundle, write_csv, write_json
from .model import evaluate_holdout, fit_referent_model
from .optimize import active_routes, solve_all
from .proofread import proofread_manuscript
from .replay import format_replay_text, replay_plates


def _jsonable_distribution(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    if isinstance(out.get("distribution_json"), dict):
        out["distribution_json"] = json.dumps(out["distribution_json"], ensure_ascii=False, sort_keys=True)
    if isinstance(out.get("distribution"), dict):
        out["distribution"] = json.dumps(out["distribution"], ensure_ascii=False, sort_keys=True)
    if isinstance(out.get("prior_person_ids"), list):
        out["prior_person_ids"] = "|".join(out["prior_person_ids"])
    return out


def _portrait_findings(opt: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = []
    for device, langs in opt["devices"].items():
        for lang, plan in langs.items():
            if not plan.get("feasible"):
                findings.append({"device": device, "language": lang, "feasible": False, "reason": plan.get("reason")})
                continue
            for trace in plan.get("traces", []):
                portrait_steps = [s for s in trace["steps"] if s["station_id"] == "portrait"]
                for s in portrait_steps:
                    findings.append(
                        {
                            "device": device,
                            "language": lang,
                            "route_id": trace["route_id"],
                            "portrait_wording_id": s["wording_id"],
                            "appellation": s["appellation"],
                            "text": s["text"],
                            "incoming_last_person_id": s["incoming_last_person_id"],
                            "intended_probability": s["intended_probability"],
                            "reference_approved": s["reference_approved"],
                            "recording_asset_id": s["recording_asset_id"],
                        }
                    )
    return findings


def run_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    closed_stations: Optional[Sequence[str]] = None,
    devices: Sequence[str] = ("wall", "pen"),
    render_audio: bool = True,
) -> Dict[str, Any]:
    bundle = load_bundle(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    closed = set(closed_stations or [])

    holdout = evaluate_holdout(bundle)
    production_model = fit_referent_model(bundle, fitted_on="all_records_for_production")
    proof_rows = proofread_manuscript(bundle, production_model)
    optimization = solve_all(bundle, production_model, devices=devices, closed_stations=closed)

    # Model and statistical deliverables.
    write_json(out / "referent_model.json", production_model)
    holdout_public = {k: v for k, v in holdout.items() if k != "train_model"}
    write_json(out / "holdout_log_loss.json", holdout_public)

    two = holdout["selected_mishearing_records"]
    write_json(out / "two_mishearing_records.json", two)
    write_csv(out / "two_mishearing_records.csv", [_jsonable_distribution(r) for r in two])
    by_lang_errors = []
    for lang, metrics in holdout["per_language"].items():
        for row in metrics.get("mishearing_records", []):
            by_lang_errors.append(row)
    write_json(out / "mishearing_records_by_language.json", by_lang_errors)
    write_csv(out / "mishearing_records_by_language.csv", [_jsonable_distribution(r) for r in by_lang_errors])

    proof_csv_rows = [_jsonable_distribution(r) for r in proof_rows]
    write_json(out / "proofreading_report.json", proof_rows)
    write_csv(out / "proofreading_report.csv", proof_csv_rows)

    # Plans and recording manifests.
    write_json(out / "optimization.json", optimization)
    plans_dir = out / "plans"
    manifest_rows = []
    for device, langs in optimization["devices"].items():
        for lang, plan in langs.items():
            write_json(plans_dir / f"{device}_{lang}.json", plan)
            if plan.get("feasible"):
                if render_audio:
                    render_plan_audio(out, plan)
                for rec in plan.get("recordings", []):
                    manifest_rows.append(
                        {
                            "device": device,
                            "language": lang,
                            "recording_asset_id": rec["recording_asset_id"],
                            "wording_id": rec["wording_id"],
                            "station_id": rec["station_id"],
                            "appellation": rec["appellation"],
                            "text": rec["text"],
                            "duration_sec": rec["duration_sec"],
                            "recording_fee": rec["recording_fee"],
                            "audio_file": rec["audio_file"],
                        }
                    )
    write_csv(out / "recording_manifest.csv", manifest_rows)

    # Deterministic route replays and M3U playlists.
    replay_dir = out / "replays"
    replay_summaries = []
    active = active_routes(bundle, closed)
    for device, langs in optimization["devices"].items():
        for lang, plan in langs.items():
            if not plan.get("feasible"):
                continue
            for route in active:
                plates = [s["station_id"] for s in route["stops"]]
                try:
                    rep = replay_plates(plan, plates)
                except KeyError:
                    continue
                stem = f"{device}_{lang}_{route['route_id']}"
                write_json(replay_dir / f"{stem}.json", rep)
                (replay_dir / f"{stem}.txt").write_text(format_replay_text(rep) + "\n", encoding="utf-8")
                m3u_for_replay(rep, replay_dir / f"{stem}.m3u")
                replay_summaries.append(
                    {"device": device, "language": lang, "route_id": route["route_id"], "plates": plates}
                )

    findings = _portrait_findings(optimization)
    summary = {
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(out.resolve()),
        "closed_stations": sorted(closed),
        "seed": bundle["config"].get("seed"),
        "holdout": {
            "split": holdout["split"],
            "mean_log_loss_nats": holdout["mean_log_loss_nats"],
            "mean_log_loss_bits": holdout["mean_log_loss_bits"],
            "per_language": {
                lang: {
                    "mean_log_loss_nats": m["mean_log_loss_nats"],
                    "mean_log_loss_bits": m["mean_log_loss_bits"],
                    "n_heldout_records": m["n_heldout_records"],
                    "n_actual_mishearings": m["n_actual_mishearings"],
                }
                for lang, m in holdout["per_language"].items()
            },
            "selected_two_record_ids": [r["record_id"] for r in two],
        },
        "proofreading": [
            {
                "line_id": r["line_id"],
                "language": r["language"],
                "approved": r["approved"],
                "decision": r["decision"],
                "intended_probability": r["intended_probability"],
                "threshold": r["threshold"],
                "reason": r["reason"],
            }
            for r in proof_rows
        ],
        "plans": {
            device: {
                lang: {
                    "feasible": plan.get("feasible", False),
                    "objective": plan.get("objective"),
                    "weighted_duration_cost": plan.get("weighted_duration_cost"),
                    "production_cost": plan.get("production_cost"),
                    "recording_count": plan.get("recording_count"),
                    "pen_capacity": plan.get("pen_capacity"),
                    "wall_bonus_used_stations": plan.get("wall_bonus_used_stations"),
                    "reason": plan.get("reason"),
                }
                for lang, plan in langs.items()
            }
            for device, langs in optimization["devices"].items()
        },
        "portrait_findings": findings,
        "replays": replay_summaries,
        "audio_note": "WAV files are deterministic quiet placeholder masters with transcript sidecars; replace them with studio audio using the same paths.",
    }
    write_json(out / "run_summary.json", summary)
    (out / "run_summary.md").write_text(render_markdown_summary(summary), encoding="utf-8")
    return summary


def render_markdown_summary(s: Dict[str, Any]) -> str:
    lines = ["# Guide Studio run summary", ""]
    lines.append(f"- Seed: `{s['seed']}`")
    lines.append(f"- Closed stations: `{', '.join(s['closed_stations']) or 'none'}`")
    h = s["holdout"]
    loss = h["mean_log_loss_nats"]
    lines.append(
        f"- Whole-script holdout log loss: `{loss:.4f}` nats"
        if isinstance(loss, (int, float))
        else "- Whole-script holdout log loss: n/a"
    )
    lines.append(f"- Two highlighted mishearing records: `{', '.join(h['selected_two_record_ids']) or 'none'}`")
    lines.append("")
    lines.append("## Plans")
    for device, langs in s["plans"].items():
        for lang, p in langs.items():
            if p["feasible"]:
                lines.append(
                    f"- `{device}/{lang}`: objective **{p['objective']:.2f}** "
                    f"(duration {p['weighted_duration_cost']:.2f} + production {p['production_cost']:.2f}), "
                    f"recordings {p['recording_count']}"
                    + (f"/{p['pen_capacity']}" if p["pen_capacity"] else "")
                    + (f", wall bonus at {p['wall_bonus_used_stations']}" if p["wall_bonus_used_stations"] else "")
                )
            else:
                lines.append(f"- `{device}/{lang}`: infeasible — {p['reason']}")
    lines.append("")
    lines.append("## Portrait station")
    for f in s["portrait_findings"]:
        if not f.get("feasible", True):
            continue
        prob = f.get("intended_probability")
        prob_text = f"{prob:.3f}" if isinstance(prob, (int, float)) else "1.000(full name)"
        lines.append(
            f"- {f['device']}/{f['language']}/{f['route_id']}: `{f['portrait_wording_id']}` "
            f"[{f['appellation'] or 'neutral'}], P(intended)={prob_text}, incoming=`{f['incoming_last_person_id'] or '∅'}`"
        )
    lines.append("")
    lines.append("## Proofreading")
    for p in s["proofreading"]:
        prob = p.get("intended_probability")
        prob_text = f"{prob:.3f}" if isinstance(prob, (int, float)) else "n/a"
        mark = "PASS" if p["approved"] else "REPLACE"
        lines.append(f"- [{mark}] `{p['line_id']}` {p['language']} P={prob_text}: {p['reason']}")
    return "\n".join(lines) + "\n"
