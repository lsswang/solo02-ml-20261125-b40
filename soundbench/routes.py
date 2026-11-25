"""Summarize the fixed-track versus scene-change probing routes."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .cli import score_probe


def _load_json(path: str) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_route_comparison(samples_dir: str, learned_dir: str, deliver_dir: str, out_json: str) -> Dict[str, Any]:
    states = ["partition_open", "partition_half", "partition_closed"]
    rooms = [os.path.join(learned_dir, f"{s}.json") for s in states]
    probe = os.path.join(samples_dir, "probes", "live_probe_partition_closed.json")

    early = [score_probe(probe, p, True) for p in rooms]
    full = [score_probe(probe, p, False) for p in rooms]
    early_best = min(early, key=lambda x: x["score"])
    full_best = min(full, key=lambda x: x["score"])
    early_threshold = early_best["score"] * 1.35 + 0.0008
    early_candidates = [s for s in early if s["score"] <= early_threshold]
    full_threshold = full_best["score"] * 1.35 + 0.0008
    full_candidates = [s for s in full if s["score"] <= full_threshold]

    robust_report = _load_json(os.path.join(deliver_dir, "robust_half_closed", "robust_verification.json"))
    universal_report = _load_json(os.path.join(deliver_dir, "universal_all_states", "robust_verification.json"))
    fixed_reports: Dict[str, Any] = {}
    for state in states:
        path = os.path.join(deliver_dir, f"fixed_{state}", "verification.json")
        data = _load_json(path)
        if data:
            fixed_reports[state] = data["constraints"]["passes"]

    with open(os.path.join(samples_dir, "recordings", "cal_original.json"), encoding="utf-8") as f:
        sr = json.load(f)["sample_rate"]
    with open(probe, encoding="utf-8") as f:
        probe_meta = json.load(f)
    echo_clear_samples = int(probe_meta.get("full_decision_samples", full_best["window_end"]))

    report: Dict[str, Any] = {
        "format": "soundbench-route-comparison-v1",
        "chosen_route": "fixed per-partition stereo tracks",
        "sample_rate": sr,
        "fixed_route": {
            "tracks": {s: os.path.join(deliver_dir, f"fixed_{s}", "speaker_track.wav") for s in states},
            "passes": fixed_reports,
            "universal_track_verified_on_all_states": (
                universal_report["robust_checks"]["passes"] if universal_report else None
            ),
            "universal_track": os.path.join(deliver_dir, "universal_all_states", "speaker_track.wav"),
            "scene_gap_samples_occupied_by_probe": 0,
            "decision_latency_samples": 0,
            "benefit": "No scene-change air time is spent probing; choose the directory matching the known partition.",
        },
        "probe_route_early_window": {
            "scores": early,
            "candidates": [os.path.splitext(os.path.basename(c["room"]))[0] for c in early_candidates],
            "decision_latency_samples": early_best["window_end"],
            "decision_latency_seconds": early_best["window_end"] / sr,
            "safe_start_after_echo_samples": echo_clear_samples,
            "safe_start_after_echo_seconds": echo_clear_samples / sr,
            "safe_branch": "half/closed use build-robust and a WAV verified on both states",
            "robust_track_passes": robust_report["robust_checks"]["passes"] if robust_report else None,
        },
        "probe_route_full_response": {
            "scores": full,
            "candidates": [os.path.splitext(os.path.basename(c["room"]))[0] for c in full_candidates],
            "decision_latency_samples": full_best["window_end"],
            "decision_latency_seconds": full_best["window_end"] / sr,
            "benefit": "Distinguishes all states only after the one-beat reflection has arrived.",
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = os.path.splitext(out_json)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Route comparison\n\n")
        f.write("Chosen complete route: **fixed per-partition tracks**.\n\n")
        f.write("| Route / observation | Latency | Result |\n|---|---:|---|\n")
        f.write("| Fixed per-state tracks | 0 samples in the scene gap | "
                + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in fixed_reports.items()) + " |\n")
        universal_pass = report["fixed_route"]["universal_track_verified_on_all_states"]
        f.write(f"| One universal fixed track | 0 samples in the scene gap | all-state verification: {universal_pass} |\n")
        e = report["probe_route_early_window"]
        f.write(f"| Early probe | {e['decision_latency_samples']} samples ({e['decision_latency_seconds']:.3f}s) | "
                f"candidates: {', '.join(e['candidates'])} |\n")
        g = report["probe_route_full_response"]
        f.write(f"| Full probe | {g['decision_latency_samples']} samples ({g['decision_latency_seconds']:.3f}s) | "
                f"candidates: {', '.join(g['candidates'])} |\n")
        f.write("\nThe early read reaches a partial decision at "
                f"{e['decision_latency_seconds']:.3f}s, but probe echo clearance and safe narration "
                f"start require {e['safe_start_after_echo_samples']} samples "
                f"({e['safe_start_after_echo_seconds']:.3f}s). The early probe deliberately cannot separate "
                "`partition_half` from `partition_closed`; that branch uses one robust WAV checked against both "
                "learned rooms rather than an unverified state-specific choice.\n")
    return report
