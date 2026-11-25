"""Command-line interface for the two-speaker sound bench."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Sequence

from .calibrate import calibrate_file, load_recording
from .diagnose import run_diagnostics
from .ir import average_rooms, convolve_room, load_room, room_to_jsonable
from .linalg import max_abs, rms
from .metrics import frame_errors, write_frame_csv
from .render import evaluate_rendered_constraints, render_program, render_robust_program
from .synth import generate_sample_tree
from .verify import verify_track
from .wavio import read_wav, write_wav


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_cue_csv(path: str, cue_checks: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "id", "cabinet", "start_sample", "end_sample", "leak_window_end",
        "active_rmse", "active_peak_error", "leakage_rms", "leakage_peak",
        "leakage_rms_pass", "leakage_peak_pass", "active_rmse_pass",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for cue in cue_checks:
            row = dict(cue)
            checks = row.pop("checks", {})
            row.update(checks)
            writer.writerow(row)


def cmd_generate(args: argparse.Namespace) -> int:
    paths = generate_sample_tree(args.out)
    print(f"Generated synthetic sample bundle in {args.out} ({len(paths)} named paths)")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    learned = calibrate_file(args.recording, args.out)
    fit = learned["fit_report"]
    print(
        f"Learned {len(learned['tap_delays'])}-tap sparse IR per speaker/zone -> {args.out}; "
        f"fit RMSE near={fit['near']['rmse']:.2e}, far={fit['far']['rmse']:.2e}"
    )
    return 0


def cmd_holdout(args: argparse.Namespace) -> int:
    room, room_data = load_room(args.learned)
    meta, sr, channels = load_recording(args.recording)
    n = min(len(ch) for ch in channels)
    predicted = convolve_room(channels[:2], room, n)
    measured = channels[2:4]
    rows = frame_errors(predicted, measured, sr, args.frame_size, args.hop_size)

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "holdout_frame_errors.csv")
    json_path = os.path.join(args.outdir, "holdout_summary.json")
    write_frame_csv(csv_path, rows)

    summary: Dict[str, Any] = {
        "format": "soundbench-holdout-report-v1",
        "recording": os.path.basename(args.recording),
        "learned_room": os.path.basename(args.learned),
        "used_in_calibration": meta.get("used_in_calibration", True),
        "frames": len(rows),
        "frame_csv": os.path.basename(csv_path),
        "zones": {},
    }
    for zone in ("near", "far"):
        key_rmse = f"{zone}_rmse"
        key_mae = f"{zone}_mae"
        rmses = [row[key_rmse] for row in rows]
        summary["zones"][zone] = {
            "overall_rmse": (sum(v * v for v in rmses) / len(rmses)) ** 0.5 if rmses else 0.0,
            "mean_frame_rmse": sum(rmses) / len(rmses) if rmses else 0.0,
            "max_frame_rmse": max(rmses, default=0.0),
            "mean_frame_mae": sum(row[key_mae] for row in rows) / len(rows) if rows else 0.0,
        }
    _write_json(json_path, summary)
    print(f"Hold-out report: {csv_path} and {json_path}")
    print(json.dumps(summary["zones"], indent=2))
    return 0


def _unplug_mask(value: str | None) -> tuple[bool, bool]:
    if not value:
        return (True, True)
    if value in {"2", "spk2", "speaker2", "right"}:
        return (True, False)
    if value in {"1", "spk1", "speaker1", "left"}:
        return (False, True)
    raise argparse.ArgumentTypeError("--unplug must be 1 or 2")


def cmd_build(args: argparse.Namespace) -> int:
    room, _ = load_room(args.room)
    enabled = _unplug_mask(args.unplug)
    rendered = render_program(args.program, room, enabled)
    os.makedirs(args.outdir, exist_ok=True)
    track_path = os.path.join(args.outdir, "speaker_track.wav")
    write_wav(track_path, int(rendered["program"]["sample_rate"]), rendered["speakers"])
    naive_path = os.path.join(args.outdir, "naive_ignores_tails.wav")
    write_wav(naive_path, int(rendered["program"]["sample_rate"]), rendered["naive_speakers"])

    constraints = evaluate_rendered_constraints(rendered)
    cue_csv = os.path.join(args.outdir, "cue_metrics.csv")
    _write_cue_csv(cue_csv, constraints["cues"])

    pcm_verify_path = os.path.join(args.outdir, "pcm_verification.json")
    mics_wav = os.path.join(args.outdir, "playback_mics.wav")
    pcm = verify_track(track_path, args.room, pcm_verify_path, mics_wav)
    _sr_pcm, pcm_speakers = read_wav(track_path)
    pcm_error_speakers = [
        [pcm_speakers[s][i] - rendered["speakers"][s][i] for i in range(len(rendered["speakers"][s]))]
        for s in range(2)
    ]
    pcm_error_mics = convolve_room(pcm_error_speakers, room)
    pcm_roundtrip = {
        "speaker_peak_error": max(max_abs(ch) for ch in pcm_error_speakers),
        "mic_near_peak_error": max_abs(pcm_error_mics[0]),
        "mic_far_peak_error": max_abs(pcm_error_mics[1]),
        "mic_near_rms_error": rms(pcm_error_mics[0]),
        "mic_far_rms_error": rms(pcm_error_mics[1]),
        "note": "Float cue leakage is around 1e-16; these are the added PCM-16 round-trip errors.",
    }

    report = {
        "format": "soundbench-build-report-v1",
        "program": os.path.abspath(args.program),
        "room": os.path.abspath(args.room),
        "speaker_track": os.path.abspath(track_path),
        "naive_tail_ignoring_track": os.path.abspath(naive_path),
        "naive_cue_reports": rendered["naive_cue_reports"],
        "cue_metrics_csv": os.path.abspath(cue_csv),
        "simulated_playback_mics": os.path.abspath(mics_wav),
        "constraints": constraints,
        "pcm_verification": pcm,
        "pcm_roundtrip_error": pcm_roundtrip,
        "timeline": {
            "content_end_samples": rendered["content_end"],
            "flush_samples": rendered["flush_samples"],
            "release_silence_samples": rendered["release_silence_samples"],
            "exported_track_samples": len(rendered["speakers"][0]),
        },
    }
    report_path = os.path.join(args.outdir, "verification.json")
    _write_json(report_path, report)
    status = "PASS" if constraints["passes"] else "FAIL/INFEASIBLE"
    print(f"[{status}] exported {track_path}; report {report_path}")
    for cue in constraints["cues"]:
        print(
            f"  {cue['id']:<18} active_rmse={cue['active_rmse']:.2e} "
            f"leak_rms={cue['leakage_rms']:.2e} leak_peak={cue['leakage_peak']:.2e}"
        )
    if args.strict and not constraints["passes"]:
        return 3
    return 0


def cmd_build_robust(args: argparse.Namespace) -> int:
    rooms = []
    room_data_list = []
    for path in args.rooms:
        room, data = load_room(path)
        rooms.append(room)
        room_data_list.append(data)
    rendered, robust = render_robust_program(args.program, rooms)
    os.makedirs(args.outdir, exist_ok=True)

    avg_room = average_rooms(rooms)
    avg_path = os.path.join(args.outdir, "averaged_room.json")
    _write_json(
        avg_path,
        {
            "format": "soundbench-learned-room-v1",
            "sample_rate": room_data_list[0].get("sample_rate", 4000),
            "ir": room_to_jsonable(avg_room),
            "note": "Average of indistinguishable room-state calibrations.",
        },
    )

    track_path = os.path.join(args.outdir, "speaker_track.wav")
    write_wav(track_path, int(rendered["program"]["sample_rate"]), rendered["speakers"])
    pcm = verify_track(track_path, avg_path, os.path.join(args.outdir, "pcm_verification.json"),
                       os.path.join(args.outdir, "playback_mics.wav"))
    report = {
        "format": "soundbench-robust-build-report-v1",
        "program": os.path.abspath(args.program),
        "rooms": [os.path.abspath(p) for p in args.rooms],
        "speaker_track": os.path.abspath(track_path),
        "averaged_room": avg_path,
        "robust_checks": robust,
        "pcm_verification_on_average_room": pcm,
    }
    report_path = os.path.join(args.outdir, "robust_verification.json")
    _write_json(report_path, report)
    status = "PASS" if robust["passes"] else "FAIL/INFEASIBLE"
    print(f"[{status}] robust track for {len(rooms)} room states -> {track_path}")
    if args.strict and not robust["passes"]:
        return 3
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    report = verify_track(args.track, args.room, args.out, args.mics_wav)
    print(json.dumps(report["zones"], indent=2))
    print(json.dumps(report["post_track_tail"], indent=2))
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    report = run_diagnostics(args.samples, args.outdir)
    print(f"Wrote diagnostics to {args.outdir}/diagnostics.json and diagnostics_summary.md")
    sd = report["sum_diff_original"]["meets_near_one_far_zero"]
    fs = not report["far_also_sum_recalibration"]["isolable"]
    un = not report["restored_sum_diff_then_speaker_2_unplugged"]["isolable"]
    print(f"sum/diff isolable={sd}; far-sum correctly non-isolable={fs}; unplug correctly impossible={un}")
    return 0


def score_probe(probe_json: str, room_path: str, early: bool) -> Dict[str, Any]:
    from .ir import convolve_room

    meta, sr, channels = load_recording(probe_json)
    room, room_data = load_room(room_path)
    n = min(len(ch) for ch in channels)
    window_end = n
    if early:
        window_end = int(meta.get("early_window_samples", meta.get("tap_delays", [0])[1] + 32 if len(meta.get("tap_delays", [])) > 1 else 64))
        window_end = min(window_end, n)
    pred = convolve_room(channels[:2], room, n)
    scores = {}
    for zone, mi in (("near", 2), ("far", 3)):
        err = [pred[mi - 2][i] - channels[mi][i] for i in range(window_end)]
        scores[zone] = (sum(e * e for e in err) / max(1, len(err))) ** 0.5
    combined = (scores["near"] ** 2 + scores["far"] ** 2) ** 0.5
    return {"room": room_path, "window_end": window_end, "near_rmse": scores["near"],
            "far_rmse": scores["far"], "score": combined}


def cmd_classify_probe(args: argparse.Namespace) -> int:
    scores = [score_probe(args.recording, p, args.early) for p in args.rooms]
    best = min(scores, key=lambda x: x["score"])
    threshold = best["score"] * 1.35 + 0.0008
    candidates = [s for s in scores if s["score"] <= threshold]
    result = {
        "recording": os.path.abspath(args.recording),
        "mode": "early_partial_probe" if args.early else "full_response_probe",
        "scores": scores,
        "best": best,
        "indistinguishable_candidates": [os.path.splitext(os.path.basename(c["room"]))[0] for c in candidates],
        "safe_action": (
            "use a robust track verified for every candidate"
            if len(candidates) > 1 else f"use the fixed track for {os.path.basename(candidates[0]['room'])}"
        ),
    }
    if args.out:
        _write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0


def cmd_compare_routes(args: argparse.Namespace) -> int:
    from .routes import write_route_comparison

    report = write_route_comparison(args.samples, args.learned_dir, args.deliverables_dir, args.out)
    print(f"Wrote route comparison to {args.out} and its Markdown companion")
    print(f"chosen={report['chosen_route']}; early candidates={report['probe_route_early_window']['candidates']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-speaker / two-zone sound exhibition workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate a small reproducible sample bundle")
    p.add_argument("--out", default="samples")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("calibrate", help="learn a sparse IR from a 4-channel recording bundle")
    p.add_argument("--recording", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("evaluate-holdout", help="report frame-wise reconstruction error on unused audio")
    p.add_argument("--learned", required=True)
    p.add_argument("--recording", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--frame-size", type=int, default=64)
    p.add_argument("--hop-size", type=int, default=32)
    p.set_defaults(func=cmd_holdout)

    p = sub.add_parser("build", help="export a playable stereo speaker track for a program")
    p.add_argument("--program", required=True)
    p.add_argument("--room", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--unplug", choices=["1", "2"], help="negative control: disconnect speaker 1 or 2")
    p.add_argument("--strict", action="store_true", help="exit 3 if acoustic constraints fail")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("build-robust", help="export one track checked against all candidate rooms")
    p.add_argument("--program", required=True)
    p.add_argument("--rooms", nargs="+", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=cmd_build_robust)

    p = sub.add_parser("verify", help="replay an exported WAV through a learned room")
    p.add_argument("--track", required=True)
    p.add_argument("--room", required=True)
    p.add_argument("--out")
    p.add_argument("--mics-wav", dest="mics_wav")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("diagnose", help="run sum/diff, far-sum and unplugged teaching cases")
    p.add_argument("--samples", default="samples")
    p.add_argument("--outdir", default="samples/diagnostics")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("compare-routes", help="write fixed-track vs live-probe comparison report")
    p.add_argument("--samples", default="samples")
    p.add_argument("--learned-dir", default="samples/learned")
    p.add_argument("--deliverables-dir", default="samples/deliverables")
    p.add_argument("--out", default="samples/reports/route_comparison.json")
    p.set_defaults(func=cmd_compare_routes)

    p = sub.add_parser("classify-probe", help="match a live probe recording against learned states")
    p.add_argument("--recording", required=True)
    p.add_argument("--rooms", nargs="+", required=True)
    p.add_argument("--early", action="store_true", help="use only the short early-response window")
    p.add_argument("--out")
    p.set_defaults(func=cmd_classify_probe)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
