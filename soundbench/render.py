"""Tail-aware two-speaker inverse rendering and constraint checks."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

from .ir import Room, average_rooms, convolve_room, h0_matrix, max_delay
from .linalg import SingularSystem, least_squares, rms, solve_2x2
from .metrics import signal_metrics
from .wavio import read_wav

ZONE_INDEX = {
    "near": 0, "left": 0, "cabinet_1": 0, "cabinet1": 0, "a": 0, "1": 0,
    "far": 1, "right": 1, "cabinet_2": 1, "cabinet2": 1, "b": 1, "2": 1,
}


def load_program(path: str) -> Tuple[Dict[str, Any], str]:
    with open(path, "r", encoding="utf-8") as f:
        program = json.load(f)
    # Small aliases make hand-edited replacements less brittle.
    if "speaker_peak_limit" not in program and "volume_limit" in program:
        program["speaker_peak_limit"] = program["volume_limit"]
    if "beat_samples" not in program and "beat_second" in program:
        program["beat_samples"] = int(round(float(program["beat_second"]) * int(program["sample_rate"])))
    return program, os.path.dirname(os.path.abspath(path))


def _load_mono(path: str, expected_sr: int) -> List[float]:
    sr, channels = read_wav(path)
    if sr != expected_sr:
        raise ValueError(f"{path}: sample rate {sr} != program rate {expected_sr}")
    if len(channels) == 1:
        return channels[0]
    n = max(len(ch) for ch in channels)
    out = [0.0] * n
    for ch in channels:
        for i, value in enumerate(ch):
            out[i] += value / len(channels)
    return out


def cue_start_sample(cue: Dict[str, Any], program: Dict[str, Any]) -> int:
    beat = int(program["beat_samples"])
    if "start_sample" in cue:
        return int(cue["start_sample"])
    if "start_second" in cue:
        return int(round(float(cue["start_second"]) * int(program["sample_rate"])))
    return int(cue["start_beat"]) * beat


def build_targets(program: Dict[str, Any], base_dir: str) -> Tuple[List[List[float]], List[Dict[str, Any]], int]:
    sr = int(program["sample_rate"])
    cues: List[Dict[str, Any]] = []
    content_end = 0
    loaded: List[Tuple[Dict[str, Any], List[float], int]] = []

    for cue in program.get("cues", []):
        clip_path = cue["clip"]
        if not os.path.isabs(clip_path):
            clip_path = os.path.normpath(os.path.join(base_dir, clip_path))
        clip = _load_mono(clip_path, sr)
        start = cue_start_sample(cue, program)
        gain = float(cue.get("gain", 1.0))
        end = start + len(clip)
        content_end = max(content_end, end)
        loaded.append((cue, clip, start))

    targets = [[0.0] * content_end, [0.0] * content_end]
    for cue, clip, start in loaded:
        zone_name = str(cue["cabinet"]).strip().lower().replace("-", "_").replace(" ", "_")
        if zone_name not in ZONE_INDEX:
            raise ValueError(f"unknown cabinet {cue['cabinet']!r}; use near/far or cabinet_1/cabinet_2")
        zone = ZONE_INDEX[zone_name]
        gain = float(cue.get("gain", 1.0))
        cue_id = cue.get("id", f"cue_{len(cues)}")
        for i, value in enumerate(clip):
            targets[zone][start + i] += gain * value
        cues.append(
            {
                "id": cue_id,
                "cabinet": str(cue["cabinet"]).lower(),
                "zone": zone,
                "start": start,
                "end": start + len(clip),
                "gain": gain,
                "clip_peak": max(abs(v) for v in clip),
            }
        )
    return targets, cues, content_end


def _tail_terms(room: Room) -> List[Tuple[int, int, int, float]]:
    terms = []
    for m in range(2):
        for s in range(2):
            for delay, coeff in room[m][s]:
                if delay > 0:
                    terms.append((m, s, delay, coeff))
    return terms


def inverse_render(
    targets: Sequence[Sequence[float]],
    room: Room,
    enabled_speakers: Tuple[bool, bool] = (True, True),
    extra_samples: int | None = None,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """Solve H0 x[n] = target[n] - tails(x[:n]) sample by sample.

    Because H0 is the invertible sum/difference matrix in the calibrated room,
    this recursion exactly includes short and one-beat-delay tails.  Extra
    zero-target samples act as a finite cancellation/flush tail.
    """
    n0 = max(len(targets[0]), len(targets[1]))
    d0 = list(targets[0]) + [0.0] * (n0 - len(targets[0]))
    d1 = list(targets[1]) + [0.0] * (n0 - len(targets[1]))
    dmax = max_delay(room)
    if extra_samples is None:
        extra_samples = 6 * dmax + 64
    total = n0 + extra_samples
    d0 += [0.0] * extra_samples
    d1 += [0.0] * extra_samples

    h0 = h0_matrix(room)
    a, b = h0[0]
    c, d = h0[1]
    det = a * d - b * c
    rank_deficient = abs(det) < 1e-10
    terms = _tail_terms(room)

    x1 = [0.0] * total
    x2 = [0.0] * total
    info = {"rank_deficient_h0": rank_deficient, "extra_samples": extra_samples}

    for n in range(total):
        r0 = d0[n]
        r1 = d1[n]
        for m, s, delay, coeff in terms:
            source = n - delay
            if source < 0:
                continue
            past = x1[source] if s == 0 else x2[source]
            contribution = coeff * past
            if m == 0:
                r0 -= contribution
            else:
                r1 -= contribution

        if not enabled_speakers[0] and not enabled_speakers[1]:
            break

        if not enabled_speakers[1]:
            col_norm = a * a + c * c
            x1[n] = 0.0 if col_norm < 1e-14 else (a * r0 + c * r1) / col_norm
            x2[n] = 0.0
        elif not enabled_speakers[0]:
            col_norm = b * b + d * d
            x1[n] = 0.0
            x2[n] = 0.0 if col_norm < 1e-14 else (b * r0 + d * r1) / col_norm
        elif rank_deficient:
            beta = least_squares([[a, b], [c, d]], [r0, r1])
            x1[n], x2[n] = beta
        else:
            x1[n], x2[n] = solve_2x2(a, b, c, d, r0, r1)

    return x1, x2, info


def naive_instantaneous_render(targets: Sequence[Sequence[float]], room: Room) -> Tuple[List[float], List[float]]:
    """Invert only H0; deliberately ignore delayed tails. Used for A/B evidence."""
    n = max(len(targets[0]), len(targets[1]))
    d0 = list(targets[0]) + [0.0] * (n - len(targets[0]))
    d1 = list(targets[1]) + [0.0] * (n - len(targets[1]))
    h0 = h0_matrix(room)
    a, b = h0[0]
    c, d = h0[1]
    det = a * d - b * c
    x1 = [0.0] * n
    x2 = [0.0] * n
    for i in range(n):
        if abs(det) < 1e-10:
            beta = least_squares([[a, b], [c, d]], [d0[i], d1[i]])
            x1[i], x2[i] = beta
        else:
            x1[i], x2[i] = solve_2x2(a, b, c, d, d0[i], d1[i])
    return x1, x2


def _cue_target_vector(program: Dict[str, Any], base_dir: str, selected_index: int) -> Tuple[List[List[float]], Dict[str, Any]]:
    # Reuse the loader but isolate one cue by temporarily filtering the program.
    single = dict(program)
    single["cues"] = [program["cues"][selected_index]]
    targets, cues, end = build_targets(single, base_dir)
    return targets, cues[0]


def render_program(
    program_path: str,
    room: Room,
    enabled_speakers: Tuple[bool, bool] = (True, True),
) -> Dict[str, Any]:
    program, base_dir = load_program(program_path)
    targets, cues, content_end = build_targets(program, base_dir)
    dmax = max_delay(room)
    beat = int(program.get("beat_samples", 1))
    extra_samples = 6 * dmax + 64

    x1, x2, inverse_info = inverse_render(targets, room, enabled_speakers, extra_samples)
    peak = max(signal_metrics(x1)["peak"], signal_metrics(x2)["peak"])
    limit = float(program.get("speaker_peak_limit", 1.0))
    requested_scale = 1.0
    if peak > limit:
        requested_scale = limit / peak
        x1 = [v * requested_scale for v in x1]
        x2 = [v * requested_scale for v in x2]

    release_samples = max(dmax + 1, int(round(float(program.get("tail_beats", 1)) * beat)))
    x1_export = x1 + [0.0] * release_samples
    x2_export = x2 + [0.0] * release_samples

    # Per-cue linear contributions.  These are used for leakage reports even
    # when cue windows overlap at a cabinet boundary.
    cue_reports: List[Dict[str, Any]] = []
    for i, cue in enumerate(program.get("cues", [])):
        cue_targets, cue_meta = _cue_target_vector(program, base_dir, i)
        c1, c2, _ = inverse_render(cue_targets, room, enabled_speakers, extra_samples)
        if requested_scale != 1.0:
            c1 = [v * requested_scale for v in c1]
            c2 = [v * requested_scale for v in c2]
        cue_mics = convolve_room([c1, c2], room)
        active_zone = cue_meta["zone"]
        leak_zone = 1 - active_zone
        start, end = cue_meta["start"], cue_meta["end"]
        eval_end = min(len(cue_mics[0]), end + dmax + 1)
        active = cue_mics[active_zone][start:end]
        wanted = cue_targets[active_zone][start:end]
        leak = cue_mics[leak_zone][start:eval_end]
        active_err = [active[j] - requested_scale * wanted[j] for j in range(min(len(active), len(wanted)))]
        cue_reports.append(
            {
                "id": cue_meta["id"],
                "cabinet": cue_meta["cabinet"],
                "start_sample": start,
                "end_sample": end,
                "leak_window_end": eval_end,
                "active_rmse": rms(active_err),
                "active_peak_error": max((abs(e) for e in active_err), default=0.0),
                "leakage_rms": rms(leak),
                "leakage_peak": max((abs(v) for v in leak), default=0.0),
            }
        )

    naive1, naive2 = naive_instantaneous_render(targets, room)
    if requested_scale != 1.0:
        naive1 = [v * requested_scale for v in naive1]
        naive2 = [v * requested_scale for v in naive2]
    naive_speakers = [
        naive1 + [0.0] * release_samples,
        naive2 + [0.0] * release_samples,
    ]
    naive_cue_reports: List[Dict[str, Any]] = []
    for i, cue in enumerate(program.get("cues", [])):
        cue_targets, cue_meta = _cue_target_vector(program, base_dir, i)
        n1, n2 = naive_instantaneous_render(cue_targets, room)
        if requested_scale != 1.0:
            n1 = [v * requested_scale for v in n1]
            n2 = [v * requested_scale for v in n2]
        cue_mics = convolve_room([n1, n2], room)
        active_zone = cue_meta["zone"]
        leak_zone = 1 - active_zone
        start, end = cue_meta["start"], cue_meta["end"]
        eval_end = min(len(cue_mics[0]), end + dmax + 1)
        leak = cue_mics[leak_zone][start:eval_end]
        active = cue_mics[active_zone][start:end]
        wanted = cue_targets[active_zone][start:end]
        active_err = [active[j] - requested_scale * wanted[j] for j in range(min(len(active), len(wanted)))]
        naive_cue_reports.append(
            {
                "id": cue_meta["id"],
                "active_rmse": rms(active_err),
                "leakage_rms": rms(leak),
                "leakage_peak": max((abs(v) for v in leak), default=0.0),
            }
        )

    return {
        "program": program,
        "program_path": os.path.abspath(program_path),
        "base_dir": base_dir,
        "speakers": [x1_export, x2_export],
        "naive_speakers": naive_speakers,
        "naive_cue_reports": naive_cue_reports,
        "content_end": content_end,
        "flush_samples": extra_samples,
        "release_silence_samples": release_samples,
        "speaker_peak_before_limit": peak,
        "applied_scale": requested_scale,
        "speaker_peak_limit": limit,
        "inverse": inverse_info,
        "cue_reports": cue_reports,
        "enabled_speakers": list(enabled_speakers),
    }


def evaluate_rendered_constraints(rendered: Dict[str, Any]) -> Dict[str, Any]:
    program = rendered["program"]
    leak_rms_max = float(program.get("leakage_rms_max", 0.02))
    leak_peak_max = float(program.get("leakage_peak_max", 0.06))
    active_max = float(program.get("active_rmse_max", 0.03))
    limit = float(program.get("speaker_peak_limit", 1.0))

    cue_checks = []
    all_pass = True
    for cue in rendered["cue_reports"]:
        checks = {
            "id": cue["id"],
            "leakage_rms_pass": cue["leakage_rms"] <= leak_rms_max,
            "leakage_peak_pass": cue["leakage_peak"] <= leak_peak_max,
            "active_rmse_pass": cue["active_rmse"] <= active_max,
        }
        cue_checks.append({**cue, "checks": checks})
        all_pass &= all(checks.values())

    final_peak = max(
        signal_metrics(rendered["speakers"][0])["peak"],
        signal_metrics(rendered["speakers"][1])["peak"],
    )
    speaker_pass = final_peak <= limit + 1e-9
    all_pass &= speaker_pass

    return {
        "passes": all_pass,
        "speaker_peak": final_peak,
        "speaker_peak_limit": limit,
        "speaker_peak_pass": speaker_pass,
        "applied_scale": rendered["applied_scale"],
        "leakage_rms_max": leak_rms_max,
        "leakage_peak_max": leak_peak_max,
        "active_rmse_max": active_max,
        "cues": cue_checks,
        "rank_deficient_h0": rendered["inverse"]["rank_deficient_h0"],
    }


def render_robust_program(
    program_path: str,
    rooms: Sequence[Room],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Render on the average room and check that one WAV works for every state."""
    if not rooms:
        raise ValueError("at least one room is required")
    avg_room = average_rooms(list(rooms))
    rendered = render_program(program_path, avg_room)
    program, base_dir = load_program(program_path)

    leak_rms_max = float(program.get("leakage_rms_max", 0.02))
    leak_peak_max = float(program.get("leakage_peak_max", 0.06))
    active_max = float(program.get("active_rmse_max", 0.03))
    scale = rendered["applied_scale"]
    extra_samples = rendered["flush_samples"]
    dmax = max_delay(avg_room)

    room_reports: List[Dict[str, Any]] = []
    overall_pass = True

    for room_index, room in enumerate(rooms):
        cue_reports: List[Dict[str, Any]] = []
        room_pass = True
        for cue_index, cue in enumerate(program.get("cues", [])):
            cue_targets, cue_meta = _cue_target_vector(program, base_dir, cue_index)
            c1, c2, _ = inverse_render(cue_targets, avg_room, (True, True), extra_samples)
            if scale != 1.0:
                c1 = [v * scale for v in c1]
                c2 = [v * scale for v in c2]
            mics = convolve_room([c1, c2], room)
            active_zone = cue_meta["zone"]
            leak_zone = 1 - active_zone
            start, end = cue_meta["start"], cue_meta["end"]
            eval_end = min(len(mics[0]), end + dmax + 1)
            active = mics[active_zone][start:end]
            wanted = cue_targets[active_zone][start:end]
            active_err = [active[i] - scale * wanted[i] for i in range(min(len(active), len(wanted)))]
            leak = mics[leak_zone][start:eval_end]
            report = {
                "id": cue_meta["id"],
                "cabinet": cue_meta["cabinet"],
                "active_rmse": rms(active_err),
                "leakage_rms": rms(leak),
                "leakage_peak": max((abs(v) for v in leak), default=0.0),
            }
            report["checks"] = {
                "active_rmse_pass": report["active_rmse"] <= active_max,
                "leakage_rms_pass": report["leakage_rms"] <= leak_rms_max,
                "leakage_peak_pass": report["leakage_peak"] <= leak_peak_max,
            }
            cue_reports.append(report)
            room_pass &= all(report["checks"].values())

        room_reports.append({"room_index": room_index, "passes": room_pass, "cues": cue_reports})
        overall_pass &= room_pass

    avg_constraints = evaluate_rendered_constraints(rendered)
    overall_pass &= avg_constraints["passes"]
    robust = {
        "passes": overall_pass,
        "method": "inverse of coefficient-averaged room, then checked on every candidate state",
        "candidate_room_count": len(rooms),
        "average_room_constraints": avg_constraints,
        "per_room": room_reports,
    }
    return rendered, robust

