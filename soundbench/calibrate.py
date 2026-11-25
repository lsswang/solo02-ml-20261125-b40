"""Learn sparse short impulse responses from a synchronized probe recording."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

from .ir import room_to_jsonable
from .linalg import least_squares, rms
from .wavio import read_wav


def resolve_relative(base_json: str, maybe_relative: str) -> str:
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base_json)), maybe_relative))


def load_recording(path: str) -> tuple[Dict[str, Any], int, List[List[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    wav_path = resolve_relative(path, meta["audio"])
    sr, channels = read_wav(wav_path)
    if sr != int(meta.get("sample_rate", sr)):
        raise ValueError(f"sample-rate mismatch in {path}")
    if len(channels) < 4:
        raise ValueError("recording must contain speaker 1, speaker 2, mic near, mic far")
    return meta, sr, channels


def fit_room(channels: Sequence[Sequence[float]], delays: Sequence[int]) -> List[List[List[tuple[int, float]]]]:
    x1, x2, y0, y1 = channels[:4]
    n = min(len(x1), len(x2), len(y0), len(y1))
    t_count = len(delays)
    p = 2 * t_count
    design: List[List[float]] = []
    targets0: List[float] = []
    targets1: List[float] = []

    for i in range(n):
        row = [0.0] * p
        for j, delay in enumerate(delays):
            source = i - delay
            if source >= 0:
                row[j] = x1[source]
                row[t_count + j] = x2[source]
        design.append(row)
        targets0.append(y0[i])
        targets1.append(y1[i])

    beta0 = least_squares(design, targets0)
    beta1 = least_squares(design, targets1)

    room: List[List[List[tuple[int, float]]]] = [[[], []], [[], []]]
    for j, delay in enumerate(delays):
        room[0][0].append((delay, beta0[j]))
        room[0][1].append((delay, beta0[t_count + j]))
        room[1][0].append((delay, beta1[j]))
        room[1][1].append((delay, beta1[t_count + j]))
    return room


def calibration_fit_report(
    room: Sequence[Sequence[Sequence[tuple[int, float]]]],
    channels: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    from .ir import convolve_room

    n = min(len(ch) for ch in channels[:4])
    pred = convolve_room([channels[0], channels[1]], room, n)  # type: ignore[arg-type]
    out: Dict[str, Any] = {}
    for zone, mic in (("near", 0), ("far", 1)):
        err = [pred[mic][i] - channels[2 + mic][i] for i in range(n)]
        out[zone] = {"rmse": rms(err), "peak_abs_error": max(abs(e) for e in err)}
    return out


def calibrate_file(recording_json: str, out_json: str) -> Dict[str, Any]:
    meta, sr, channels = load_recording(recording_json)
    delays = meta.get("tap_delays")
    if delays is None:
        taps = int(meta.get("ir_taps", 8))
        delays = list(range(taps))
    delays = [int(d) for d in delays]

    room = fit_room(channels, delays)
    fit_report = calibration_fit_report(room, channels)

    learned = {
        "format": "soundbench-learned-room-v1",
        "source_recording": os.path.basename(recording_json),
        "sample_rate": sr,
        "partition_position": meta.get("partition_position"),
        "tap_delays": delays,
        "ir": room_to_jsonable(room),
        "fit_report": fit_report,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(learned, f, indent=2, ensure_ascii=False)
    return learned
