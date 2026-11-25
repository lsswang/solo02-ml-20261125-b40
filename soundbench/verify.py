"""Verify an exported two-channel WAV by convolution with a learned room."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .ir import convolve_room, load_room, max_delay
from .linalg import max_abs, rms
from .wavio import read_wav, write_wav


def verify_track(
    track_path: str,
    room_path: str,
    out_json: str | None = None,
    mics_wav: str | None = None,
) -> Dict[str, Any]:
    room, room_data = load_room(room_path)
    sr_track, speakers = read_wav(track_path)
    if len(speakers) != 2:
        raise ValueError("exported track must be stereo (speaker 1, speaker 2)")
    sr_room = int(room_data.get("sample_rate", sr_track))
    if sr_track != sr_room:
        raise ValueError(f"sample-rate mismatch: track {sr_track}, room {sr_room}")

    dmax = max_delay(room)
    mics = convolve_room(speakers, room)
    n_track = max(len(ch) for ch in speakers)
    tail = [mics[m][n_track:] for m in range(2)]

    report: Dict[str, Any] = {
        "format": "soundbench-verification-v1",
        "track": os.path.basename(track_path),
        "room": os.path.basename(room_path),
        "sample_rate": sr_track,
        "track_samples": n_track,
        "simulated_mic_samples": len(mics[0]),
        "max_ir_delay": dmax,
        "zones": {
            "near": {"full_rms": rms(mics[0]), "full_peak": max_abs(mics[0])},
            "far": {"full_rms": rms(mics[1]), "full_peak": max_abs(mics[1])},
        },
        "post_track_tail": {
            "near_peak": max_abs(tail[0]),
            "far_peak": max_abs(tail[1]),
            "near_rms": rms(tail[0]),
            "far_rms": rms(tail[1]),
            "samples": len(tail[0]),
        },
        "speaker_peaks": [max_abs(speakers[0]), max_abs(speakers[1])],
    }

    if mics_wav:
        # The simulated microphone signals are written for human/automated
        # playback inspection.  Normalization is unnecessary for passing
        # tracks; write_wav clips only an out-of-range negative control.
        write_wav(mics_wav, sr_track, mics)
        report["simulated_mic_wav"] = os.path.basename(mics_wav)

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    return report
