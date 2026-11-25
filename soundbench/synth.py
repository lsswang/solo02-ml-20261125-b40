"""Deterministic synthetic rooms, probes, narration clips and programs."""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Sequence, Tuple

from .ir import Room, room_to_jsonable
from .wavio import write_wav

SAMPLE_RATE = 4000
BEAT_SAMPLES = 1000  # 0.25 s at 4 kHz
TAP_DELAYS = [0, 3, BEAT_SAMPLES]

# The tailless teaching example is exactly near = s1+s2, far = s1-s2.
SUM_DIFF_ROOM: Room = [
    [[(0, 1.0)], [(0, 1.0)]],
    [[(0, 1.0)], [(0, -1.0)]],
]

# Both points receive the sum: the two zones are no longer separable.
FAR_SUM_ROOM: Room = [
    [[(0, 1.0)], [(0, 1.0)]],
    [[(0, 1.0)], [(0, 1.0)]],
]


def _tail_room(scale: float, early_scale: float) -> Room:
    h0 = [
        [[(0, 1.0)], [(0, 1.0)]],
        [[(0, 1.0)], [(0, -1.0)]],
    ]
    early = [
        [(3, 0.05 * early_scale)],
        [(3, -0.04 * early_scale)],
    ]
    early_far = [
        [(3, -0.03 * early_scale)],
        [(3, 0.06 * early_scale)],
    ]
    beat_near_1 = [(BEAT_SAMPLES, -0.18 * scale)]
    beat_near_2 = [(BEAT_SAMPLES, 0.14 * scale)]
    beat_far_1 = [(BEAT_SAMPLES, 0.10 * scale)]
    beat_far_2 = [(BEAT_SAMPLES, -0.16 * scale)]
    return [
        [h0[0][0] + early[0] + beat_near_1, h0[0][1] + early[1] + beat_near_2],
        [h0[1][0] + early_far[0] + beat_far_1, h0[1][1] + early_far[1] + beat_far_2],
    ]


ROOM_STATES: Dict[str, Dict[str, Any]] = {
    # Fast early probe distinguishes open from the two heavier states.
    "partition_open": {
        "partition_position": 0,
        "room": _tail_room(0.55, 0.75),
        "description": "partition open; short and one-beat reflections are weak",
    },
    "partition_half": {
        "partition_position": 1,
        "room": _tail_room(1.00, 1.00),
        "description": "middle partition; original calibrated state",
    },
    "partition_closed": {
        "partition_position": 2,
        "room": _tail_room(1.22, 1.00),
        "description": "closed partition; early path matches half, late reflection is stronger",
    },
}


def _envelope(n: int, attack: int = 24, release: int = 80) -> List[float]:
    env = [1.0] * n
    attack = min(attack, n // 4)
    release = min(release, n // 4)
    for i in range(attack):
        env[i] = i / max(1, attack)
    for i in range(release):
        env[n - 1 - i] = i / max(1, release)
    return env


def narration_clip(seed: int, n: int = 2 * BEAT_SAMPLES, peak: float = 0.68) -> List[float]:
    """A short deterministic speech-like harmonic warble."""
    rng = random.Random(seed)
    f0 = rng.choice([170.0, 195.0, 220.0, 245.0, 275.0])
    out: List[float] = []
    for i in range(n):
        t = i / SAMPLE_RATE
        # A small phrase-like pitch movement.
        pitch = f0 * (1.0 + 0.08 * math.sin(2.0 * math.pi * 1.7 * t + seed))
        phase = 2.0 * math.pi * pitch * t
        value = (
            0.62 * math.sin(phase)
            + 0.24 * math.sin(2.0 * phase + 0.4)
            + 0.10 * math.sin(3.0 * phase - 0.7)
        )
        syllables = 0.72 + 0.28 * math.sin(2.0 * math.pi * 3.2 * t + seed * 0.31)
        out.append(value * syllables)
    env = _envelope(n)
    out = [v * e for v, e in zip(out, env)]
    m = max(abs(v) for v in out) or 1.0
    return [peak * v / m for v in out]


def calibration_sentence(n: int = BEAT_SAMPLES, peak: float = 1.0) -> List[float]:
    """A steady calibration sentence with gentle onset/release."""
    out = []
    for i in range(n):
        t = i / SAMPLE_RATE
        value = 0.7 * math.sin(2 * math.pi * 330 * t) + 0.2 * math.sin(2 * math.pi * 660 * t)
        out.append(value)
    env = _envelope(n, attack=32, release=64)
    m = max(abs(v * e) for v, e in zip(out, env)) or 1.0
    return [peak * v * e / m for v, e in zip(out, env)]


def probe_pair(n_active: int, seed: int, amplitude: float = 0.28) -> List[List[float]]:
    """Two independent deterministic wideband probe channels."""
    rng = random.Random(seed)
    x1: List[float] = []
    x2: List[float] = []
    # Small lead silence gives the recording a clear time origin.
    for _ in range(16):
        x1.append(0.0)
        x2.append(0.0)
    for i in range(n_active):
        # LFSR-like random polarity plus a weak chirp improves conditioning.
        a = 1.0 if rng.randrange(2) else -1.0
        b = 1.0 if rng.randrange(2) else -1.0
        chirp = math.sin(2 * math.pi * (180 + 420 * i / max(1, n_active)) * i / SAMPLE_RATE)
        x1.append(amplitude * (0.78 * a + 0.22 * chirp))
        x2.append(amplitude * (0.78 * b - 0.18 * chirp))
    return [x1, x2]


def _add_noise(channels: Sequence[List[float]], amplitude: float, seed: int) -> None:
    rng = random.Random(seed)
    for ch in channels:
        for i in range(len(ch)):
            ch[i] += amplitude * (2.0 * rng.random() - 1.0)


def make_recording_bundle(
    out_json: str,
    name: str,
    room: Room,
    tap_delays: Sequence[int],
    partition_position: str | int | None,
    probe_seed: int,
    probe_samples: int = 2048,
    noise_amplitude: float = 0.00025,
    sample_rate: int = SAMPLE_RATE,
    used_in_calibration: bool = True,
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Write a synchronized 4-channel calibration/holdout WAV + sidecar."""
    from .ir import convolve_room, max_delay

    dmax = max_delay(room)
    probes = probe_pair(probe_samples, probe_seed)
    total = len(probes[0]) + dmax + 32
    spk = [probes[0] + [0.0] * (total - len(probes[0])), probes[1] + [0.0] * (total - len(probes[1]))]
    mics = convolve_room(spk, room, total)
    if noise_amplitude:
        _add_noise(mics, noise_amplitude, probe_seed + 90000)

    wav_path = os.path.splitext(out_json)[0] + ".wav"
    write_wav(wav_path, sample_rate, [spk[0], spk[1], mics[0], mics[1]])
    meta = {
        "format": "soundbench-recording-v1",
        "name": name,
        "audio": os.path.basename(wav_path),
        "sample_rate": sample_rate,
        "channels": ["speaker_1_probe", "speaker_2_probe", "mic_near", "mic_far"],
        "partition_position": partition_position,
        "tap_delays": list(tap_delays),
        "used_in_calibration": used_in_calibration,
        "noise_note": "deterministic synthetic microphone noise is included in the WAV",
    }
    if extra_metadata:
        meta.update(extra_metadata)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def make_holdout_bundle(out_json: str, room: Room, seed: int = 4242) -> None:
    """Create a complete unused stereo program and its synchronized mics."""
    from .ir import convolve_room, max_delay

    rng = random.Random(seed)
    n = 3600
    x1: List[float] = []
    x2: List[float] = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = min(1.0, i / 40.0, max(0.0, (n - i) / 80.0))
        a = 0.34 * env * (
            0.6 * math.sin(2 * math.pi * 260 * t)
            + 0.25 * math.sin(2 * math.pi * 523 * t + 0.3)
        )
        b = 0.34 * env * (
            0.6 * math.sin(2 * math.pi * 392 * t + 0.8)
            + 0.25 * math.sin(2 * math.pi * 784 * t)
        )
        # A deterministic quiet pseudo-random component makes alignment errors
        # visible without dominating the harmonic material.
        a += 0.025 * (2 * rng.random() - 1)
        b += 0.025 * (2 * rng.random() - 1)
        x1.append(a)
        x2.append(b)

    dmax = max_delay(room)
    total = n + dmax + 16
    spk = [x1 + [0.0] * (total - n), x2 + [0.0] * (total - n)]
    mics = convolve_room(spk, room, total)
    _add_noise(mics, 0.00025, seed + 7)
    wav_path = os.path.splitext(out_json)[0] + ".wav"
    write_wav(wav_path, SAMPLE_RATE, [spk[0], spk[1], mics[0], mics[1]])
    write_wav(os.path.join(os.path.dirname(out_json), "holdout_speakers.wav"), SAMPLE_RATE, spk)
    meta = {
        "format": "soundbench-recording-v1",
        "name": "unused stereo hold-out",
        "audio": os.path.basename(wav_path),
        "sample_rate": SAMPLE_RATE,
        "channels": ["unused_speaker_1", "unused_speaker_2", "mic_near", "mic_far"],
        "partition_position": "partition_half",
        "tap_delays": TAP_DELAYS,
        "used_in_calibration": False,
        "note": "Neither speaker channel was used to fit the original calibration.",
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_sample_tree(outdir: str) -> Dict[str, str]:
    """Generate every sample referred to by the README/demo."""
    paths: Dict[str, str] = {}
    clips_dir = os.path.join(outdir, "clips")
    rec_dir = os.path.join(outdir, "recordings")
    probe_dir = os.path.join(outdir, "probes")
    prog_dir = os.path.join(outdir, "programs")
    room_dir = os.path.join(outdir, "rooms")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(probe_dir, exist_ok=True)
    os.makedirs(prog_dir, exist_ok=True)
    os.makedirs(room_dir, exist_ok=True)

    # Four curator narrations.
    for i in range(4):
        clip = narration_clip(100 + i)
        path = os.path.join(clips_dir, f"narration_{i+1}.wav")
        write_wav(path, SAMPLE_RATE, [clip])
        paths[f"clip_{i+1}"] = path

    scene_near = narration_clip(201, n=BEAT_SAMPLES, peak=0.72)
    scene_far = narration_clip(202, n=BEAT_SAMPLES, peak=0.72)
    write_wav(os.path.join(clips_dir, "scene_near.wav"), SAMPLE_RATE, [scene_near])
    write_wav(os.path.join(clips_dir, "scene_far.wav"), SAMPLE_RATE, [scene_far])

    cal_sentence = calibration_sentence()
    write_wav(os.path.join(clips_dir, "calibration_sentence.wav"), SAMPLE_RATE, [cal_sentence])

    # Ground-truth rooms are supplied only to make the synthetic benchmark
    # reproducible; the CLI learner reads the recordings, not these coefficients.
    truth = {
        "format": "soundbench-ground-truth-rooms-v1",
        "sample_rate": SAMPLE_RATE,
        "beat_samples": BEAT_SAMPLES,
        "states": {
            name: {
                "partition_position": spec["partition_position"],
                "description": spec["description"],
                "ir": room_to_jsonable(spec["room"]),
            }
            for name, spec in ROOM_STATES.items()
        },
        "sum_diff": {"ir": room_to_jsonable(SUM_DIFF_ROOM), "tap_delays": [0]},
        "far_sum": {"ir": room_to_jsonable(FAR_SUM_ROOM), "tap_delays": [0]},
    }
    truth_path = os.path.join(room_dir, "ground_truth_rooms.json")
    _write_json(truth_path, truth)
    paths["truth"] = truth_path

    # Main state recordings and a deliberately different probe realization.
    original_json = os.path.join(rec_dir, "cal_original.json")
    make_recording_bundle(
        original_json, "original middle partition", ROOM_STATES["partition_half"]["room"],
        TAP_DELAYS, "partition_half", probe_seed=1001,
    )
    paths["cal_original"] = original_json

    new_probe_json = os.path.join(rec_dir, "cal_new_probe.json")
    make_recording_bundle(
        new_probe_json, "same room, new probe realization", ROOM_STATES["partition_half"]["room"],
        TAP_DELAYS, "partition_half", probe_seed=777, probe_samples=2600,
    )
    paths["cal_new_probe"] = new_probe_json

    for name, spec in ROOM_STATES.items():
        p = os.path.join(rec_dir, f"cal_{name}.json")
        make_recording_bundle(
            p, name, spec["room"], TAP_DELAYS, name,
            probe_seed=3000 + spec["partition_position"],
        )
        paths[f"cal_{name}"] = p

        live_p = os.path.join(probe_dir, f"live_probe_{name}.json")
        make_recording_bundle(
            live_p, f"short scene-change probe for {name}", spec["room"], TAP_DELAYS, name,
            probe_seed=6100 + spec["partition_position"], probe_samples=128,
            used_in_calibration=False,
            extra_metadata={
                "purpose": "scene_change_state_probe",
                "probe_active_samples": 144,
                "early_window_samples": 160,
                "full_decision_samples": 128 + 16 + BEAT_SAMPLES + 16,
            },
        )
        paths[f"probe_{name}"] = live_p

    sumdiff_json = os.path.join(rec_dir, "cal_sum_diff.json")
    make_recording_bundle(
        sumdiff_json, "tailless near=sum far=difference", SUM_DIFF_ROOM, [0],
        "tailless_sum_diff", probe_seed=501, probe_samples=1024, noise_amplitude=0.0,
    )
    paths["cal_sum_diff"] = sumdiff_json

    farsum_json = os.path.join(rec_dir, "cal_far_sum.json")
    make_recording_bundle(
        farsum_json, "bad state: far also receives sum", FAR_SUM_ROOM, [0],
        "tailless_far_sum", probe_seed=502, probe_samples=1024, noise_amplitude=0.0,
    )
    paths["cal_far_sum"] = farsum_json

    holdout_json = os.path.join(rec_dir, "holdout.json")
    make_holdout_bundle(holdout_json, ROOM_STATES["partition_half"]["room"])
    paths["holdout"] = holdout_json

    beat_seconds = BEAT_SAMPLES / SAMPLE_RATE
    curator = {
        "format": "soundbench-program-v1",
        "name": "four-cabinet rotation",
        "sample_rate": SAMPLE_RATE,
        "beat_samples": BEAT_SAMPLES,
        "lead_silence_beats": 1,
        "tail_beats": 2,
        "speaker_peak_limit": 0.90,
        "leakage_rms_max": 0.020,
        "leakage_peak_max": 0.060,
        "active_rmse_max": 0.030,
        "cabinet_by_beat": [
            "silence", "near", "near", "far", "far",
            "near", "near", "far", "far", "release", "release",
        ],
        "cues": [
            {"id": "line_1", "clip": "../clips/narration_1.wav", "start_beat": 1, "cabinet": "near", "gain": 1.0},
            {"id": "line_2", "clip": "../clips/narration_2.wav", "start_beat": 3, "cabinet": "far", "gain": 1.0},
            {"id": "line_3", "clip": "../clips/narration_3.wav", "start_beat": 5, "cabinet": "near", "gain": 1.0},
            {"id": "line_4", "clip": "../clips/narration_4.wav", "start_beat": 7, "cabinet": "far", "gain": 1.0},
        ],
        "switch_times_seconds": [beat_seconds, 3 * beat_seconds, 5 * beat_seconds, 7 * beat_seconds],
    }
    curator_path = os.path.join(prog_dir, "curator_program.json")
    _write_json(curator_path, curator)
    paths["program_curator"] = curator_path

    scene = {
        "format": "soundbench-program-v1",
        "name": "one-beat-tail scene change",
        "sample_rate": SAMPLE_RATE,
        "beat_samples": BEAT_SAMPLES,
        "lead_silence_beats": 1,
        "scene_change_gap_beats": 1,
        "tail_beats": 1,
        "speaker_peak_limit": 0.90,
        "leakage_rms_max": 0.020,
        "leakage_peak_max": 0.060,
        "active_rmse_max": 0.030,
        "cabinet_by_beat": ["silence", "near", "scene_change_gap", "far", "release"],
        "cues": [
            {"id": "scene_left_line", "clip": "../clips/scene_near.wav", "start_beat": 1, "cabinet": "near", "gain": 1.0},
            {"id": "scene_right_line", "clip": "../clips/scene_far.wav", "start_beat": 3, "cabinet": "far", "gain": 1.0},
        ],
        "switch_times_seconds": [beat_seconds, 3 * beat_seconds],
        "note": "The room has a nonzero reflection at exactly one beat delay.",
    }
    scene_path = os.path.join(prog_dir, "scene_change_program.json")
    _write_json(scene_path, scene)
    paths["program_scene"] = scene_path

    return paths
