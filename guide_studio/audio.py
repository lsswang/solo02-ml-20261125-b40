from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Any, Dict, Iterable


def render_placeholder_wav(path: Path, duration_sec: float, seed_text: str = "") -> None:
    """Write a quiet, playable chime placeholder of the contracted duration.

    The project has no offline TTS dependency.  Studio recordings can replace the
    WAV files while retaining filenames and durations.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    duration = max(0.2, float(duration_sec))
    n = int(sample_rate * duration)
    # Deterministic variant-specific pitch.
    base = 220 + (sum(seed_text.encode("utf-8")) % 18) * 18
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            t = i / sample_rate
            attack = min(1.0, t / 0.04)
            release = min(1.0, max(0.0, (duration - t) / 0.08))
            env = 0.10 * min(attack, release)
            # Two soft harmonics plus a very quiet periodic station marker.
            sample = env * (
                0.75 * math.sin(2 * math.pi * base * t)
                + 0.20 * math.sin(2 * math.pi * base * 1.5 * t)
            )
            frames.extend(struct.pack("<h", int(max(-1, min(1, sample)) * 32767)))
        wf.writeframes(bytes(frames))


def render_plan_audio(output_dir: Path, plan: Dict[str, Any]) -> int:
    if not plan.get("feasible"):
        return 0
    count = 0
    for rec in plan.get("recordings", []):
        wav_path = output_dir / rec["audio_file"]
        render_placeholder_wav(wav_path, float(rec["duration_sec"]), rec["recording_asset_id"])
        txt_path = output_dir / rec["transcript_file"]
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(
            f"{rec['text']}\n\nstation={rec['station_id']} language={rec['language']} "
            f"wording={rec['wording_id']} fee={rec['recording_fee']}\n",
            encoding="utf-8",
        )
        count += 1
    return count


def m3u_for_replay(replay: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U"]
    total = 0
    for s in replay["steps"]:
        dur = int(round(float(s.get("duration_sec") or 0)))
        total += dur
        lines.append(f"#EXTINF:{dur},{s['station_id']} {s['wording_id']}")
        lines.append(f"../{s['audio_file']}" if not s["audio_file"].startswith("/") else s["audio_file"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
