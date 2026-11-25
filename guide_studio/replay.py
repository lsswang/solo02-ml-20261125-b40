from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def replay_plates(plan: Dict[str, Any], plates: Sequence[str]) -> Dict[str, Any]:
    if not plan.get("feasible"):
        raise ValueError(f"Plan is not feasible: {plan.get('reason')}")
    device = plan["device"]
    assignment = plan["assignment"]
    groups = {g["device_key"]: g for g in plan["groups"]}
    recordings = {r["recording_asset_id"]: r for r in plan.get("recordings", [])}
    # wording_id -> recording via assignment/traces is less direct; build wording index.
    wording_to_recording = {r["wording_id"]: r for r in plan.get("recordings", [])}

    steps = []
    for i, station in enumerate(plates):
        if device == "wall":
            prev1 = plates[i - 1] if i >= 1 else "∅"
            key = "|".join((plan["language"], station, "prev1=" + prev1))
            memory = {"previous_station": prev1}
        else:
            prev1 = plates[i - 1] if i >= 1 else "∅"
            prev2 = plates[i - 2] if i >= 2 else "∅"
            key = "|".join((plan["language"], station, "prev2=" + prev2, "prev1=" + prev1))
            memory = {"second_previous_station": prev2, "previous_station": prev1}
        group = groups.get(key)
        if group is None:
            raise KeyError(f"No {device} recording group for key {key}. Is this plate string an active route?")
        wording_id = assignment[group["group_id"]]
        rec = wording_to_recording.get(wording_id, {})
        steps.append(
            {
                "stop_order": i + 1,
                "station_id": station,
                "device_memory": memory,
                "device_key": key,
                "group_id": group["group_id"],
                "wording_id": wording_id,
                "recording_asset_id": rec.get("recording_asset_id", wording_id),
                "audio_file": rec.get("audio_file", ""),
                "text": rec.get("text", ""),
                "duration_sec": rec.get("duration_sec"),
            }
        )
    total_duration = sum(float(s["duration_sec"] or 0) for s in steps)
    return {
        "device": device,
        "language": plan["language"],
        "plates": list(plates),
        "steps": steps,
        "unweighted_play_duration_sec": total_duration,
        "note": "Duration shown is the produced recording duration; objective weights route-level totals.",
    }


def format_replay_text(replay: Dict[str, Any]) -> str:
    lines = [
        f"Replay — device={replay['device']} language={replay['language']} plates={' -> '.join(replay['plates'])}",
        f"Total playing duration: {replay['unweighted_play_duration_sec']:.2f}s",
    ]
    for s in replay["steps"]:
        mem = s["device_memory"]
        if replay["device"] == "wall":
            mem_text = f"prev={mem['previous_station']}"
        else:
            mem_text = f"prev2={mem['second_previous_station']}, prev1={mem['previous_station']}"
        lines.append(
            f"{s['stop_order']:>2}. [{s['station_id']} | {mem_text}] "
            f"({s['recording_asset_id']}, {s['duration_sec']}s) {s['text']}"
        )
    return "\n".join(lines)
