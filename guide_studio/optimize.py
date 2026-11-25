from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .model import evaluate_reference

START = "∅"


def active_routes(bundle: Dict[str, Any], closed_stations: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    closed = set(closed_stations or [])
    out = []
    for rid, route in sorted(bundle["routes"].items()):
        stops = bundle["route_stops"].get(rid, [])
        if any(s["station_id"] in closed for s in stops):
            continue
        out.append({**route, "stops": stops})
    return out


def _device_key(device: str, language: str, station_id: str, i: int, stop_ids: Sequence[str]) -> Tuple[str, ...]:
    prev1 = stop_ids[i - 1] if i >= 1 else START
    if device == "wall":
        return (language, station_id, "prev1=" + prev1)
    if device == "pen":
        prev2 = stop_ids[i - 2] if i >= 2 else START
        return (language, station_id, "prev2=" + prev2, "prev1=" + prev1)
    raise ValueError(device)


def _compatible_variants(
    bundle: Dict[str, Any], language: str, group_occurrences: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    station_id = group_occurrences[0]["station_id"]
    person_ids = {p["person_id"] for p in bundle["persons"].get(language, [])}
    variants = [
        w
        for w in bundle["wordings"]
        if w["language"] == language and w["station_id"] == station_id
    ]
    compatible = []
    for v in sorted(variants, key=lambda w: w["wording_id"]):
        ok = True
        for occ in group_occurrences:
            target = occ.get("stop_target_person_id") or ""
            intended = v.get("intended_person_id") or target
            if target and v.get("intended_person_id") and v["intended_person_id"] != target:
                ok = False
            if intended and intended not in person_ids:
                ok = False
        if ok:
            compatible.append(v)
    return compatible


def _asset_id(v: Dict[str, Any]) -> str:
    return v.get("recording_asset_id") or v["wording_id"]


def solve_language_device(
    bundle: Dict[str, Any],
    model: Dict[str, Any],
    language: str,
    device: str,
    closed_stations: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    routes = active_routes(bundle, closed_stations)
    if not routes:
        return {"device": device, "language": language, "feasible": False, "reason": "no active routes"}

    occurrences: List[Dict[str, Any]] = []
    groups_map: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for route in routes:
        stop_ids = [s["station_id"] for s in route["stops"]]
        for i, stop in enumerate(route["stops"]):
            key = _device_key(device, language, stop["station_id"], i, stop_ids)
            occ = {
                "occurrence_id": f"{route['route_id']}:{stop['stop_order']}",
                "route_id": route["route_id"],
                "stop_order": stop["stop_order"],
                "station_id": stop["station_id"],
                "stop_target_person_id": stop.get("target_person_id", ""),
                "route_weight": route["weight"],
                "device_key": "|".join(key),
            }
            occurrences.append(occ)
            groups_map[key].append(occ)

    groups = []
    choice_lists = []
    for gi, (key, occs) in enumerate(sorted(groups_map.items(), key=lambda kv: "|".join(kv[0]))):
        variants = _compatible_variants(bundle, language, occs)
        gid = f"G{gi+1:02d}"
        group = {
            "group_id": gid,
            "device_key": "|".join(key),
            "key_tuple": list(key),
            "station_id": occs[0]["station_id"],
            "occurrence_ids": [o["occurrence_id"] for o in occs],
            "compatible_wording_ids": [v["wording_id"] for v in variants],
        }
        groups.append(group)
        choice_lists.append(variants)
        if not variants:
            return {
                "device": device,
                "language": language,
                "feasible": False,
                "reason": f"no semantically compatible wording for group {group['device_key']}",
                "groups": [{k: v for k, v in group.items() if k != "key_tuple"} for group in groups],
            }

    variants_by_id = {w["wording_id"]: w for w in bundle["wordings"] if w["language"] == language}
    time_weight = float(bundle["config"].get("time_cost_per_second", 1.0))
    pen_capacity = int(bundle["config"].get("pen_capacity_per_language", 10**9))
    wall_bonus = int(bundle["config"].get("wall_bonus_stations_per_language", 1))

    best: Optional[Dict[str, Any]] = None
    feasible_count = 0
    evaluated_count = 0
    # The sample is deliberately <=6 stations. Product enumeration is exact and
    # reproducible; sort choices for deterministic tie-breaking.
    for assignment_tuple in itertools.product(*choice_lists):
        evaluated_count += 1
        assignment = {groups[i]["group_id"]: assignment_tuple[i]["wording_id"] for i in range(len(groups))}
        traces = []
        weighted_duration = 0.0
        valid = True
        invalid_reason = ""

        for route in routes:
            last_person = ""
            steps = []
            stop_ids = [s["station_id"] for s in route["stops"]]
            for i, stop in enumerate(route["stops"]):
                key = _device_key(device, language, stop["station_id"], i, stop_ids)
                group = next(g for g in groups if g["device_key"] == "|".join(key))
                wid = assignment[group["group_id"]]
                v = variants_by_id[wid]
                target = stop.get("target_person_id", "") or v.get("intended_person_id", "")
                if v.get("appellation"):
                    ev = evaluate_reference(
                        bundle,
                        model,
                        language=language,
                        wording=v.get("appellation", ""),
                        intended_person_id=target,
                        site_id=stop["station_id"],
                        last_person_id=last_person,
                    )
                    if not ev["approved"]:
                        valid = False
                        invalid_reason = ev["reason"]
                        break
                else:
                    ev = {
                        "approved": True,
                        "reason": "neutral wording without an appellation",
                        "intended_probability": None,
                        "top_person_id": "",
                    }
                contribution = route["weight"] * time_weight * float(v["duration_sec"])
                weighted_duration += contribution
                incoming = last_person
                if v.get("state_person_id"):
                    last_person = v["state_person_id"]
                steps.append(
                    {
                        "stop_order": stop["stop_order"],
                        "station_id": stop["station_id"],
                        "device_key": group["device_key"],
                        "group_id": group["group_id"],
                        "wording_id": wid,
                        "recording_asset_id": _asset_id(v),
                        "text": v.get("text", ""),
                        "appellation": v.get("appellation", ""),
                        "duration_sec": v["duration_sec"],
                        "route_weight": route["weight"],
                        "weighted_duration_cost": contribution,
                        "incoming_last_person_id": incoming,
                        "outgoing_last_person_id": last_person,
                        "reference_approved": ev["approved"],
                        "reference_reason": ev["reason"],
                        "intended_probability": ev.get("intended_probability"),
                        "top_person_id": ev.get("top_person_id"),
                    }
                )
            if not valid:
                break
            traces.append({"route_id": route["route_id"], "route_weight": route["weight"], "steps": steps})

        if not valid:
            continue

        selected_assets = sorted({_asset_id(variants_by_id[wid]) for wid in assignment.values()})
        recording_count = len(selected_assets)
        production_cost = 0.0
        fee_by_asset: Dict[str, float] = {}
        for wid in set(assignment.values()):
            v = variants_by_id[wid]
            aid = _asset_id(v)
            fee_by_asset.setdefault(aid, float(v["recording_fee"]))
        production_cost = sum(fee_by_asset.values())

        constraint = {"ok": True, "reason": ""}
        wall_versions: Dict[str, Set[str]] = defaultdict(set)
        if device == "pen" and recording_count > pen_capacity:
            constraint = {"ok": False, "reason": f"{recording_count} recordings exceed pen capacity {pen_capacity}"}
        if device == "wall":
            for g in groups:
                wid = assignment[g["group_id"]]
                wall_versions[g["station_id"]].add(_asset_id(variants_by_id[wid]))
            multi = {sid: len(assets) for sid, assets in wall_versions.items() if len(assets) > 1}
            if any(n > 2 for n in multi.values()):
                constraint = {"ok": False, "reason": "a station cannot use more than one extra wall version"}
            elif len(multi) > wall_bonus:
                constraint = {"ok": False, "reason": f"wall bonus allows {wall_bonus} multi-version station(s), got {len(multi)}"}
        if not constraint["ok"]:
            continue

        feasible_count += 1
        objective = weighted_duration + production_cost
        wall_bonus_used = sorted([sid for sid, assets in wall_versions.items() if len(assets) > 1]) if device == "wall" else []
        candidate = {
            "device": device,
            "language": language,
            "feasible": True,
            "objective": objective,
            "weighted_duration_cost": weighted_duration,
            "production_cost": production_cost,
            "recording_count": recording_count,
            "pen_capacity": pen_capacity if device == "pen" else None,
            "capacity_remaining": (pen_capacity - recording_count) if device == "pen" else None,
            "wall_bonus_used_stations": wall_bonus_used,
            "wall_bonus_allowed": wall_bonus if device == "wall" else None,
            "selected_asset_ids": selected_assets,
            "assignment": assignment,
            "traces": traces,
            "groups": [{k: v for k, v in g.items() if k != "key_tuple"} for g in groups],
            "fee_by_asset_id": fee_by_asset,
            "evaluated_assignments": evaluated_count,
            "feasible_assignments": feasible_count,
        }
        tie = (round(objective, 9), recording_count, tuple(assignment[g["group_id"]] for g in groups))
        if best is None or tie < best["_tie"]:
            candidate["_tie"] = tie
            best = candidate

    if best is None:
        return {
            "device": device,
            "language": language,
            "feasible": False,
            "reason": "no feasible reference-safe assignment under device/capacity constraints",
            "evaluated_assignments": evaluated_count,
            "groups": [{k: v for k, v in g.items() if k != "key_tuple"} for g in groups],
        }

    best.pop("_tie", None)
    # Attach recording details.
    asset_to_wording = {}
    for wid in best["assignment"].values():
        v = variants_by_id[wid]
        asset_to_wording[_asset_id(v)] = v
    best["recordings"] = [
        {
            "recording_asset_id": aid,
            "wording_id": asset_to_wording[aid]["wording_id"],
            "language": language,
            "station_id": asset_to_wording[aid]["station_id"],
            "appellation": asset_to_wording[aid]["appellation"],
            "text": asset_to_wording[aid]["text"],
            "duration_sec": asset_to_wording[aid]["duration_sec"],
            "recording_fee": asset_to_wording[aid]["recording_fee"],
            "audio_file": f"audio/{language}/{aid}.wav",
            "transcript_file": f"audio/{language}/{aid}.txt",
        }
        for aid in best["selected_asset_ids"]
    ]
    return best


def solve_all(
    bundle: Dict[str, Any],
    model: Dict[str, Any],
    devices: Sequence[str] = ("wall", "pen"),
    closed_stations: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    result = {"closed_stations": sorted(closed_stations or []), "devices": {}}
    for device in devices:
        result["devices"][device] = {}
        for language in sorted(bundle["persons"].keys()):
            result["devices"][device][language] = solve_language_device(
                bundle, model, language, device, closed_stations
            )
    return result


def plan_for_replay(optimization: Dict[str, Any], device: str, language: str) -> Dict[str, Any]:
    return optimization["devices"][device][language]
