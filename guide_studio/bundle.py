from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REQUIRED_FILES = [
    "config.json",
    "persons.csv",
    "appellations.csv",
    "listening_records.csv",
    "manuscript_lines.csv",
    "stations.csv",
    "routes.csv",
    "route_stops.csv",
    "wordings.csv",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).replace(";", "|").split("|") if x.strip()]


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def load_bundle(input_dir: str | Path) -> Dict[str, Any]:
    root = Path(input_dir)
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Input bundle {root} is missing: {', '.join(missing)}")

    config = read_json(root / "config.json")
    config.setdefault("seed", 20261125)
    config.setdefault("holdout_fraction", 0.25)
    config.setdefault("smoothing_alpha", 0.5)
    config.setdefault("mixture_weights", [0.03, 0.07, 0.25, 0.65])
    config.setdefault("pen_capacity_per_language", 7)
    config.setdefault("wall_bonus_stations_per_language", 1)
    config.setdefault("time_cost_per_second", 1.0)
    config.setdefault("mishearing_count", 2)
    threshold = config.setdefault("short_form_threshold", 0.8)
    if isinstance(threshold, (int, float)):
        config["short_form_threshold"] = {"zh": float(threshold), "en": float(threshold)}
    else:
        config["short_form_threshold"] = {k: float(v) for k, v in threshold.items()}
    for lang in ("zh", "en"):
        config["short_form_threshold"].setdefault(lang, 0.8)

    persons_raw = read_csv(root / "persons.csv")
    persons: Dict[str, List[Dict[str, str]]] = {}
    for r in persons_raw:
        persons.setdefault(r["language"], []).append(
            {"person_id": r["person_id"].strip(), "display_name": r["display_name"].strip()}
        )

    appellations = []
    for r in read_csv(root / "appellations.csv"):
        appellations.append(
            {
                "language": r["language"].strip(),
                "appellation": r["appellation"].strip(),
                "kind": r.get("kind", "short").strip() or "short",
                "person_id": (r.get("person_id") or "").strip(),
            }
        )

    records = []
    for r in read_csv(root / "listening_records.csv"):
        prior = split_list(r.get("prior_person_ids", ""))
        records.append(
            {
                "record_id": r["record_id"].strip(),
                "script_id": r["script_id"].strip(),
                "language": r["language"].strip(),
                "site_id": r["site_id"].strip(),
                "prior_person_ids": prior,
                "last_person_id": prior[-1] if prior else "",
                "heard_wording": r["heard_wording"].strip(),
                "chosen_person_id": r["chosen_person_id"].strip(),
                "intended_person_id": r.get("intended_person_id", "").strip(),
                "listener_id": r.get("listener_id", "").strip(),
            }
        )

    manuscript = []
    for r in read_csv(root / "manuscript_lines.csv"):
        prior = split_list(r.get("prior_person_ids", ""))
        manuscript.append(
            {
                "line_id": r["line_id"].strip(),
                "script_id": r["script_id"].strip(),
                "language": r["language"].strip(),
                "line_order": int(r.get("line_order", "0") or 0),
                "site_id": r["site_id"].strip(),
                "prior_person_ids": prior,
                "last_person_id": prior[-1] if prior else "",
                "wording": r["wording"].strip(),
                "intended_person_id": r.get("intended_person_id", "").strip(),
                "text": r.get("text", "").strip(),
            }
        )

    stations = {}
    for r in read_csv(root / "stations.csv"):
        stations[r["station_id"].strip()] = {
            "station_id": r["station_id"].strip(),
            "display_name": r.get("display_name", r["station_id"]).strip(),
            "silent": _as_bool(r.get("silent", "false")),
        }

    routes = {}
    for r in read_csv(root / "routes.csv"):
        routes[r["route_id"].strip()] = {
            "route_id": r["route_id"].strip(),
            "display_name": r.get("display_name", r["route_id"]).strip(),
            "weight": float(r.get("weight", "1") or 1.0),
            "gate_station_id": r.get("gate_station_id", "").strip(),
        }

    stops = []
    for r in read_csv(root / "route_stops.csv"):
        stops.append(
            {
                "route_id": r["route_id"].strip(),
                "stop_order": int(r["stop_order"]),
                "station_id": r["station_id"].strip(),
                "target_person_id": (r.get("target_person_id") or "").strip(),
            }
        )
    stops.sort(key=lambda x: (x["route_id"], x["stop_order"]))
    route_stops: Dict[str, List[Dict[str, Any]]] = {}
    for s in stops:
        route_stops.setdefault(s["route_id"], []).append(s)

    wordings = []
    for r in read_csv(root / "wordings.csv"):
        wordings.append(
            {
                "wording_id": r["wording_id"].strip(),
                "language": r["language"].strip(),
                "station_id": r["station_id"].strip(),
                "variant_label": r.get("variant_label", "").strip(),
                "appellation": (r.get("appellation") or "").strip(),
                "text": r.get("text", "").strip(),
                "recording_fee": float(r.get("recording_fee", "0") or 0.0),
                "duration_sec": float(r.get("duration_sec", "0") or 0.0),
                "intended_person_id": (r.get("intended_person_id") or "").strip(),
                "state_person_id": (r.get("state_person_id") or "").strip(),
                "recording_asset_id": (r.get("recording_asset_id") or "").strip(),
            }
        )

    bundle = {
        "root": str(root),
        "config": config,
        "persons": persons,
        "appellations": appellations,
        "records": records,
        "manuscript": manuscript,
        "stations": stations,
        "routes": routes,
        "route_stops": route_stops,
        "wordings": wordings,
    }
    validate_bundle(bundle)
    return bundle


def validate_bundle(b: Dict[str, Any]) -> None:
    for lang, people in b["persons"].items():
        if len({p["person_id"] for p in people}) != len(people):
            raise ValueError(f"Duplicate person_id for language {lang}")
    station_ids = set(b["stations"])
    for rid, stops in b["route_stops"].items():
        if rid not in b["routes"]:
            raise ValueError(f"route_stops references unknown route {rid}")
        orders = [s["stop_order"] for s in stops]
        if orders != list(range(1, len(stops) + 1)):
            raise ValueError(f"Route {rid} stop_order must be contiguous from 1")
        for s in stops:
            if s["station_id"] not in station_ids:
                raise ValueError(f"Route {rid} references unknown station {s['station_id']}")
    for w in b["wordings"]:
        if w["station_id"] not in station_ids:
            raise ValueError(f"Wording {w['wording_id']} references unknown station")
        if w["duration_sec"] < 0 or w["recording_fee"] < 0:
            raise ValueError(f"Wording {w['wording_id']} has negative duration/fee")
    record_ids = [r["record_id"] for r in b["records"]]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("listening_records record_id values must be unique")

    full_maps = {}
    for a in b["appellations"]:
        if a.get("kind") == "full" and a.get("person_id"):
            key = (a["language"], a["appellation"])
            previous = full_maps.setdefault(key, a["person_id"])
            if previous != a["person_id"]:
                raise ValueError(f"Full appellation {key} maps to multiple persons")
