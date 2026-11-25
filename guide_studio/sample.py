from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .bundle import write_csv, write_json


SCRIPTS = [f"S{i:02d}" for i in range(1, 9)]
LANGS = ("zh", "en")
PRONOUN = {"zh": "她", "en": "she"}


def _base_config(seed: int) -> Dict[str, Any]:
    return {
        "museum_name": "林/周肖像馆双语导览",
        "seed": seed,
        "holdout_fraction": 0.25,
        "smoothing_alpha": 0.5,
        "mixture_weights": [0.03, 0.07, 0.25, 0.65],
        "short_form_threshold": {"zh": 0.8, "en": 0.8},
        "pen_capacity_per_language": 7,
        "wall_bonus_stations_per_language": 1,
        "time_cost_per_second": 1.0,
        "mishearing_count": 2,
        "duration_unit": "synthetic second-cost units",
        "fee_unit": "synthetic fee units; added directly to weighted seconds",
    }


def _persons_rows() -> List[Dict[str, str]]:
    rows = []
    names = {
        "lin": {"zh": "林岚", "en": "Lin Lan"},
        "zhou": {"zh": "周琴", "en": "Zhou Qin"},
        "chen": {"zh": "陈屿", "en": "Chen Yu"},
    }
    for pid, by_lang in names.items():
        for lang, name in by_lang.items():
            rows.append({"language": lang, "person_id": pid, "display_name": name})
    return rows


def _appellation_rows() -> List[Dict[str, str]]:
    data = [
        ("zh", "林岚", "full", "lin"),
        ("zh", "林岚女士", "full", "lin"),
        ("zh", "周琴", "full", "zhou"),
        ("zh", "周琴女士", "full", "zhou"),
        ("zh", "陈屿", "full", "chen"),
        ("zh", "她", "short", ""),
        ("zh", "林", "short", "lin"),
        ("zh", "周", "short", "zhou"),
        ("zh", "他", "short", ""),
        ("en", "Lin Lan", "full", "lin"),
        ("en", "Zhou Qin", "full", "zhou"),
        ("en", "Chen Yu", "full", "chen"),
        ("en", "Ms. Lin", "short", "lin"),
        ("en", "Ms. Zhou", "short", "zhou"),
        ("en", "Lin", "short", "lin"),
        ("en", "Zhou", "short", "zhou"),
        ("en", "she", "short", ""),
        ("en", "her", "short", ""),
        ("en", "he", "short", ""),
        ("en", "him", "short", ""),
    ]
    return [
        {"language": lang, "appellation": app, "kind": kind, "person_id": pid}
        for lang, app, kind, pid in data
    ]


def _station_rows() -> List[Dict[str, str]]:
    return [
        {"station_id": "east_gate", "display_name": "东门", "silent": "false"},
        {"station_id": "lin_hall", "display_name": "林岚展区", "silent": "false"},
        {"station_id": "west_gate", "display_name": "西门", "silent": "false"},
        {"station_id": "zhou_hall", "display_name": "周琴展区", "silent": "false"},
        {"station_id": "silent_porch", "display_name": "静默廊口", "silent": "true"},
        {"station_id": "portrait", "display_name": "肖像站", "silent": "false"},
    ]


def _route_rows() -> List[Dict[str, Any]]:
    return [
        {"route_id": "east", "display_name": "东路：东门—林岚—静默廊—肖像", "weight": 3, "gate_station_id": "east_gate"},
        {"route_id": "west", "display_name": "西路：西门—周琴—静默廊—肖像", "weight": 2, "gate_station_id": "west_gate"},
    ]


def _stop_rows() -> List[Dict[str, Any]]:
    def row(route, order, station, target=""):
        return {"route_id": route, "stop_order": order, "station_id": station, "target_person_id": target}

    return [
        row("east", 1, "east_gate", "lin"),
        row("east", 2, "lin_hall", "lin"),
        row("east", 3, "silent_porch", "lin"),
        row("east", 4, "portrait", "lin"),
        row("west", 1, "west_gate", "zhou"),
        row("west", 2, "zhou_hall", "zhou"),
        row("west", 3, "silent_porch", "zhou"),
        row("west", 4, "portrait", "lin"),
    ]


def _wording_rows(high_portrait_short_fee: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(lang, wid, station, label, app, text, fee, dur, intended="", state=""):
        rows.append(
            {
                "wording_id": wid,
                "language": lang,
                "station_id": station,
                "variant_label": label,
                "appellation": app,
                "text": text,
                "recording_fee": fee,
                "duration_sec": dur,
                "intended_person_id": intended,
                "state_person_id": state,
            }
        )

    for lang in LANGS:
        if lang == "zh":
            add(lang, "zh_e_gate_name", "east_gate", "点名林岚", "林岚", "东门入线：先读林岚。", 4, 4, "lin", "lin")
            add(lang, "zh_e_gate_generic", "east_gate", "通用开场", "", "东门入线，缓步进展厅。", 4, 3, "", "")
            add(lang, "zh_lin_full", "lin_hall", "全称", "林岚", "林岚常在窗前作画。", 4, 4, "lin", "lin")
            add(lang, "zh_lin_short", "lin_hall", "短称她", "她", "她常在窗前作画。", 4, 2, "lin", "lin")
            add(lang, "zh_w_gate_name", "west_gate", "点名周琴", "周琴", "西门入线：先读周琴。", 4, 4, "zhou", "zhou")
            add(lang, "zh_w_gate_generic", "west_gate", "通用开场", "", "西门入线，缓步进展厅。", 4, 3, "", "")
            add(lang, "zh_zhou_full", "zhou_hall", "全称", "周琴", "周琴常在灯下设色。", 4, 4, "zhou", "zhou")
            add(lang, "zh_zhou_short", "zhou_hall", "短称她", "她", "她常在灯下设色。", 4, 2, "zhou", "zhou")
            add(lang, "zh_porch_common", "silent_porch", "两路共用", "", "静默廊口，请把刚才的故事放在心里。", 4, 3, "", "")
            add(lang, "zh_porch_east", "silent_porch", "东路增录", "林岚", "想着林岚，再穿过静默廊。", 4, 2, "lin", "lin")
            add(lang, "zh_porch_west", "silent_porch", "西路增录", "周琴", "想着周琴，再穿过静默廊。", 4, 2, "zhou", "zhou")
            add(lang, "zh_portrait_full", "portrait", "全称共用", "林岚", "林岚，正在这幅肖像里。", 4, 4, "lin", "lin")
            add(lang, "zh_portrait_she", "portrait", "短称她", "她", "她，正在这幅肖像里。", 10 if high_portrait_short_fee else 4, 2, "lin", "lin")
        else:
            add(lang, "en_e_gate_name", "east_gate", "name Lin", "Lin Lan", "East gate: begin with Lin Lan.", 4, 4, "lin", "lin")
            add(lang, "en_e_gate_generic", "east_gate", "generic", "", "East gate: enter slowly.", 4, 3, "", "")
            add(lang, "en_lin_full", "lin_hall", "full name", "Lin Lan", "Lin Lan often painted by the window.", 4, 4, "lin", "lin")
            add(lang, "en_lin_short", "lin_hall", "short she", "she", "She often painted by the window.", 4, 2, "lin", "lin")
            add(lang, "en_w_gate_name", "west_gate", "name Zhou", "Zhou Qin", "West gate: begin with Zhou Qin.", 4, 4, "zhou", "zhou")
            add(lang, "en_w_gate_generic", "west_gate", "generic", "", "West gate: enter slowly.", 4, 3, "", "")
            add(lang, "en_zhou_full", "zhou_hall", "full name", "Zhou Qin", "Zhou Qin mixed colors beneath the lamp.", 4, 4, "zhou", "zhou")
            add(lang, "en_zhou_short", "zhou_hall", "short she", "she", "She mixed colors beneath the lamp.", 4, 2, "zhou", "zhou")
            add(lang, "en_porch_common", "silent_porch", "shared", "", "Silent porch: carry the recent story with you.", 4, 3, "", "")
            add(lang, "en_porch_east", "silent_porch", "East version", "Lin Lan", "Think of Lin Lan, then cross the silent porch.", 4, 2, "lin", "lin")
            add(lang, "en_porch_west", "silent_porch", "West version", "Zhou Qin", "Think of Zhou Qin, then cross the silent porch.", 4, 2, "zhou", "zhou")
            add(lang, "en_portrait_full", "portrait", "full shared", "Lin Lan", "Lin Lan is present in this portrait.", 4, 4, "lin", "lin")
            add(lang, "en_portrait_she", "portrait", "short she", "she", "She is present in this portrait.", 10 if high_portrait_short_fee else 4, 2, "lin", "lin")
    return rows


def _manuscript_rows(revised: bool = False) -> List[Dict[str, Any]]:
    west_zh_wording = "林岚" if revised else "她"
    west_en_wording = "Lin Lan" if revised else "she"
    return [
        {"line_id": "ML-ZH-01", "script_id": "MS-ZH-EAST", "language": "zh", "line_order": 1,
         "site_id": "portrait", "prior_person_ids": "lin", "wording": "她", "intended_person_id": "lin",
         "text": "她，正在这幅肖像里。"},
        {"line_id": "ML-ZH-02", "script_id": "MS-ZH-WEST", "language": "zh", "line_order": 1,
         "site_id": "portrait", "prior_person_ids": "zhou", "wording": west_zh_wording, "intended_person_id": "lin",
         "text": "林岚，正在这幅肖像里。"},
        {"line_id": "ML-ZH-03", "script_id": "MS-ZH-WEST", "language": "zh", "line_order": 2,
         "site_id": "portrait", "prior_person_ids": "zhou", "wording": "林岚", "intended_person_id": "lin",
         "text": "林岚，正在这幅肖像里。"},
        {"line_id": "ML-EN-01", "script_id": "MS-EN-EAST", "language": "en", "line_order": 1,
         "site_id": "portrait", "prior_person_ids": "lin", "wording": "she", "intended_person_id": "lin",
         "text": "She is present in this portrait."},
        {"line_id": "ML-EN-02", "script_id": "MS-EN-WEST", "language": "en", "line_order": 1,
         "site_id": "portrait", "prior_person_ids": "zhou", "wording": west_en_wording, "intended_person_id": "lin",
         "text": "Lin Lan is present in this portrait."},
        {"line_id": "ML-EN-03", "script_id": "MS-EN-WEST", "language": "en", "line_order": 2,
         "site_id": "portrait", "prior_person_ids": "zhou", "wording": "Lin Lan", "intended_person_id": "lin",
         "text": "Lin Lan is present in this portrait."},
    ]


def _record_rows(west_lin: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    counter = 1

    def add(lang, script, site, prior, wording, chosen, intended, listener):
        nonlocal counter
        rows.append(
            {
                "record_id": f"R-{lang.upper()}-{counter:04d}",
                "script_id": script,
                "language": lang,
                "site_id": site,
                "prior_person_ids": prior,
                "heard_wording": wording,
                "chosen_person_id": chosen,
                "intended_person_id": intended,
                "listener_id": listener,
            }
        )
        counter += 1

    def batch(lang, site, prior, wording, counts, intended, tag, n_total):
        # Distribute every batch deterministically across all eight scripts so a
        # whole-script holdout never removes an entire context.
        i = 0
        for pid, n in counts:
            for _ in range(n):
                script = SCRIPTS[i % len(SCRIPTS)]
                add(lang, script, site, prior, wording, pid, intended, f"L-{tag}-{i+1:03d}")
                i += 1

    for lang in LANGS:
        p = PRONOUN[lang]
        # East/root Lin context.
        batch(lang, "portrait", "lin", p, [("lin", 23), ("zhou", 1)], "lin", "east", 24)
        batch(lang, "lin_hall", "lin", p, [("lin", 11), ("zhou", 1)], "lin", "linhall", 12)
        # West/Zhou context. The replacement west_lin material flips the dominant identification.
        if west_lin:
            batch(lang, "portrait", "zhou", p, [("lin", 23), ("zhou", 1)], "lin", "westlin", 24)
        else:
            batch(lang, "portrait", "zhou", p, [("zhou", 23), ("lin", 1)], "zhou", "westzhou", 24)
        batch(lang, "zhou_hall", "zhou", p, [("zhou", 11), ("lin", 1)], "zhou", "zhouhall", 12)

        # One genuine wrong identification in every script/language, placed in a
        # Chen context so these audit records do not erase the Lin/Zhou contrast.
        for i, script in enumerate(SCRIPTS):
            add(
                lang, script, "east_gate", "chen", p, "zhou", "lin",
                f"L-mishear-{lang}-{i+1:02d}",
            )
    return rows


def generate_sample(output_dir: str | Path, scenario: str = "baseline", seed: int = 20261125) -> Path:
    west_lin = scenario in {"west_lin", "west-lin", "alt"}
    high_fee = scenario in {"high_fee", "high-fee"}
    revised_script = scenario in {"revised_script", "revised-script", "revision"}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", _base_config(seed))
    write_csv(root / "persons.csv", _persons_rows())
    write_csv(root / "appellations.csv", _appellation_rows())
    write_csv(root / "listening_records.csv", _record_rows(west_lin=west_lin))
    write_csv(root / "manuscript_lines.csv", _manuscript_rows(revised=revised_script))
    write_csv(root / "stations.csv", _station_rows())
    write_csv(root / "routes.csv", _route_rows())
    write_csv(root / "route_stops.csv", _stop_rows())
    write_csv(root / "wordings.csv", _wording_rows(high_portrait_short_fee=high_fee))
    scenario_md = root / "SCENARIO.md"
    if west_lin:
        scenario_text = "Replacement listening material: West-route listeners now reliably identify `lin`.\n"
    elif high_fee:
        scenario_text = "Cost sensitivity: the extra portrait short recording has fee 10 instead of 4.\n"
    elif revised_script:
        scenario_text = "Script replacement: the West manuscript already replaces the unsafe pronoun with the full name.\n"
    else:
        scenario_text = "Baseline listening material: East context points to `lin`, West context points to `zhou`.\n"
    scenario_md.write_text("# Scenario\n\n" + scenario_text, encoding="utf-8")
    return root
