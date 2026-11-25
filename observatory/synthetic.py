from __future__ import annotations

import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from . import MAX_CANDIDATE_NIGHTS


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_calibration(
    out_dir: Path,
    *,
    responses: dict[str, tuple[float, float]],
    read_sigma: tuple[float, float],
    bg_sigma: tuple[float, float],
    repeats: int,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    levels = [("L1", 2.0), ("L2", 4.0), ("L3", 6.0)]
    bg_rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    night = 0
    for level_name, flux in levels:
        for _rep in range(repeats):
            night += 1
            true_bg1 = 100.0 + rng.gauss(0.0, 1.5)
            true_bg2 = 80.0 + rng.gauss(0.0, 1.2)
            measured_bg1 = true_bg1 + rng.gauss(0.0, bg_sigma[0])
            measured_bg2 = true_bg2 + rng.gauss(0.0, bg_sigma[1])
            bg_rows.append(
                {
                    "night": night,
                    "bg1": f"{measured_bg1:.6f}",
                    "bg2": f"{measured_bg2:.6f}",
                    "bg_sigma1": bg_sigma[0],
                    "bg_sigma2": bg_sigma[1],
                }
            )
            for star in ("甲", "乙"):
                masked = "乙" if star == "甲" else "甲"
                r1, r2 = responses[star]
                raw1 = true_bg1 + r1 * flux + rng.gauss(0.0, read_sigma[0])
                raw2 = true_bg2 + r2 * flux + rng.gauss(0.0, read_sigma[1])
                cal_rows.append(
                    {
                        "night": night,
                        "direction": f"cover_{masked}_measure_{star}",
                        "masked_star": masked,
                        "measured_star": star,
                        "brightness_level": level_name,
                        "known_flux": flux,
                        "raw_ap1": f"{raw1:.6f}",
                        "raw_ap2": f"{raw2:.6f}",
                        "sigma_ap1": read_sigma[0],
                        "sigma_ap2": read_sigma[1],
                    }
                )
    _write_csv(out_dir / "backgrounds.csv", ["night", "bg1", "bg2", "bg_sigma1", "bg_sigma2"], bg_rows)
    _write_csv(
        out_dir / "calibration.csv",
        [
            "night",
            "direction",
            "masked_star",
            "measured_star",
            "brightness_level",
            "known_flux",
            "raw_ap1",
            "raw_ap2",
            "sigma_ap1",
            "sigma_ap2",
        ],
        cal_rows,
    )


def _shift_rows_three() -> list[dict[str, Any]]:
    start = date(2026, 11, 20)
    costs = {1: 1.0, 2: 1.0, 3: 1.2}
    rows = []
    for n in range(1, 4):
        rows.append(
            {
                "night": n,
                "date": (start + timedelta(days=n - 1)).isoformat(),
                "cost": costs[n],
                "first_eligible": "true" if n == 1 else "false",
                "voucher_eligible": "true" if n in (2, 3) else "false",
                "fixed_eligible": "true",
                "note": "canonical three-night test",
            }
        )
    return rows


def _shift_rows_ab(open_voucher_night: int) -> list[dict[str, Any]]:
    start = date(2026, 12, 1)
    rows = []
    for n in range(1, MAX_CANDIDATE_NIGHTS + 1):
        is_first = n == 1
        is_voucher = n == open_voucher_night
        rows.append(
            {
                "night": n,
                "date": (start + timedelta(days=n - 1)).isoformat(),
                "cost": 1.0 if is_first else (1.1 if is_voucher else 0.0),
                "first_eligible": "true" if is_first else "false",
                "voucher_eligible": "true" if is_voucher else "false",
                "fixed_eligible": "true" if is_first or is_voucher else "false",
                "note": f"only voucher night {open_voucher_night} is open" if is_voucher else ("fixed first night" if is_first else "not bookable"),
            }
        )
    return rows


def generate_sample_tree(root: str | Path = "sample_inputs") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # Canonical original calibration:
    # 甲 unit response (1, 1), 乙 unit response (1, 0.5).
    write_calibration(
        root / "calibration_original",
        responses={"甲": (1.00, 1.00), "乙": (1.00, 0.50)},
        read_sigma=(0.10, 0.10),
        bg_sigma=(0.12, 0.12),
        repeats=4,
        seed=20261125,
    )

    # A genuine second calibration run.  Aperture two responses are now close
    # enough that their bounded/95% response intervals overlap.
    write_calibration(
        root / "calibration_recalibrated",
        responses={"甲": (1.00, 0.56), "乙": (1.00, 0.50)},
        read_sigma=(0.34, 0.34),
        bg_sigma=(0.22, 0.22),
        repeats=2,
        seed=7719,
    )

    _write_json(
        root / "observations" / "drop_6_3.json",
        {
            "observation_id": "SIM-DROP-6-3",
            "drops": [6.0, 3.0],
            "error_bounds": [0.30, 0.30],
            "response_uncertainty": "se95",
            "note": "Canonical simultaneous two-aperture dimming. Error is an absolute bounded error per aperture.",
        },
    )
    _write_json(
        root / "observations" / "drop_6_3_central.json",
        {
            "observation_id": "TEXTBOOK-CENTRAL-6-3",
            "drops": [6.0, 3.0],
            "error_bounds": [0.30, 0.30],
            "response_uncertainty": "none",
            "note": "Use fitted central responses only, matching the textbook (1,1)/(1,0.5) example.",
        },
    )
    _write_json(
        root / "calibration_textbook_exact.json",
        {
            "schema": "observatory-calibration-v1",
            "note": "Exact unit-response fixture for the (6,3) textbook check.",
            "responses": {
                "甲": {"ap1": 1.0, "ap2": 1.0},
                "乙": {"ap1": 1.0, "ap2": 0.5},
            },
            "standard_errors": {
                "甲": {"ap1": 0.0, "ap2": 0.0},
                "乙": {"ap1": 0.0, "ap2": 0.0},
            },
            "response_intervals_95": {
                "甲": {"ap1": [1.0, 1.0], "ap2": [1.0, 1.0]},
                "乙": {"ap1": [1.0, 1.0], "ap2": [0.5, 0.5]},
            },
            "columns_overlap_within_95": False,
        },
    )

    ep_four = [
        {
            "ephemeris_id": "E000",
            "star": "乙",
            "transit_time": "2026-11-20T20:00:00",
            "pattern": "000",
            "note": "canonical future light/dark string 000",
        },
        {
            "ephemeris_id": "E010",
            "star": "乙",
            "transit_time": "2026-11-20T20:20:00",
            "pattern": "010",
            "note": "canonical future light/dark string 010",
        },
        {
            "ephemeris_id": "E100",
            "star": "乙",
            "transit_time": "2026-11-20T20:40:00",
            "pattern": "100",
            "note": "canonical future light/dark string 100",
        },
        {
            "ephemeris_id": "E101",
            "star": "乙",
            "transit_time": "2026-11-20T21:00:00",
            "pattern": "101",
            "note": "canonical future light/dark string 101",
        },
    ]
    _write_csv(
        root / "ephemerides" / "ephemerides_four.csv",
        ["ephemeris_id", "star", "transit_time", "pattern", "note"],
        ep_four,
    )
    _write_json(
        root / "applications" / "order_four.json",
        {
            "order_id": "ORD-FOUR-0001",
            "horizon_nights": 3,
            "ephemerides_file": "../ephemerides/ephemerides_four.csv",
            "known_clear_nights": [],
            "voucher_fee": 0.0,
        },
    )
    _write_csv(
        root / "shifts" / "shifts_three.csv",
        ["night", "date", "cost", "first_eligible", "voucher_eligible", "fixed_eligible", "note"],
        _shift_rows_three(),
    )

    ep_ab = [
        {
            "ephemeris_id": "B4",
            "star": "乙",
            "transit_time": "2026-12-04T22:10:00",
            "pattern": "00010000",
            "note": "乙 ephemeris: dims only on night 4",
        },
        {
            "ephemeris_id": "B6",
            "star": "乙",
            "transit_time": "2026-12-06T22:10:00",
            "pattern": "00000100",
            "note": "乙 ephemeris: dims only on night 6",
        },
        {
            "ephemeris_id": "A4",
            "star": "甲",
            "transit_time": "2026-12-04T22:40:00",
            "pattern": "00010000",
            "note": "甲 ephemeris with exactly the same future binary pattern as B4",
        },
    ]
    _write_csv(
        root / "ephemerides" / "ephemerides_ab.csv",
        ["ephemeris_id", "star", "transit_time", "pattern", "note"],
        ep_ab,
    )
    _write_csv(
        root / "records" / "clear_ab.csv",
        ["record_id", "night", "result", "source", "note"],
        [
            {
                "record_id": "CLR-AB-N2",
                "night": 2,
                "result": "bright",
                "source": "verified empty-field patrol",
                "note": "All three current ephemerides predict bright night 2; record still enters the compatibility filter.",
            }
        ],
    )
    _write_json(
        root / "applications" / "order_ab.json",
        {
            "order_id": "ORD-AB-0002",
            "horizon_nights": MAX_CANDIDATE_NIGHTS,
            "ephemerides_file": "../ephemerides/ephemerides_ab.csv",
            "clear_records_file": "../records/clear_ab.csv",
            "known_clear_nights": [],
            "voucher_fee": 0.5,
        },
    )
    shift_fields = ["night", "date", "cost", "first_eligible", "voucher_eligible", "fixed_eligible", "note"]
    _write_csv(root / "shifts" / "shifts_ab_open4.csv", shift_fields, _shift_rows_ab(4))
    _write_csv(root / "shifts" / "shifts_ab_open5.csv", shift_fields, _shift_rows_ab(5))

    # A same-format replacement shift table can also be supplied by the grader;
    # this template documents every accepted column.
    _write_json(
        root / "README_input_formats.json",
        {
            "calibration_directory": ["backgrounds.csv", "calibration.csv"],
            "observation_file": {"drops": [6.0, 3.0], "error_bounds": [0.3, 0.3]},
            "application_file": {
                "order_id": "string",
                "horizon_nights": "1..8",
                "ephemerides": "list inline or use ephemerides_file CSV",
                "known_clear_nights": [2],
                "voucher_fee": 0.5,
            },
            "shift_csv_columns": shift_fields,
            "rule": "Replace any sample input with another file/directory of the same schema and rerun the same command.",
        },
    )
    return root
