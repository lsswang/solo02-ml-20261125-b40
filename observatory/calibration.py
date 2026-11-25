from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import MAX_CANDIDATE_NIGHTS


STAR_ORDER = ("甲", "乙")
APERTURE_ORDER = ("ap1", "ap2")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _as_float(row: dict[str, str], key: str, default: float | None = None) -> float:
    if key not in row or row[key] == "":
        if default is None:
            raise ValueError(f"missing numeric column {key!r}")
        return default
    return float(row[key])


@dataclass(frozen=True)
class Background:
    night: int
    ap1: float
    ap2: float
    sigma1: float
    sigma2: float


@dataclass(frozen=True)
class CalRow:
    night: int
    masked_star: str
    measured_star: str
    level: str
    known_flux: float
    raw1: float
    raw2: float
    sigma1: float
    sigma2: float


@dataclass(frozen=True)
class CalibrationInput:
    backgrounds: dict[int, Background]
    rows: list[CalRow]
    source_dir: Path

    @classmethod
    def load(cls, directory: str | Path) -> "CalibrationInput":
        d = Path(directory)
        bg_path = d / "backgrounds.csv"
        cal_path = d / "calibration.csv"
        if not bg_path.exists() or not cal_path.exists():
            raise FileNotFoundError(
                f"calibration directory must contain backgrounds.csv and calibration.csv: {d}"
            )

        backgrounds: dict[int, Background] = {}
        for row in _read_csv(bg_path):
            night = int(row["night"])
            backgrounds[night] = Background(
                night=night,
                ap1=_as_float(row, "bg1"),
                ap2=_as_float(row, "bg2"),
                sigma1=_as_float(row, "bg_sigma1", 0.05),
                sigma2=_as_float(row, "bg_sigma2", 0.05),
            )

        rows: list[CalRow] = []
        for row in _read_csv(cal_path):
            night = int(row["night"])
            if night not in backgrounds:
                raise ValueError(f"calibration row references night {night}, absent from backgrounds.csv")
            measured = row.get("measured_star", "").strip()
            masked = row.get("masked_star", "").strip()
            if not measured:
                # Backwards-compatible fallback for the original two-star
                # sample convention. Same-format files should include
                # measured_star explicitly and may use arbitrary labels.
                measured = next((s for s in STAR_ORDER if s != masked), "")
            if not measured:
                raise ValueError("measured_star must be a non-empty star label")

            def first_float(*keys: str, default: float | None = None) -> float:
                for key in keys:
                    if key in row and row[key] != "":
                        return float(row[key])
                if default is not None:
                    return default
                raise ValueError(f"missing numeric column; tried {', '.join(keys)}")

            rows.append(
                CalRow(
                    night=night,
                    masked_star=masked,
                    measured_star=measured,
                    level=str(row.get("brightness_level", row.get("level", ""))),
                    known_flux=first_float("known_flux"),
                    raw1=first_float("raw_ap1", "raw1"),
                    raw2=first_float("raw_ap2", "raw2"),
                    sigma1=first_float("sigma_ap1", "sigma1", default=0.15),
                    sigma2=first_float("sigma_ap2", "sigma2", default=0.15),
                )
            )
        if not rows:
            raise ValueError("calibration.csv contains no rows")
        return cls(backgrounds=backgrounds, rows=rows, source_dir=d)


def _fit_one(rows: Iterable[tuple[float, float, float]]) -> dict[str, float]:
    """Weighted non-negative through-origin fit y = r*x.

    Rows are (x=known flux, y=background-subtracted reading, sigma).
    """
    sx2w = 0.0
    sxyw = 0.0
    for x, y, sigma in rows:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        w = 1.0 / (sigma * sigma)
        sx2w += w * x * x
        sxyw += w * x * y
    if sx2w <= 0:
        raise ValueError("no positive known-flux samples for response fit")
    r_free = sxyw / sx2w
    r = max(0.0, r_free)

    # Weighted residual over-dispersion.  Formal errors are enlarged if the
    # repeated measurements scatter more than their declared read noise.
    n = 0
    sse = 0.0
    for x, y, sigma in rows:
        w = 1.0 / (sigma * sigma)
        sse += w * (y - r * x) ** 2
        n += 1
    df = max(1, n - 1)
    phi = max(1.0, sse / df)
    se = math.sqrt(phi / sx2w)
    return {"r": r, "se": se, "n": n, "weighted_sse": sse, "phi": phi}


def _stars_in_order(cal: CalibrationInput) -> list[str]:
    stars = list(dict.fromkeys(row.measured_star for row in cal.rows))
    if not stars:
        raise ValueError("no measured stars found in calibration.csv")
    return stars


def _fit_subset(cal: CalibrationInput, level_exclude: str | None = None) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = {s: [] for s in _stars_in_order(cal)}
    for star in out:
        for aperture_index in (0, 1):
            samples: list[tuple[float, float, float]] = []
            for row in cal.rows:
                if row.measured_star != star:
                    continue
                if level_exclude is not None and row.level == level_exclude:
                    continue
                bg = cal.backgrounds[row.night]
                bg_value = bg.ap1 if aperture_index == 0 else bg.ap2
                bg_sigma = bg.sigma1 if aperture_index == 0 else bg.sigma2
                raw = row.raw1 if aperture_index == 0 else row.raw2
                raw_sigma = row.sigma1 if aperture_index == 0 else row.sigma2
                y = raw - bg_value
                sigma = math.hypot(raw_sigma, bg_sigma)
                samples.append((row.known_flux, y, sigma))
            if not samples:
                raise ValueError(f"no calibration samples for star {star}, aperture index {aperture_index}")
            fit = _fit_one(samples)
            fit["aperture_index"] = aperture_index
            out[star].append(fit)
    return out


def _holdout_report(cal: CalibrationInput, z: float) -> list[dict[str, Any]]:
    levels = sorted({r.level for r in cal.rows if r.level != ""})
    report: list[dict[str, Any]] = []
    for level in levels:
        fits = _fit_subset(cal, level_exclude=level)
        errors: list[float] = []
        response_errors: list[float] = []
        standardized: list[float] = []
        for row in cal.rows:
            if row.level != level:
                continue
            bg = cal.backgrounds[row.night]
            for idx, (raw, raw_sigma, bg_value, bg_sigma) in enumerate(
                (
                    (row.raw1, row.sigma1, bg.ap1, bg.sigma1),
                    (row.raw2, row.sigma2, bg.ap2, bg.sigma2),
                )
            ):
                r = fits[row.measured_star][idx]["r"]
                predicted = bg_value + r * row.known_flux
                residual = raw - predicted
                sigma = math.hypot(raw_sigma, bg_sigma)
                errors.append(abs(residual))
                standardized.append(abs(residual) / sigma)
                if row.known_flux > 0:
                    response_errors.append(abs((raw - bg_value) / row.known_flux - r))
        report.append(
            {
                "held_out_level": level,
                "n_rows": sum(1 for r in cal.rows if r.level == level),
                "max_abs_residual": max(errors),
                "rmse_residual": math.sqrt(sum(e * e for e in errors) / len(errors)),
                "max_standardized_residual": max(standardized),
                "max_response_equivalent_error": max(response_errors, default=0.0),
                "z_used": z,
            }
        )
    return report


def calibrate(cal: CalibrationInput, z: float = 1.96) -> dict[str, Any]:
    fits = _fit_subset(cal)
    stars = list(fits.keys())
    responses: dict[str, dict[str, float]] = {}
    standard_errors: dict[str, dict[str, float]] = {}
    intervals: dict[str, dict[str, list[float]]] = {}
    for star in stars:
        responses[star] = {}
        standard_errors[star] = {}
        intervals[star] = {}
        for idx, ap in enumerate(APERTURE_ORDER):
            f = fits[star][idx]
            half = z * f["se"]
            r = f["r"]
            responses[star][ap] = r
            standard_errors[star][ap] = f["se"]
            intervals[star][ap] = [max(0.0, r - half), r + half]

    # Pairwise column-overlap report.  The sample problem has exactly two
    # stars; keeping it pairwise also allows another same-format star labels.
    pairwise_overlap: dict[str, Any] = {}
    all_overlap = True
    for i, star_a in enumerate(stars):
        for star_b in stars[i + 1 :]:
            pair_key = f"{star_a}__{star_b}"
            pairwise_overlap[pair_key] = {"star_names": [star_a, star_b], "apertures": {}}
            for ap in APERTURE_ORDER:
                lo_a, hi_a = intervals[star_a][ap]
                lo_b, hi_b = intervals[star_b][ap]
                common_lo = max(lo_a, lo_b)
                common_hi = min(hi_a, hi_b)
                ap_overlap = common_lo <= common_hi + 1e-12
                all_overlap = all_overlap and ap_overlap
                pairwise_overlap[pair_key]["apertures"][ap] = {
                    "overlap": ap_overlap,
                    "common_interval": [max(0.0, common_lo), max(common_lo, common_hi)] if ap_overlap else None,
                    f"{star_a}_interval": [lo_a, hi_a],
                    f"{star_b}_interval": [lo_b, hi_b],
                }

    primary_overlap: dict[str, Any] = {}
    if len(stars) >= 2:
        star_a, star_b = stars[0], stars[1]
        for ap in APERTURE_ORDER:
            lo_a, hi_a = intervals[star_a][ap]
            lo_b, hi_b = intervals[star_b][ap]
            common_lo = max(lo_a, lo_b)
            common_hi = min(hi_a, hi_b)
            ap_overlap = common_lo <= common_hi + 1e-12
            primary_overlap[ap] = {
                "stars": [star_a, star_b],
                "overlap": ap_overlap,
                "common_interval": [max(0.0, common_lo), max(common_lo, common_hi)] if ap_overlap else None,
                "star_1_interval": [lo_a, hi_a],
                "star_2_interval": [lo_b, hi_b],
            }

    holdout = _holdout_report(cal, z)
    # Conservative empirical response tolerance: statistical interval plus the
    # largest response-equivalent discrepancy seen in whole-level holdouts.
    validation_margin: dict[str, dict[str, float]] = {s: {} for s in stars}
    max_holdout_response_error = max(
        (item["max_response_equivalent_error"] for item in holdout), default=0.0
    )
    for star in stars:
        for ap in APERTURE_ORDER:
            validation_margin[star][ap] = z * standard_errors[star][ap] + max_holdout_response_error

    return {
        "schema": "observatory-calibration-v1",
        "source_dir": str(cal.source_dir),
        "stars": stars,
        "apertures": list(APERTURE_ORDER),
        "z": z,
        "responses": responses,
        "standard_errors": standard_errors,
        "response_intervals_95": intervals,
        "response_validation_margin": validation_margin,
        "max_holdout_response_error": max_holdout_response_error,
        "columns_overlap_within_95": all_overlap if len(stars) >= 2 else False,
        "column_overlap_by_aperture": primary_overlap,
        "pairwise_column_overlap": pairwise_overlap,
        "holdout_validation": holdout,
        "max_candidate_nights": MAX_CANDIDATE_NIGHTS,
    }


def write_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p
