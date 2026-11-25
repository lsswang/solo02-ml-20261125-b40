from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .calibration import APERTURE_ORDER, STAR_ORDER, write_json


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def depth_interval_for_channel(
    y: float, error: float, r_lo: float, r_hi: float, eps: float = 1e-9
) -> tuple[float, float] | None:
    """Possible non-negative depth interval for one bounded channel.

    There must exist r in [r_lo, r_hi] and depth d >= 0 such that
    |y - d*r| <= error.  Returns (low, high), with math.inf as high when the
    response lower bound is zero.  Returns None when impossible.
    """
    if error < 0:
        raise ValueError("error bound must be non-negative")
    if r_lo < -eps or r_hi + eps < r_lo:
        raise ValueError("response interval must satisfy 0 <= r_lo <= r_hi")
    r_lo = max(0.0, r_lo)
    r_hi = max(r_lo, r_hi)
    lower_target = y - error
    upper_target = y + error

    if upper_target < -eps:
        return None
    if r_hi <= eps:
        # Zero response means this channel cannot see any depth.  It is only
        # compatible if the allowed target interval contains zero.
        if lower_target <= eps <= upper_target + eps:
            return (0.0, math.inf)
        return None

    low = max(0.0, lower_target / r_hi) if lower_target > 0 else 0.0
    if upper_target < 0:
        return None
    if r_lo <= eps:
        high = math.inf
    else:
        high = upper_target / r_lo
    if low <= high + eps:
        return (low, high)
    return None


def intersect_intervals(intervals: list[tuple[float, float]], eps: float = 1e-8) -> tuple[float, float] | None:
    low = max(iv[0] for iv in intervals)
    high = min(iv[1] for iv in intervals)
    if low <= high + eps:
        return (max(0.0, low), high)
    return None


def response_bounds(
    calibration: dict[str, Any], star: str, uncertainty: str
) -> tuple[list[float], list[float]]:
    central = calibration["responses"][star]
    if uncertainty == "none":
        return [central[a] for a in APERTURE_ORDER], [central[a] for a in APERTURE_ORDER]
    if uncertainty == "se95":
        iv = calibration["response_intervals_95"][star]
        return [iv[a][0] for a in APERTURE_ORDER], [iv[a][1] for a in APERTURE_ORDER]
    if uncertainty == "validation":
        # Central +/- (statistical half-width + whole-level holdout margin).
        margins = calibration["response_validation_margin"][star]
        lo: list[float] = []
        hi: list[float] = []
        for a in APERTURE_ORDER:
            r = central[a]
            m = margins[a]
            lo.append(max(0.0, r - m))
            hi.append(r + m)
        return lo, hi
    raise ValueError(f"unknown response uncertainty mode: {uncertainty}")


def evaluate_observation(
    calibration: dict[str, Any],
    drops: list[float] | tuple[float, float],
    errors: list[float] | tuple[float, float] | float,
    uncertainty: str = "se95",
) -> dict[str, Any]:
    if len(drops) != 2:
        raise ValueError("this instrument model expects exactly two aperture drops")
    if isinstance(errors, (int, float)):
        error_list = [float(errors), float(errors)]
    else:
        error_list = [float(e) for e in errors]
        if len(error_list) != 2:
            raise ValueError("expected two error bounds or one common bound")

    result: dict[str, Any] = {
        "schema": "observatory-query-result-v1",
        "drops": [float(drops[0]), float(drops[1])],
        "error_bounds": error_list,
        "response_uncertainty": uncertainty,
        "stars": {},
        "supported_stars": [],
    }

    star_names = list(calibration.get("responses", {}).keys())
    for star in star_names:
        r_lo, r_hi = response_bounds(calibration, star, uncertainty)
        channel_intervals = []
        channel_details = []
        possible = True
        for k, aperture in enumerate(APERTURE_ORDER):
            iv = depth_interval_for_channel(drops[k], error_list[k], r_lo[k], r_hi[k])
            if iv is None:
                possible = False
                channel_details.append(
                    {
                        "aperture": aperture,
                        "possible": False,
                        "response_interval": [r_lo[k], r_hi[k]],
                    }
                )
                break
            channel_intervals.append(iv)
            channel_details.append(
                {
                    "aperture": aperture,
                    "possible": True,
                    "response_interval": [r_lo[k], r_hi[k]],
                    "depth_interval": [iv[0], None if math.isinf(iv[1]) else iv[1]],
                }
            )
        depth_interval = intersect_intervals(channel_intervals) if possible else None
        if depth_interval is None:
            possible = False
        star_result: dict[str, Any] = {
            "possible": possible,
            "response_central": calibration["responses"][star],
            "channel_details": channel_details,
        }
        if possible and depth_interval is not None:
            lo, hi = depth_interval
            star_result["depth_interval"] = [lo, None if math.isinf(hi) else hi]
            star_result["depth_interval_text"] = (
                f"[{lo:.4g}, +inf)" if math.isinf(hi) else f"[{lo:.4g}, {hi:.4g}]"
            )
            result["supported_stars"].append(star)
        else:
            star_result["depth_interval"] = None
            star_result["reason"] = "two aperture depth intervals do not intersect within bounds"
        result["stars"][star] = star_result

    result["exactly_one_supported"] = len(result["supported_stars"]) == 1
    result["columns_overlap_within_calibration_errors"] = calibration.get(
        "columns_overlap_within_95"
    )
    return result


def run_query_file(
    calibration_path: str | Path,
    observation_path: str | Path,
    output_path: str | Path | None = None,
    uncertainty_override: str | None = None,
) -> dict[str, Any]:
    calibration = load_json(calibration_path)
    obs = load_json(observation_path)
    uncertainty = uncertainty_override or obs.get("response_uncertainty", "se95")
    result = evaluate_observation(
        calibration=calibration,
        drops=obs["drops"],
        errors=obs.get("error_bounds", obs.get("errors", 0.0)),
        uncertainty=uncertainty,
    )
    result["observation_id"] = obs.get("observation_id")
    result["observation_file"] = str(Path(observation_path))
    result["calibration_file"] = str(Path(calibration_path))
    if output_path:
        write_json(result, output_path)
    return result
