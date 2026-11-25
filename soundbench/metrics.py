"""Frame-wise and whole-signal error metrics."""

from __future__ import annotations

import csv
import math
from typing import Any, Dict, List, Sequence

from .linalg import max_abs, rms


def difference(a: Sequence[float], b: Sequence[float]) -> List[float]:
    n = min(len(a), len(b))
    return [a[i] - b[i] for i in range(n)]


def frame_errors(
    predicted: Sequence[Sequence[float]],
    measured: Sequence[Sequence[float]],
    sample_rate: int,
    frame_size: int = 64,
    hop_size: int = 32,
) -> List[Dict[str, float]]:
    """Return one RMSE row per overlapping analysis frame.

    The short synthetic signals use 64-sample frames by default.  The last
    incomplete frame is included so that tail errors are not hidden.
    """
    n = min(len(predicted[0]), len(measured[0]), len(predicted[1]), len(measured[1]))
    rows: List[Dict[str, float]] = []
    start = 0
    frame_index = 0
    while start < n:
        end = min(n, start + frame_size)
        row: Dict[str, float] = {
            "frame": frame_index,
            "start_sample": start,
            "end_sample": end,
            "start_second": round(start / sample_rate, 6),
        }
        for zone, pi, mi in (("near", 0, 0), ("far", 1, 1)):
            se = 0.0
            ae = 0.0
            count = end - start
            for i in range(start, end):
                e = predicted[pi][i] - measured[mi][i]
                se += e * e
                ae += abs(e)
            row[f"{zone}_rmse"] = math.sqrt(se / count)
            row[f"{zone}_mae"] = ae / count
        rows.append(row)
        frame_index += 1
        start += hop_size
    return rows


def write_frame_csv(path: str, rows: Sequence[Dict[str, float]]) -> None:
    fieldnames = [
        "frame",
        "start_sample",
        "end_sample",
        "start_second",
        "near_rmse",
        "near_mae",
        "far_rmse",
        "far_mae",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def signal_metrics(x: Sequence[float]) -> Dict[str, float]:
    return {"rms": rms(x), "peak": max_abs(x)}


def leakage_report(
    mic: Sequence[float],
    target: Sequence[float],
    sample_rate: int,
    window_start: int = 0,
    window_end: int | None = None,
) -> Dict[str, float]:
    """Compare a zone signal with its wanted target inside a window."""
    if window_end is None:
        window_end = min(len(mic), len(target))
    window_end = min(window_end, len(mic), len(target))
    got = mic[window_start:window_end]
    wanted = target[window_start:window_end]
    err = difference(got, wanted)
    return {
        "window_start": window_start,
        "window_end": window_end,
        "rms": rms(got),
        "peak": max_abs(got),
        "target_rms": rms(wanted),
        "target_peak": max_abs(wanted),
        "rmse": rms(err),
        "max_error": max_abs(err),
    }
