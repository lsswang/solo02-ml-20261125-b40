from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Sequence

from .grid import DELTAS
from .model import FEATURES

# Four reproducible polarization-channel signatures for directed slopes.
# The table is synthetic, not a physical instrument model; it gives the
# calibrator separated but overlapping class clouds on an ordinary CPU.
CHANNEL_CENTERS: dict[int, tuple[float, float, float, float]] = {
    -2: (-2.10, -1.35, 1.75, 0.55),
    -1: (-1.05, -0.62, 0.88, 0.24),
    0: (0.00, 0.08, 0.00, -0.06),
    1: (1.02, 0.58, -0.92, -0.20),
    2: (2.05, 1.28, -1.72, -0.48),
}


def read_channels(row: dict) -> list[float]:
    return [float(row[c]) for c in FEATURES]


def make_reading(delta: int, rng: random.Random, noise: float = 0.34) -> list[float]:
    center = CHANNEL_CENTERS[delta]
    return [v + rng.gauss(0.0, noise) for v in center]


def blend_reading(weights: dict[int, float]) -> list[float]:
    """Noise-free mixture of class signatures for hand-authored scan fixtures."""
    total_w = sum(weights.values())
    out = [0.0, 0.0, 0.0, 0.0]
    for d, w in weights.items():
        for j, v in enumerate(CHANNEL_CENTERS[d]):
            out[j] += (w / total_w) * v
    return [round(v, 6) for v in out]


def edge_identity(rows: int, cols: int, r: int, c: int, kind: str) -> tuple[str, int, int, int, int]:
    """Return 1-based edge id and directed endpoints.

    Horizontal edges go left-to-right; vertical edges go top-to-bottom.
    ``r,c`` are the first endpoint's 1-based coordinates.
    """
    if kind == "H":
        return f"H-r{r}-c{c}", r, c, r, c + 1
    return f"V-r{r}-c{c}", r, c, r + 1, c


def grid_edges(rows: int, cols: int) -> list[tuple[str, int, int, int, int]]:
    out = []
    for r in range(1, rows + 1):
        for c in range(1, cols):
            out.append(edge_identity(rows, cols, r, c, "H"))
    for r in range(1, rows):
        for c in range(1, cols + 1):
            out.append(edge_identity(rows, cols, r, c, "V"))
    return out


def write_scan_csv(path: Path, scan_id: str, rows: int, cols: int,
                   readings: dict[str, Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scan_id", "rows", "cols", "edge_id", "u_row", "u_col", "v_row", "v_col", *FEATURES]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for edge_id, ur, uc, vr, vc in grid_edges(rows, cols):
            x = readings[edge_id]
            w.writerow({
                "scan_id": scan_id, "rows": rows, "cols": cols, "edge_id": edge_id,
                "u_row": ur, "u_col": uc, "v_row": vr, "v_col": vc,
                FEATURES[0]: f"{x[0]:.6f}", FEATURES[1]: f"{x[1]:.6f}",
                FEATURES[2]: f"{x[2]:.6f}", FEATURES[3]: f"{x[3]:.6f}",
            })


def random_smooth_heights(rows: int, cols: int, rng: random.Random) -> list[list[int]]:
    """Random 0..3 plate with every adjacent difference in calibrated -2..2."""
    h = [[1] * cols for _ in range(rows)]
    h[0][0] = rng.choice([0, 1, 1, 2, 2, 3])
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                continue
            neighbors = []
            if c > 0:
                neighbors.append(h[r][c - 1])
            if r > 0:
                neighbors.append(h[r - 1][c])
            allowed = [
                value for value in range(4)
                if all(abs(value - old) <= 2 for old in neighbors)
            ]
            base = rng.choice(neighbors)
            preferred = min(3, max(0, base + rng.choice([-1, 0, 0, 1])))
            if preferred in allowed and rng.random() < 0.75:
                h[r][c] = preferred
            else:
                h[r][c] = rng.choice(allowed)
    return h


def heights_to_readings(h: list[list[int]], rng: random.Random,
                        noise: float = 0.30) -> dict[str, list[float]]:
    rows, cols = len(h), len(h[0])
    readings: dict[str, list[float]] = {}
    for edge_id, ur, uc, vr, vc in grid_edges(rows, cols):
        d = h[vr - 1][vc - 1] - h[ur - 1][uc - 1]
        # Smooth generator normally stays in the calibrated -2..2 band.
        d = max(-2, min(2, d))
        readings[edge_id] = make_reading(d, rng, noise=noise)
    return readings


def generate_project(outdir: str | Path, seed: int = 15,
                     scan_rows: int = 2, scan_cols: int = 4) -> dict[str, Path]:
    """Create small training coupons, one whole held-out coupon and a new scan."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    train_path = out / "calibration_train.csv"
    holdout_path = out / "calibration_holdout.csv"
    n_train = 700
    with train_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "coupon_id", "delta", *FEATURES]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(n_train):
            d = DELTAS[i % len(DELTAS)] if i < len(DELTAS) * 4 else rng.choice(DELTAS)
            x = make_reading(d, rng)
            w.writerow({"sample_id": f"S{i:04d}", "coupon_id": f"C{i // 30:03d}",
                        "delta": d, **{FEATURES[j]: f"{x[j]:.6f}" for j in range(4)}})

    holdout_h = random_smooth_heights(3, 4, rng)
    holdout_rows = []
    for edge_id, ur, uc, vr, vc in grid_edges(3, 4):
        d = holdout_h[vr - 1][vc - 1] - holdout_h[ur - 1][uc - 1]
        d = max(-2, min(2, d))
        x = make_reading(d, rng, noise=0.30)
        holdout_rows.append({
            "sample_id": edge_id, "coupon_id": "HELDOUT-3X4", "delta": d,
            **{FEATURES[j]: f"{x[j]:.6f}" for j in range(4)},
        })
    with holdout_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "coupon_id", "delta", *FEATURES])
        w.writeheader()
        w.writerows(holdout_rows)

    scan_h = random_smooth_heights(scan_rows, scan_cols, rng)
    scan_readings = heights_to_readings(scan_h, rng)
    scan_path = out / "new_scan.csv"
    write_scan_csv(scan_path, "NEW-SCAN", scan_rows, scan_cols, scan_readings)

    truth_path = out / "new_scan_true_heights.csv"
    with truth_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "true_height"])
        for r in range(scan_rows):
            for c in range(scan_cols):
                w.writerow([r + 1, c + 1, scan_h[r][c]])

    order_path = out / "new_scan_work_order.json"
    import json
    holes = [{"row": 1, "col": 1, "height": scan_h[0][0]}]
    if scan_rows > 1:
        holes.append({"row": scan_rows, "col": 1, "height": scan_h[-1][0]})
    order = {
        "order_id": f"WO-GENERATED-{scan_rows}X{scan_cols}",
        "rows": scan_rows,
        "cols": scan_cols,
        "holes": holes,
        "gauge": {},
        "method_policy": "any",
        "seam_columns": list(range(2, scan_cols + 1)),
        "contingencies": [],
        "note": "Generated work order. Holes copy true left-edge heights only to guarantee a feasible smoke test.",
    }
    order_path.write_text(json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"train": train_path, "holdout": holdout_path, "scan": scan_path,
            "truth": truth_path, "order": order_path}
