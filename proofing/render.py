"""Sheet rendering under a registration offset, and compliance checking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .jobs import Cell, Job, Recipe, enumerate_offsets
from .model import CHANNELS, Response

SCRAP_LAYERS = 3


@dataclass
class CellResult:
    cell: Cell
    layers: List[Tuple[str, float]]
    density: Optional[List[float]]      # None => scrapped (>=3 layers)
    per_channel_err: Optional[List[float]]
    ok: bool
    worst_channel: Optional[int] = None


@dataclass
class OffsetResult:
    offset: Dict[str, int]
    scrap_cells: List[Cell]
    target_cells: Dict[Cell, CellResult]
    worst_relief: float                  # max( |err| - tol ); <=0 means all pass
    worst_cell: Optional[Cell] = None
    worst_channel: Optional[int] = None
    equal_channel_gap: float = 0.0
    equal_channels_ok: bool = True


@dataclass
class Evaluation:
    recipe: str
    offsets: List[OffsetResult]
    passes: bool
    scrap: bool
    worst_relief: float
    worst_offset: Optional[Dict[str, int]] = None
    worst_cell: Optional[Cell] = None
    worst_channel: Optional[int] = None
    equal_channel_gap: float = 0.0
    equal_channels_ok: bool = True

    def summary(self) -> dict:
        return {
            "recipe": self.recipe,
            "passes": self.passes,
            "scrap": self.scrap,
            "worst_relief": self.worst_relief,
            "worst_offset": self.worst_offset,
            "worst_cell": self.worst_cell,
            "worst_channel": self.worst_channel,
            "equal_channel_gap": self.equal_channel_gap,
            "equal_channels_ok": self.equal_channels_ok,
        }


def _layers_at(job: Job, recipe: Recipe, offset: Dict[str, int], r: int, c: int):
    layers: List[Tuple[str, float]] = []
    for ink in recipe.inks:              # keep deposit order
        cov_map = job.coverage.get(ink)
        if not cov_map:
            continue
        d = offset.get(ink, 0)
        cov = cov_map.get((r, c - d), 0.0)   # shift right by d
        if cov > 0.0:
            layers.append((ink, cov))
    return layers


def _cell_density(resp: Response, layers: List[Tuple[str, float]]):
    if len(layers) >= SCRAP_LAYERS:
        return None
    return resp.density_vector(layers)


def render_sheet(job: Job, resp: Response, recipe: Recipe,
                 offset: Dict[str, int]) -> OffsetResult:
    scrap_cells: List[Cell] = []
    # whole-sheet scrap scan (a place stacked to three layers is scrapped)
    for r in range(job.height):
        for c in range(job.width):
            if len(_layers_at(job, recipe, offset, r, c)) >= SCRAP_LAYERS:
                scrap_cells.append((r, c))

    equal_ch = job.equal_channels
    target_cells: Dict[Cell, CellResult] = {}
    worst_relief = -1e9
    worst_cell = None
    worst_channel = None
    eq_gap = 0.0
    eq_ok = True
    for cell, (density, tol) in job.targets.items():
        layers = _layers_at(job, recipe, offset, *cell)
        pred = _cell_density(resp, layers)
        if pred is None:
            target_cells[cell] = CellResult(cell, layers, None, None, False, None)
            eq_ok = False
            if 1e9 > worst_relief:
                worst_relief = 1e9
                worst_cell, worst_channel = cell, None
            continue
        errs = [pred[ch] - density[ch] for ch in CHANNELS]
        relief = max(abs(errs[ch]) - tol for ch in CHANNELS)
        ok = relief <= 1e-12
        wc = max(CHANNELS, key=lambda ch: abs(errs[ch]))
        if equal_ch:
            gap = max(pred[ch] for ch in equal_ch) - min(pred[ch] for ch in equal_ch)
            eq_gap = max(eq_gap, gap)
            if gap > tol:
                ok = False
                eq_ok = False
                relief = max(relief, gap - tol)
        target_cells[cell] = CellResult(cell, layers, pred, errs, ok, wc)
        if relief > worst_relief:
            worst_relief = relief
            worst_cell, worst_channel = cell, wc

    return OffsetResult(
        offset=dict(offset),
        scrap_cells=scrap_cells,
        target_cells=target_cells,
        worst_relief=worst_relief,
        worst_cell=worst_cell,
        worst_channel=worst_channel,
        equal_channel_gap=eq_gap,
        equal_channels_ok=eq_ok,
    )


def evaluate(job: Job, resp: Response, recipe_name: str) -> Evaluation:
    recipe = job.recipe(recipe_name)
    offs = enumerate_offsets(job, recipe)
    results = [render_sheet(job, resp, recipe, o) for o in offs]

    scrap = any(r.scrap_cells for r in results)
    passes = (not scrap) and all(r.worst_relief <= 1e-12 for r in results)

    worst = max(results, key=lambda r: (1e9 if r.scrap_cells else r.worst_relief))
    relief = 1e9 if worst.scrap_cells else worst.worst_relief
    return Evaluation(
        recipe=recipe_name,
        offsets=results,
        passes=passes,
        scrap=scrap,
        worst_relief=relief,
        worst_offset=worst.offset,
        worst_cell=worst.worst_cell if not worst.scrap_cells else (worst.scrap_cells[0] if worst.scrap_cells else None),
        worst_channel=worst.worst_channel,
        equal_channel_gap=max(r.equal_channel_gap for r in results),
        equal_channels_ok=all(r.equal_channels_ok for r in results),
    )
