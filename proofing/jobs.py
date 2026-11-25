"""Job specification, print recipes ("印法") and plate geometry.

Geometry convention
-------------------
* The sheet is a ``width x height`` grid of cells, indices ``(row, col)``.
* A *plate* belongs to one ink and carries a coverage map over the design
  cells.  With at most three inks (A, B, S) at most three plates are made.
* A *recipe* ("印法") is an ordered tuple of inks in **deposit order**
  (first element is laid down first / sits underneath).  Two process inks give
  the familiar 正序 / 反序 pair; the spot recipe is a single plate.

Registration offset
-------------------
A per-plate integer shift in cells, ``delta``.  ``delta = +1`` means the plate
is moved one cell to the **right**, so ink intended for design column ``c``
lands on sheet column ``c + 1``.  The allowed offset set is a job parameter
(``offsets`` below) because a small "no misregistration" order has only the
all-zero offset while an edge-critical order must be checked against a range.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

Cell = Tuple[int, int]

# --------------------------------------------------------------------------- #
# recipes / 印法
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Recipe:
    name: str
    inks: Tuple[str, ...]        # deposit order: first = under

    @property
    def deposit_order_name(self) -> str | None:
        """For a two-process recipe, the deposit-order key used by the model."""
        if len(self.inks) == 2 and set(self.inks) == {"A", "B"}:
            return self.inks[0] + self.inks[1]
        return None

    @property
    def uses_spot(self) -> bool:
        return "S" in self.inks


RECIPES: Dict[str, Recipe] = {
    # 正序: B laid first (under), A on top  -> on Paper1 reaches 0.90
    "forward": Recipe("forward", ("B", "A")),
    # 反序: A first (under), B on top        -> on Paper1 reaches 0.70
    "reverse": Recipe("reverse", ("A", "B")),
    # 专色: single spot plate                -> reaches 0.90 but costs more
    "spot": Recipe("spot", ("S",)),
    # A three-plate "full house" used to exercise the 3-layer scrap rule.
    "triple": Recipe("triple", ("B", "A", "S")),
}

# The small order is restricted to exactly these three printing methods.
SMALL_ORDER_RECIPES = ("forward", "reverse", "spot")


# --------------------------------------------------------------------------- #
# job
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    name: str
    paper: str
    width: int
    height: int
    coverage: Dict[str, Dict[Cell, float]]
    targets: Dict[Cell, Tuple[List[float], float]]
    recipes_allowed: List[str]
    locked_order: str | None
    offsets: dict
    costs: Dict[str, float]
    equal_channels: List[int] = field(default_factory=list)  # channels that must agree

    # -- derived ----------------------------------------------------------- #
    def target_cells(self) -> List[Cell]:
        return list(self.targets.keys())

    def recipe(self, name: str) -> Recipe:
        return RECIPES[name]

    def recipe_allowed(self, name: str) -> bool:
        if name not in self.recipes_allowed:
            return False
        rec = RECIPES[name]
        if self.locked_order is None:
            return True
        if rec.uses_spot or rec.deposit_order_name is None:
            return True
        return rec.deposit_order_name == self.locked_order

    def recipe_cost(self, name: str) -> float:
        rec = RECIPES[name]
        base = len(rec.inks) * self.costs.get("plate", 0.0)
        if rec.uses_spot:
            base += self.costs.get("spot", 0.0)
        return base


# --------------------------------------------------------------------------- #
# construction from JSON spec
# --------------------------------------------------------------------------- #
def _fill_rect(cov: Dict[Cell, float], rect: Sequence, width: int, height: int) -> None:
    r0, c0, r1, c1 = rect[:4]
    val = float(rect[4]) if len(rect) > 4 else 1.0
    for r in range(max(0, r0), min(height - 1, r1) + 1):
        for c in range(max(0, c0), min(width - 1, c1) + 1):
            cov[(r, c)] = val


def job_from_dict(spec: dict) -> Job:
    width = int(spec["width"])
    height = int(spec["height"])
    coverage: Dict[str, Dict[Cell, float]] = {}
    for ink, rects in spec.get("rects", {}).items():
        cov: Dict[Cell, float] = {}
        for rect in rects:
            _fill_rect(cov, rect, width, height)
        coverage[ink] = cov
    # explicit per-cell coverage overrides
    for ink, cds in spec.get("cells", {}).items():
        cov = coverage.setdefault(ink, {})
        for key, val in cds.items():
            r, c = (int(x) for x in key.split(","))
            cov[(r, c)] = float(val)

    targets: Dict[Cell, Tuple[List[float], float]] = {}
    for tr in spec.get("target_rects", []):
        rect = tr["rect"]
        density = [float(x) for x in tr["density"]]
        tol = float(tr.get("tol", 0.02))
        r0, c0, r1, c1 = rect[:4]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                targets[(r, c)] = (density, tol)
    for tc in spec.get("target_cells", []):
        r, c = tc["cell"]
        targets[(int(r), int(c))] = ([float(x) for x in tc["density"]],
                                     float(tc.get("tol", 0.02)))

    return Job(
        name=spec["name"],
        paper=spec["paper"],
        width=width,
        height=height,
        coverage=coverage,
        targets=targets,
        recipes_allowed=list(spec.get("recipes_allowed", SMALL_ORDER_RECIPES)),
        locked_order=spec.get("locked_order"),
        offsets=spec.get("offsets", {"mode": "zero"}),
        costs={"plate": float(spec.get("costs", {}).get("plate", 100.0)),
               "spot": float(spec.get("costs", {}).get("spot", 250.0))},
        equal_channels=[int(x) for x in spec.get("equal_channels", [])],
    )


def load_job(path: str) -> Job:
    with open(path, encoding="utf-8") as fh:
        return job_from_dict(json.load(fh))


# --------------------------------------------------------------------------- #
# offset enumeration
# --------------------------------------------------------------------------- #
def enumerate_offsets(job: Job, recipe: Recipe) -> List[Dict[str, int]]:
    spec = job.offsets
    mode = spec.get("mode", "zero")
    inks = recipe.inks
    if mode == "zero":
        return [{ink: 0 for ink in inks}]
    if mode == "all":
        rng = spec.get("range", [-1, 0, 1])
        return [dict(zip(inks, combo)) for combo in itertools.product(rng, repeat=len(inks))]
    if mode == "list":
        out = []
        for d in spec.get("values", []):
            out.append({ink: int(d.get(ink, 0)) for ink in inks})
        # de-duplicate while preserving order
        seen, uniq = set(), []
        for o in out:
            key = tuple(sorted(o.items()))
            if key not in seen:
                seen.add(key)
                uniq.append(o)
        return uniq
    raise ValueError(f"unknown offsets mode {mode!r}")


# --------------------------------------------------------------------------- #
# example job specs
# --------------------------------------------------------------------------- #
def example_job_specs() -> Dict[str, dict]:
    """The example jobs described in the task.

    The wide block's centre target is density 0.90 in every channel with
    tolerance 0.02.  ``Paper1`` reaches it with the cheap 正序; ``Paper2`` only
    with the spot plate; locking the press to 反序 also forces the spot.
    """
    block_target = {"rect": [0, 0, 2, 4], "density": [0.90, 0.90, 0.90], "tol": 0.02}

    def wide(name, paper, **over):
        spec = {
            "name": name,
            "paper": paper,
            "width": 9,
            "height": 3,
            "rects": {"A": [[0, 2, 2, 6]], "B": [[0, 2, 2, 6]], "S": [[0, 2, 2, 6]]},
            "target_rects": [{"rect": [0, 2, 2, 6], "density": [0.90, 0.90, 0.90], "tol": 0.02}],
            "recipes_allowed": list(SMALL_ORDER_RECIPES),
            "offsets": {"mode": "zero"},
            "costs": {"plate": 100.0, "spot": 250.0},
            # "the other two channels are equal": channels 1 and 2 must agree
            "equal_channels": [1, 2],
        }
        spec.update(over)
        return spec

    jobs = {
        # 1. clean sheet, no misregistration -> cheap 正序
        "wide_block": wide("wide_block", "Paper1"),
        # 2. press locked to 反序 -> reverse only gives 0.70 -> spot
        "wide_block_locked_reverse": wide(
            "wide_block_locked_reverse", "Paper1", locked_order="AB"),
        # 3. new paper: 正序 only reaches 0.75 -> spot
        "wide_block_paper2": wide("wide_block_paper2", "Paper2"),
        # 4. five-cell-wide edge image, every +/-1 per-plate offset -> edge failures
        "edge_image": wide("edge_image", "Paper1", offsets={"mode": "all", "range": [-1, 0, 1]}),
        # 5. same edge image but registration known-zero -> 正序 passes every cell
        "edge_image_locked_zero": wide("edge_image_locked_zero", "Paper1",
                                       offsets={"mode": "zero"}),
        # 6. three-plate recipe is the only option and stacks 3 layers -> scrap
        "scrap_demo": wide("scrap_demo", "Paper1", recipes_allowed=["triple"]),
        # 7. three-plate vs spot: planner drops the scrapping triple, keeps spot
        "triple_demo": wide("triple_demo", "Paper1",
                            recipes_allowed=["triple", "spot"]),
        # 8. intermediate-coverage tone ramp: targets are the *learned* forward
        #    curve at several coverages; only the forward recipe reproduces it,
        #    so this proves the planner uses the intermediate response (not just
        #    the full-coverage endpoint).
        "tone_ramp": _tone_ramp_spec(),
    }
    return jobs


def _tone_ramp_spec() -> dict:
    """A row of patches at increasing coverage, both process plates inked to
    the same coverage in each patch.  Targets are the *forward* forward (BA)
    densities at those intermediate coverages; only the forward recipe hits
    them.  (Values are the Paper1 forward curve, rounded to 3 dp.)"""
    coverages = [0.2, 0.35, 0.5, 0.65, 0.8]
    forward_d = {
        0.20: [0.148, 0.150, 0.147],
        0.35: [0.256, 0.257, 0.254],
        0.50: [0.361, 0.363, 0.359],
        0.65: [0.471, 0.473, 0.469],
        0.80: [0.602, 0.604, 0.601],
    }
    rects = {"A": [], "B": [], "S": []}
    target_cells = []
    for i, c in enumerate(coverages):
        for ink in ("A", "B", "S"):
            rects[ink].append([0, i, 0, i, c])
        target_cells.append({"cell": [0, i], "density": forward_d[c], "tol": 0.02})
    return {
        "name": "tone_ramp",
        "paper": "Paper1",
        "width": len(coverages),
        "height": 1,
        "rects": rects,
        "target_cells": target_cells,
        "recipes_allowed": list(SMALL_ORDER_RECIPES),
        "offsets": {"mode": "zero"},
        "costs": {"plate": 100.0, "spot": 250.0},
    }


def write_example_jobs(outdir: str) -> List[str]:
    import os
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for name, spec in example_job_specs().items():
        p = os.path.join(outdir, f"{name}.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2, sort_keys=True)
        paths.append(p)
    return paths
