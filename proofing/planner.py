"""Print-order planner.

This implements **route 1**: *direct output*.  We pick a recipe (版 and 印序)
before loading paper, and require the sheet to be compliant under **every**
allowed per-plate offset.  The cost is just the plate cost (plus spot setup).

A second module, :mod:`proofing.trial`, implements route 2 (burn a trial proof
first); both return a :class:`Plan` so the CLI and reporting are shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .jobs import Job
from .model import Response
from .render import Evaluation, evaluate


@dataclass
class Plan:
    route: str
    recipe: Optional[str]
    cost: float
    feasible: bool
    evidence: dict = field(default_factory=dict)
    checks: Dict[str, Evaluation] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "recipe": self.recipe,
            "cost": self.cost,
            "feasible": self.feasible,
            "evidence": self.evidence,
            "checks": {k: v.summary() for k, v in self.checks.items()},
        }


def _recipe_order(recipe_name: str, job: Job) -> int:
    """Stable preference tie-break: cheaper / fewer plates first, then canonical order."""
    rec = job.recipe(recipe_name)
    return (job.recipe_cost(recipe_name), len(rec.inks), recipe_name)


def plan_direct(job: Job, resp: Response) -> Plan:
    """Route 1: choose the cheapest recipe compliant at *all* allowed offsets."""
    candidates: Dict[str, Evaluation] = {}
    for name in job.recipes_allowed:
        if not job.recipe_allowed(name):
            continue
        candidates[name] = evaluate(job, resp, name)

    if not candidates:
        return Plan(
            route="direct",
            recipe=None,
            cost=0.0,
            feasible=False,
            evidence={
                "reason": ("no recipe is admissible for this job "
                           "(check recipes_allowed / locked_order)"),
                "failures": {},
                "considered": {},
            },
            checks={},
        )

    feasible = [n for n, ev in candidates.items() if ev.passes]
    if feasible:
        best = min(feasible, key=lambda n: _recipe_order(n, job))
        cost = job.recipe_cost(best)
        return Plan(
            route="direct",
            recipe=best,
            cost=cost,
            feasible=True,
            evidence={
                "reason": "cheapest recipe passing every allowed offset",
                "considered": {n: ev.summary() for n, ev in candidates.items()},
            },
            checks={n: candidates[n] for n in candidates},
        )

    # No single recipe works under all offsets -> show why (scrap evidence).
    detail = {}
    any_scrap = False
    for n, ev in candidates.items():
        fails = []
        for off in ev.offsets:
            if off.scrap_cells:
                any_scrap = True
                fails.append({"offset": off.offset, "scrap_cells": off.scrap_cells,
                              "kind": "scrap_3_layers"})
            elif off.worst_relief > 1e-12:
                fails.append({"offset": off.offset, "worst_cell": off.worst_cell,
                              "worst_channel": off.worst_channel,
                              "relief": off.worst_relief, "kind": "out_of_tol"})
        detail[n] = fails
    return Plan(
        route="direct",
        recipe=None,
        cost=0.0,
        feasible=False,
        evidence={
            "reason": ("no single recipe is compliant under every allowed offset"
                       + ("; at least one stacks three layers (scrap)"
                          if any_scrap else "")),
            "failures": detail,
            "considered": {n: ev.summary() for n, ev in candidates.items()},
        },
        checks=candidates,
    )
