"""POSTER PROOFING — learned colour-ramp response + print-order planner.

See ``README.md`` for the full task description, the two routes, and how to
swap in a new input of the same format.
"""

from .model import Response, fit_all, fit_paper
from .jobs import Job, Recipe, load_job, job_from_dict, RECIPES
from .render import evaluate, render_sheet
from .planner import Plan, plan_direct

__all__ = [
    "Response", "fit_all", "fit_paper",
    "Job", "Recipe", "load_job", "job_from_dict", "RECIPES",
    "evaluate", "render_sheet",
    "Plan", "plan_direct",
]
