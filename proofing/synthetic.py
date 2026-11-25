"""Deterministic synthetic "material" for the proofing project.

We *pretend* two sheets of paper accumulated colour-ramp step wedges:

* single-ink ramps for inks A, B (process) and S (spot);
* dual-ink ramps in both print orders ``A_over_B`` and ``B_over_A``.

The hidden truth follows the same physical form the learner uses
(:mod:`proofing.model`), with per-paper, per-channel dot gain and per-order,
per-channel ink-interaction.  Full-coverage densities are tuned so that

===========  =========  =========  ========
paper        forward    reverse    spot
             (A over B) (B over A)
===========  =========  =========  ========
Paper1       0.90       0.70       0.90
Paper2       0.73       0.70       0.90
===========  =========  =========  ========

Target ``0.90`` with tolerance ``0.02`` therefore admits forward on Paper1 and
spot on either paper, but *not* forward on Paper2 and *not* reverse anywhere.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .model import CHANNELS, INKS, ORDERS, Response, dual_D, _nelder_mead

# --------------------------------------------------------------------------- #
# hidden truth
# --------------------------------------------------------------------------- #
# ``D_forward`` is the full-coverage density of the "正序" recipe = A pressed
# onto B, i.e. deposit order BA (B laid down first / under, A on top).  ``D_reverse``
# is the full-coverage density of the reverse recipe AB.  The ``k_*`` values are
# the ink-interaction params used when the solid densities are solved.
PAPER_SPEC = {
    "Paper1": dict(k_forward=0.10, k_reverse=0.06, D_forward=0.90, D_reverse=0.70, gain_scale=1.00),
    # New paper: the cheap forward order only reaches 0.75 -> too far from the
    # 0.90 target, so the spot recipe is the only compliant choice.
    "Paper2": dict(k_forward=0.06, k_reverse=0.08, D_forward=0.75, D_reverse=0.70, gain_scale=1.55),
}
BASE_GAIN = {"A": 0.055, "B": 0.050, "S": 0.040}
# Per-channel dot-gain factors: at *full* coverage phi(1,g)=1 so density is
# channel-neutral (the wide-block target is achromatic), but the intermediate
# ramps genuinely differ per channel, so a 3-channel fit is required.
CH_GAIN_FACTOR = {0: 1.00, 1: 1.20, 2: 0.82}
SPOT_SOLID = 0.90

# Deposit-order convention used throughout: "BA" = B first (under) then A on
# top = A pressing B = 正序; "AB" = A first then B = 反序.
FORWARD_ORDER = "BA"
REVERSE_ORDER = "AB"

TRAIN_SINGLE_COV = [0.0, 0.25, 0.5, 0.75, 1.0]
HOLDOUT_SINGLE_COV = [0.125, 0.375, 0.625, 0.875]
TRAIN_DUAL_PAIRS = [(cu, cv) for cu in (0.0, 0.5, 1.0) for cv in (0.0, 0.5, 1.0)]
HOLDOUT_DUAL_PAIRS = [(cu, cv) for cu in (0.125, 0.375, 0.625, 0.875)
                      for cv in (0.125, 0.375, 0.625, 0.875)]
NOISE_SIGMA = 0.0015


def _solve_solids(D_forward: float, D_reverse: float,
                  k_forward: float, k_reverse: float) -> Tuple[float, float]:
    """Find (sA, sB) matching forward (A over B) and reverse (B over A) full densities."""
    def errs(p):
        sA, sB = p
        if not (0.001 <= sA <= 3.5 and 0.001 <= sB <= 3.5):
            return 1e12 + sA * sA + sB * sB
        # forward: under = B (sB), over = A (sA)
        f = dual_D(1.0, 1.0, sB, sA, 0.0, 0.0, k_forward) - D_forward
        # reverse: under = A (sA), over = B (sB)
        r = dual_D(1.0, 1.0, sA, sB, 0.0, 0.0, k_reverse) - D_reverse
        return f * f + r * r

    (sA, sB), _ = _nelder_mead(errs, [0.8, 0.8], steps=[0.1, 0.1], iters=2000)
    return sA, sB


def build_truth() -> Dict[str, Response]:
    return build_truth_from_spec(PAPER_SPEC)


def build_truth_from_spec(paper_spec: Dict[str, dict]) -> Dict[str, Response]:
    """Build the hidden truth for an arbitrary paper spec of the same format.

    Each entry needs ``k_forward``, ``k_reverse``, ``D_forward``, ``D_reverse``
    and ``gain_scale``.  This lets a verifier drop in a new material spec and
    regenerate ramps -> new deliverables without touching the code.
    """
    truth: Dict[str, Response] = {}
    for paper, spec in paper_spec.items():
        if paper.startswith("_") or not isinstance(spec, dict):
            continue  # skip comments / metadata
        sA, sB = _solve_solids(spec["D_forward"], spec["D_reverse"],
                               spec["k_forward"], spec["k_reverse"])
        g: Dict[str, Dict[int, float]] = {}
        for ink in INKS:
            g[ink] = {ch: BASE_GAIN[ink] * spec["gain_scale"] * CH_GAIN_FACTOR[ch]
                      for ch in CHANNELS}
        s = {ink: {ch: (SPOT_SOLID if ink == "S" else (sA if ink == "A" else sB))
                   for ch in CHANNELS} for ink in INKS}
        k = {
            # forward (正序, BA) and reverse (反序, AB) interaction params
            "BA": {ch: min(0.95, spec["k_forward"]) for ch in CHANNELS},
            "AB": {ch: min(0.95, spec["k_reverse"]) for ch in CHANNELS},
        }
        truth[paper] = Response(paper=paper, g=g, s=s, k=k)
    return truth


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def _noise(rng: random.Random) -> float:
    return rng.gauss(0.0, NOISE_SIGMA)


def sample_single(truth: Dict[str, Response]) -> List[dict]:
    rng = random.Random(20241125)
    rows: List[dict] = []
    for paper, resp in truth.items():
        for ink in INKS:
            for split, covs in (("train", TRAIN_SINGLE_COV), ("holdout", HOLDOUT_SINGLE_COV)):
                for c in covs:
                    ds = [resp.single_density(ink, c, ch) + _noise(rng) for ch in CHANNELS]
                    rows.append(dict(paper=paper, ink=ink, coverage=c,
                                     d0=ds[0], d1=ds[1], d2=ds[2], split=split))
    return rows


def sample_dual(truth: Dict[str, Response]) -> List[dict]:
    rng = random.Random(20241126)
    rows: List[dict] = []
    for paper, resp in truth.items():
        for order in ORDERS:
            for split, pairs in (("train", TRAIN_DUAL_PAIRS), ("holdout", HOLDOUT_DUAL_PAIRS)):
                for cu, cv in pairs:
                    ds = [resp.dual_density(order, cu, cv, ch) + _noise(rng) for ch in CHANNELS]
                    rows.append(dict(paper=paper, order=order, cu=cu, cv=cv,
                                     d0=ds[0], d1=ds[1], d2=ds[2], split=split))
    return rows


# --------------------------------------------------------------------------- #
# CSV I/O
# --------------------------------------------------------------------------- #
SINGLE_FIELDS = ["paper", "ink", "coverage", "d0", "d1", "d2", "split"]
DUAL_FIELDS = ["paper", "order", "cu", "cv", "d0", "d1", "d2", "split"]


def _write_csv(path: str, fields: List[str], rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in fields})


def write_dataset(outdir: str, truth: Dict[str, Response]) -> Dict[str, str]:
    single = sample_single(truth)
    dual = sample_dual(truth)
    os.makedirs(outdir, exist_ok=True)
    sp = os.path.join(outdir, "ramps_single.csv")
    dp = os.path.join(outdir, "ramps_dual.csv")
    _write_csv(sp, SINGLE_FIELDS, single)
    _write_csv(dp, DUAL_FIELDS, dual)
    tp = os.path.join(outdir, "truth.json")
    with open(tp, "w", encoding="utf-8") as fh:
        json.dump({p: r.to_dict() for p, r in truth.items()}, fh, indent=2, sort_keys=True)
    return {"single": sp, "dual": dp, "truth": tp}
