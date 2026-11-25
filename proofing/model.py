"""Learned colour-ramp response for a paper / print-order combination.

Physical-form model
--------------------
A single ink *u* at coverage ``c`` on a paper with dot-gain ``g`` transmits

    T_u(c) = 1 - phi(c, g) * (1 - 10**-s_u),      phi(c, g) = c + g*c*(1-c)

so the optical density is ``D = -log10(T)``.  ``s_u`` is the solid density
(reached at c=1, because phi(1,g)=1).

Two inks printed one over the other (``order`` says which ink is *under* and
which is *over*) multiply their transmissions through an interaction factor
``kappa`` that depends on the *under* ink's coverage:

    kappa(cu) = 1 - (1 - k) * phi(cu, g_under)
    T = T_under(cu) * [ 1 - phi(cv, g_over) * kappa(cu) * (1 - 10**-s_over) ]
    D = -log10(T)

``k`` is the ink-ink (or "coverage interaction") parameter and is learned
*per paper and per print order*, which is what makes the response vary with
paper, order and dual-ink coverage interaction.

Everything here is pure Python (no numpy) and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

INKS = ("A", "B", "S")                      # two process inks + one spot ink
# Orders are named by *deposit sequence*: "AB" means A is laid down first
# (under) then B on top; "BA" is the reverse.  We avoid the words
# "over/under" in identifiers so there is no ambiguity.
ORDERS = ("AB", "BA")
UNDER_OF = {"AB": "A", "BA": "B"}
OVER_OF = {"AB": "B", "BA": "A"}
CHANNELS = (0, 1, 2)


# --------------------------------------------------------------------------- #
# scalar physics
# --------------------------------------------------------------------------- #
def phi(c: float, g: float) -> float:
    c = min(max(c, 0.0), 1.0)
    v = c + g * c * (1.0 - c)
    return min(max(v, 0.0), 1.0)


def single_T(c: float, g: float, s: float) -> float:
    return 1.0 - phi(c, g) * (1.0 - 10.0 ** (-s))


def single_D(c: float, g: float, s: float) -> float:
    t = single_T(c, g, s)
    return -math.log10(min(max(t, 1e-12), 1.0))


def dual_D(cu: float, cv: float, s_under: float, s_over: float,
           g_under: float, g_over: float, k: float) -> float:
    t_under = single_T(cu, g_under, s_under)
    kappa = 1.0 - (1.0 - k) * phi(cu, g_under)
    t_over = 1.0 - phi(cv, g_over) * kappa * (1.0 - 10.0 ** (-s_over))
    t = t_under * t_over
    return -math.log10(min(max(t, 1e-12), 1.0))


# --------------------------------------------------------------------------- #
# tiny optimisers (pure Python)
# --------------------------------------------------------------------------- #
def _nelder_mead(f, x0, steps=0.1, iters=800, tol=1e-11):
    n = len(x0)
    if isinstance(steps, (int, float)):
        steps = [steps] * n
    sim = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += steps[i]
        sim.append(p)
    vals = [f(p) for p in sim]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        sim = [sim[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = [sum(sim[i][j] for i in range(n)) / n for j in range(n)]
        xr = [cen[j] + (cen[j] - sim[-1][j]) for j in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [cen[j] + 2.0 * (cen[j] - sim[-1][j]) for j in range(n)]
            fe = f(xe)
            sim[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            sim[-1], vals[-1] = xr, fr
        else:
            xc = [cen[j] + 0.5 * (sim[-1][j] - cen[j]) for j in range(n)]
            fc = f(xc)
            if fc < vals[-1]:
                sim[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    sim[i] = [(sim[i][j] + sim[0][j]) / 2.0 for j in range(n)]
                    vals[i] = f(sim[i])
    bi = min(range(n + 1), key=lambda i: vals[i])
    return sim[bi], vals[bi]


def _golden(f, lo, hi, iters=200):
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return (a + b) / 2.0


# --------------------------------------------------------------------------- #
# response object
# --------------------------------------------------------------------------- #
@dataclass
class Response:
    """A fully learned response for one paper."""
    paper: str
    g: Dict[str, Dict[int, float]] = field(default_factory=dict)   # ink -> ch -> gain
    s: Dict[str, Dict[int, float]] = field(default_factory=dict)   # ink -> ch -> solid
    k: Dict[str, Dict[int, float]] = field(default_factory=dict)   # order -> ch -> k

    # -- per-channel prediction ------------------------------------------- #
    def single_density(self, ink: str, c: float, ch: int) -> float:
        return single_D(c, self.g[ink][ch], self.s[ink][ch])

    def dual_density(self, order: str, cu: float, cv: float, ch: int) -> float:
        u, v = UNDER_OF[order], OVER_OF[order]
        return dual_D(cu, cv,
                      self.s[u][ch], self.s[v][ch],
                      self.g[u][ch], self.g[v][ch],
                      self.k[order][ch])

    def density_vector(self, layers: Sequence[Tuple[str, float]]) -> List[float]:
        """Density of a cell carrying ``layers`` = [(ink, coverage), ...].

        The caller guarantees len(layers) in {0, 1, 2}; three-or-more is a
        scrap condition handled by the planner.  For two process inks the
        result depends on the ink identity order (A under B vs B under A).
        """
        if len(layers) == 0:
            return [0.0, 0.0, 0.0]
        if len(layers) == 1:
            ink, c = layers[0]
            return [self.single_density(ink, c, ch) for ch in CHANNELS]
        (i1, c1), (i2, c2) = layers
        if {i1, i2} == {"A", "B"}:
            order = "AB" if i1 == "A" else "BA"
            return [self.dual_density(order, c1, c2, ch) for ch in CHANNELS]
        # e.g. spot next to a process ink: approximate as independent product
        out = []
        for ch in CHANNELS:
            t = 10.0 ** (-self.single_density(i1, c1, ch))
            t *= 10.0 ** (-self.single_density(i2, c2, ch))
            out.append(-math.log10(max(t, 1e-12)))
        return out

    def full_dual(self, order: str) -> List[float]:
        return [self.dual_density(order, 1.0, 1.0, ch) for ch in CHANNELS]

    def full_single(self, ink: str) -> List[float]:
        return [self.single_density(ink, 1.0, ch) for ch in CHANNELS]

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "paper": self.paper,
            "g": {i: {str(c): v for c, v in d.items()} for i, d in self.g.items()},
            "s": {i: {str(c): v for c, v in d.items()} for i, d in self.s.items()},
            "k": {o: {str(c): v for c, v in d.items()} for o, d in self.k.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Response":
        return cls(
            paper=d["paper"],
            g={i: {int(c): v for c, v in dd.items()} for i, dd in d["g"].items()},
            s={i: {int(c): v for c, v in dd.items()} for i, dd in d["s"].items()},
            k={o: {int(c): v for c, v in dd.items()} for o, dd in d["k"].items()},
        )


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def _rmse(errs: List[float]) -> float:
    return math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else 0.0


def fit_paper(paper: str, single_rows: List[dict], dual_rows: List[dict]) -> Tuple[Response, dict]:
    """Fit one paper's response from its single- and dual-ink ramp rows.

    A row is a dict with keys from the CSV schema:
      single: ink, coverage, d0, d1, d2, split
      dual  : order, cu, cv, d0, d1, d2, split
    """
    resp = Response(paper=paper)

    # ---- single-ink: fit (s, g) per ink per channel on training rows ----- #
    for ink in INKS:
        rows = [r for r in single_rows if r["ink"] == ink]
        resp.s[ink] = {}
        resp.g[ink] = {}
        for ch in CHANNELS:
            train = [(r["coverage"], r[f"d{ch}"]) for r in rows if r["split"] == "train"]
            if not train:
                # no data for this ink on this paper: neutral fallback
                resp.s[ink][ch] = 0.0
                resp.g[ink][ch] = 0.1
                continue
            # start the solid density at the observed density near full coverage
            s0 = max(max(d for c, d in train if c >= 0.75), 0.05)

            def obj(p, _train=train):
                s, g = p
                if not (0.001 <= s <= 3.5 and 0.0 <= g <= 1.2):
                    return 1e12 + (s * s + g * g)
                return sum((single_D(c, g, s) - d) ** 2 for c, d in _train)

            (s_hat, g_hat), _ = _nelder_mead(obj, [s0, 0.1], steps=[0.1, 0.05])
            resp.s[ink][ch] = max(s_hat, 0.0)
            resp.g[ink][ch] = min(max(g_hat, 0.0), 1.2)

    # ---- dual-ink: fit k per order per channel on training rows ---------- #
    for order in ORDERS:
        rows = [r for r in dual_rows if r["order"] == order]
        resp.k[order] = {}
        u, v = UNDER_OF[order], OVER_OF[order]
        for ch in CHANNELS:
            train = [(r["cu"], r["cv"], r[f"d{ch}"]) for r in rows if r["split"] == "train"]
            if not train:
                resp.k[order][ch] = 0.5
                continue

            def obj(k, _train=train, _ch=ch, _u=u, _v=v):
                if not (0.0 <= k <= 1.0):
                    return 1e12
                return sum(
                    (dual_D(cu, cv, resp.s[_u][_ch], resp.s[_v][_ch],
                            resp.g[_u][_ch], resp.g[_v][_ch], k) - d) ** 2
                    for cu, cv, d in _train
                )

            resp.k[order][ch] = _golden(obj, 0.0, 1.0)

    # ---- hold-out report: the held-out *intermediate coverage* ramps ------ #
    report = {
        "paper": paper,
        "single": _holdout_report(
            [r for r in single_rows if r["split"] == "holdout"],
            lambda r, ch: resp.single_density(r["ink"], r["coverage"], ch),
        ),
        "dual": _holdout_report(
            [r for r in dual_rows if r["split"] == "holdout"],
            lambda r, ch: resp.dual_density(r["order"], r["cu"], r["cv"], ch),
        ),
    }
    return resp, report


def _holdout_report(rows: List[dict], pred) -> dict:
    errs = []
    worst = (0.0, None)
    for r in rows:
        for ch in CHANNELS:
            e = pred(r, ch) - r[f"d{ch}"]
            errs.append(e)
            if abs(e) > abs(worst[0]):
                worst = (e, {"row": r, "ch": ch})
    return {
        "n": len(rows) * len(CHANNELS),
        "rmse": _rmse(errs),
        "max_abs": max((abs(e) for e in errs), default=0.0),
        "worst": None if worst[1] is None else {"err": worst[0], **worst[1]},
    }


def fit_all(single_rows: List[dict], dual_rows: List[dict]) -> Tuple[Dict[str, Response], dict]:
    papers = sorted({r["paper"] for r in single_rows} | {r["paper"] for r in dual_rows})
    responses, reports = {}, {}
    for p in papers:
        sr = [r for r in single_rows if r["paper"] == p]
        dr = [r for r in dual_rows if r["paper"] == p]
        resp, rep = fit_paper(p, sr, dr)
        responses[p] = resp
        reports[p] = rep
    return responses, reports
