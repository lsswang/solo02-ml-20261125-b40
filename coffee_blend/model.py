from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .utils import fractions, sigmoid


@dataclass
class TastingVote:
    vote_id: str
    roast_id: str
    round_no: str
    formula_id: str
    judge_id: str
    composition: Dict[str, int]
    y: int
    weight: float = 1.0


@dataclass
class BlendModel:
    """Quadratic mixture logistic model with batch terms and pair synergy.

    logit P(similar) = intercept
        + sum_b x_b * beta_b
        + sum_{b<c} x_b*x_c * gamma_bc

    A positive pair coefficient is what lets two disliked single-origin cups
    produce a high similarity probability after blending.
    """

    batches: List[str]
    pairs: List[Tuple[str, str]]
    coefficients: List[float]
    parts_per_cup: int = 4
    l2: float = 1e-5
    train_votes: int = 0
    train_weight: float = 0.0
    feature_names: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.feature_names is None:
            names = ["intercept"] + [f"batch::{b}" for b in self.batches]
            names += [f"pair::{a}*{b}" for a, b in self.pairs]
            self.feature_names = names

    def vectorize(self, composition: Dict[str, int]) -> List[float]:
        x = fractions(composition, self.parts_per_cup)
        unknown = sorted(set(x) - set(self.batches))
        if unknown:
            raise KeyError(
                "model has no coefficient for batch(es): "
                + ", ".join(unknown)
                + "; refit against the swapped sensory input first"
            )
        row = [1.0]
        row.extend(x.get(b, 0.0) for b in self.batches)
        for a, b in self.pairs:
            row.append(x.get(a, 0.0) * x.get(b, 0.0))
        return row

    def logit(self, composition: Dict[str, int]) -> float:
        z = self.vectorize(composition)
        return sum(zi * beta for zi, beta in zip(z, self.coefficients))

    def predict_proba(self, composition: Dict[str, int]) -> float:
        return sigmoid(self.logit(composition))

    def coefficient_map(self) -> Dict[str, float]:
        return {name: float(beta) for name, beta in zip(self.feature_names, self.coefficients)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "quadratic_mixture_logistic_ridge",
            "parts_per_cup": self.parts_per_cup,
            "batches": self.batches,
            "pairs": [list(p) for p in self.pairs],
            "feature_names": self.feature_names,
            "coefficients": [float(x) for x in self.coefficients],
            "coefficient_map": self.coefficient_map(),
            "l2": self.l2,
            "train_votes": self.train_votes,
            "train_weight": self.train_weight,
            "prediction_formula": (
                "sigmoid(intercept + sum fraction_b*beta_b + "
                "sum_{b<c} fraction_b*fraction_c*gamma_bc)"
            ),
        }

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "BlendModel":
        return cls(
            batches=list(obj["batches"]),
            pairs=[(a, b) for a, b in obj.get("pairs", [])],
            coefficients=list(obj["coefficients"]),
            parts_per_cup=int(obj.get("parts_per_cup", 4)),
            l2=float(obj.get("l2", 1e-5)),
            train_votes=int(obj.get("train_votes", 0)),
            train_weight=float(obj.get("train_weight", 0.0)),
            feature_names=list(obj.get("feature_names")) if obj.get("feature_names") else None,  # type: ignore[arg-type]
        )


def all_pairs(batches: Sequence[str]) -> List[Tuple[str, str]]:
    return [(batches[i], batches[j]) for i in range(len(batches)) for j in range(i + 1, len(batches))]


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting; adequate for tiny models."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            # Add a tiny diagonal instead of failing on near-collinear mixtures.
            m[col][col] += 1e-8
            pivot = col
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if abs(factor) < 1e-15:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def fit_blend_model(
    votes: Sequence[TastingVote],
    *,
    batches: Optional[Sequence[str]] = None,
    parts_per_cup: int = 4,
    l2: float = 1e-5,
    max_iter: int = 60,
    tolerance: float = 1e-9,
) -> BlendModel:
    if not votes:
        raise ValueError("cannot fit model with zero tasting votes")
    if batches is None:
        batch_set = set()
        for v in votes:
            batch_set.update(v.composition.keys())
        batches = sorted(batch_set)
    batches = list(batches)
    pairs = all_pairs(batches)
    feature_names = ["intercept"] + [f"batch::{b}" for b in batches]
    feature_names += [f"pair::{a}*{b}" for a, b in pairs]
    k = len(feature_names)

    # Build feature rows once.
    tmp_model = BlendModel(batches, pairs, [0.0] * k, parts_per_cup, l2, feature_names=feature_names)
    xs: List[List[float]] = []
    ys: List[float] = []
    ws: List[float] = []
    for vote in votes:
        xs.append(tmp_model.vectorize(vote.composition))
        ys.append(float(vote.y))
        ws.append(max(0.0, float(vote.weight)))

    beta = [0.0] * k
    prev_loss: Optional[float] = None
    for _iteration in range(max_iter):
        grad = [0.0] * k
        hess = [[0.0] * k for _ in range(k)]
        loss = 0.0
        weight_total = 0.0
        for x, y, w in zip(xs, ys, ws):
            if w == 0.0:
                continue
            z = sum(xj * bj for xj, bj in zip(x, beta))
            p = sigmoid(z)
            d = p - y
            for j in range(k):
                grad[j] += w * d * x[j]
                for h in range(j, k):
                    hess[j][h] += w * p * (1.0 - p) * x[j] * x[h]
            eps = 1e-15
            pc = min(1 - eps, max(eps, p))
            loss -= w * (y * math.log(pc) + (1 - y) * math.log(1 - pc))
            weight_total += w

        if weight_total <= 0:
            raise ValueError("all training weights are zero")
        for j in range(k):
            grad[j] /= weight_total
            for h in range(j, k):
                hess[j][h] /= weight_total
        loss /= weight_total

        # Ridge penalties are on the mean Bernoulli loss so their strength is
        # comparable when a verifier swaps in a differently sized CSV.
        for j in range(1, k):
            grad[j] += l2 * beta[j]
            hess[j][j] += l2
            loss += 0.5 * l2 * beta[j] * beta[j]

        for j in range(k):
            for h in range(j + 1, k):
                hess[h][j] = hess[j][h]

        delta = solve_linear(hess, grad)
        max_step = max(abs(d) for d in delta) if delta else 0.0
        # A simple damping guard keeps early Newton steps finite.
        step = 1.0
        if max_step > 8.0:
            step = 8.0 / max_step
        for j in range(k):
            beta[j] -= step * delta[j]

        if prev_loss is not None and abs(prev_loss - loss) < tolerance:
            break
        prev_loss = loss

    train_votes = sum(1 for v in votes if v.weight > 0)
    train_weight = float(sum(max(0.0, v.weight) for v in votes))
    return BlendModel(
        batches=batches,
        pairs=pairs,
        coefficients=beta,
        parts_per_cup=parts_per_cup,
        l2=l2,
        train_votes=train_votes,
        train_weight=train_weight,
        feature_names=feature_names,
    )
