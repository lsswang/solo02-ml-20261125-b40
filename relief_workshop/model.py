from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .grid import DELTAS

FEATURES = ("ch0", "ch1", "ch2", "ch3")


def logsumexp(values: Sequence[float]) -> float:
    m = max(values)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in values))


@dataclass
class GaussianNBModel:
    priors: dict[int, float]
    means: dict[int, list[float]]
    variances: dict[int, list[float]]
    var_floor: float = 0.12
    label_smoothing: float = 0.004

    def to_dict(self) -> dict:
        return {
            "model_type": "gaussian_naive_bayes",
            "classes": list(DELTAS),
            "feature_columns": list(FEATURES),
            "var_floor": self.var_floor,
            "label_smoothing": self.label_smoothing,
            "priors": {str(d): self.priors[d] for d in DELTAS},
            "means": {str(d): self.means[d] for d in DELTAS},
            "variances": {str(d): self.variances[d] for d in DELTAS},
        }

    @classmethod
    def from_dict(cls, obj: dict) -> "GaussianNBModel":
        return cls(
            priors={int(k): float(v) for k, v in obj["priors"].items()},
            means={int(k): [float(x) for x in v] for k, v in obj["means"].items()},
            variances={int(k): [float(x) for x in v] for k, v in obj["variances"].items()},
            var_floor=float(obj.get("var_floor", 0.12)),
            label_smoothing=float(obj.get("label_smoothing", 0.004)),
        )

    def log_probabilities(self, x: Sequence[float]) -> dict[int, float]:
        logs: dict[int, float] = {}
        for d in DELTAS:
            lp = math.log(self.priors[d]) if self.priors[d] > 0 else -math.inf
            mu = self.means[d]
            va = self.variances[d]
            for j, xj in enumerate(x):
                v = max(va[j], self.var_floor)
                lp += -0.5 * math.log(2.0 * math.pi * v)
                lp += -0.5 * (xj - mu[j]) ** 2 / v
            logs[d] = lp
        z = logsumexp(list(logs.values()))
        eps = self.label_smoothing
        out: dict[int, float] = {}
        for d in DELTAS:
            raw_p = math.exp(logs[d] - z)
            p = (1.0 - eps) * raw_p + eps / len(DELTAS)
            out[d] = math.log(p)
        return out

    def probabilities(self, x: Sequence[float]) -> dict[int, float]:
        lp = self.log_probabilities(x)
        return {d: math.exp(v) for d, v in lp.items()}


def train_gaussian_nb(rows: Sequence[dict], var_floor: float = 0.12,
                      label_smoothing: float = 0.004) -> GaussianNBModel:
    by_class: dict[int, list[list[float]]] = {d: [] for d in DELTAS}
    for row in rows:
        d = int(row["delta"])
        if d not in by_class:
            raise ValueError(f"calibration label {d} is outside {DELTAS}")
        by_class[d].append([float(row[f]) for f in FEATURES])

    n = len(rows)
    priors: dict[int, float] = {}
    means: dict[int, list[float]] = {}
    variances: dict[int, list[float]] = {}
    for d in DELTAS:
        xs = by_class[d]
        # Laplace-style floor keeps a rare class in the five-class table.
        priors[d] = (len(xs) + 1.0) / (n + len(DELTAS))
        if xs:
            dim = len(FEATURES)
            mu = [sum(x[j] for x in xs) / len(xs) for j in range(dim)]
            var = []
            for j in range(dim):
                v = sum((x[j] - mu[j]) ** 2 for x in xs) / max(1, len(xs) - 1)
                var.append(max(v, 1e-6))
        else:
            mu = [0.0] * len(FEATURES)
            var = [1.0] * len(FEATURES)
        means[d] = mu
        variances[d] = var
    return GaussianNBModel(priors, means, variances, var_floor, label_smoothing)


def mean_log_loss(model: GaussianNBModel, rows: Sequence[dict]) -> tuple[float, float]:
    total = 0.0
    correct = 0
    for row in rows:
        d = int(row["delta"])
        x = [float(row[f]) for f in FEATURES]
        lp = model.log_probabilities(x)
        total -= lp[d]
        pred = max(lp, key=lp.get)
        correct += int(pred == d)
    return total / max(1, len(rows)), correct / max(1, len(rows))
