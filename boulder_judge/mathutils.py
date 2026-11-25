"""Small numerical helpers.

The reference implementation intentionally uses only the Python standard
library.  Clips are at most 16 frames in the demonstrations, so log-space
dynamic programming is fast enough and avoids underflow.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

NEG_INF = float("-inf")


def logsumexp(values: Iterable[float]) -> float:
    """Numerically stable log-sum-exp; all -inf inputs yield -inf."""
    vals = list(values)
    m = max(vals) if vals else NEG_INF
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(math.fsum(math.exp(v - m) for v in vals))


def exp_normalize(log_weights: Sequence[float]) -> list[float]:
    """Return probabilities from unnormalized log weights."""
    if not log_weights:
        return []
    total_log = logsumexp(log_weights)
    if total_log == NEG_INF:
        n = len(log_weights)
        return [1.0 / n] * n
    return [math.exp(x - total_log) for x in log_weights]


def tv_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """Total variation distance between two discrete distributions."""
    return 0.5 * math.fsum(abs(float(a) - float(b)) for a, b in zip(p, q))


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in nats (0 means identical)."""
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def kl(xs: Sequence[float], ys: Sequence[float]) -> float:
        total = 0.0
        for x, y in zip(xs, ys):
            if x > 1e-15:
                y = max(y, 1e-15)
                total += x * math.log(x / y)
        return total

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
