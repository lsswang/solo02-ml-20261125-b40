"""Tiny linear-algebra routines for short calibration systems."""

from __future__ import annotations

import math
from typing import List, Sequence

Matrix = List[List[float]]
Vector = List[float]


class SingularSystem(RuntimeError):
    pass


def solve_square(a: Matrix, b: Vector) -> Vector:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise SingularSystem(f"singular matrix at column {col}")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        pivot_value = m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / pivot_value
            if factor == 0.0:
                continue
            m[r][col] = 0.0
            for c in range(col + 1, n + 1):
                m[r][c] -= factor * m[col][c]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        value = m[i][n]
        for j in range(i + 1, n):
            value -= m[i][j] * x[j]
        if abs(m[i][i]) < 1e-12:
            raise SingularSystem(f"singular matrix at row {i}")
        x[i] = value / m[i][i]
    return x


def least_squares(design: Sequence[Sequence[float]], target: Sequence[float]) -> Vector:
    """Solve ``argmin ||X beta - y||^2`` via normal equations.

    Calibration designs in this project are intentionally small and full rank.
    A rank failure raises instead of silently producing a misleading room.
    """
    rows = len(design)
    if rows != len(target):
        raise ValueError("design and target length differ")
    p = len(design[0]) if rows else 0
    if p == 0:
        return []

    xtx: Matrix = [[0.0] * p for _ in range(p)]
    xty: Vector = [0.0] * p

    for row, y in zip(design, target):
        for i in range(p):
            ri = row[i]
            if ri == 0.0:
                continue
            xty[i] += ri * y
            # Symmetric accumulation: only the upper triangle is needed.
            for j in range(i, p):
                xtx[i][j] += ri * row[j]

    for i in range(p):
        for j in range(i + 1, p):
            xtx[j][i] = xtx[i][j]

    # A small ridge only stabilizes PCM quantization; it is negligible for
    # exact synthetic coefficients.
    for i in range(p):
        xtx[i][i] += 1e-10

    return solve_square(xtx, xty)


def solve_2x2(a: float, b: float, c: float, d: float, r0: float, r1: float) -> tuple[float, float]:
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise SingularSystem("H0 is rank deficient")
    return ((d * r0 - b * r1) / det, (a * r1 - c * r0) / det)


def max_abs(values: Sequence[float]) -> float:
    return max((abs(v) for v in values), default=0.0)


def rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))
