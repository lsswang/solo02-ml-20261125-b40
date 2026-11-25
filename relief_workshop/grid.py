from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
from typing import Iterable

DELTAS = (-2, -1, 0, 1, 2)
NEG_INF = -math.inf
HEIGHTS = (0, 1, 2, 3)
ALL_HEIGHT_MASK = 0b1111


def delta_index(d: int) -> int:
    return d + 2


def coord_to_index(row: int, col: int, cols: int, base: int = 0) -> int:
    """row/col are in ``base`` coordinates; internal indices are zero based."""
    return (row - base) * cols + (col - base)


def index_to_coord(index: int, cols: int, base: int = 1) -> tuple[int, int]:
    r, c = divmod(index, cols)
    return r + base, c + base


@dataclass
class Edge:
    edge_id: str
    u: int
    v: int
    # log weights for directed delta h[v] - h[u], ordered -2,-1,0,1,2
    logw: list[float]
    raw_prob: list[float] | None = None
    allowed_mask: list[bool] | None = None

    def weight_uv(self, hu: int, hv: int) -> float:
        d = hv - hu
        if d < -2 or d > 2:
            return NEG_INF
        return self.logw[delta_index(d)]

    def weight_pair(self, a: int, b: int, ha: int, hb: int) -> float:
        """Weight for the directed edge stored as u->v when querying a->b."""
        if a == self.u and b == self.v:
            return self.weight_uv(ha, hb)
        if a == self.v and b == self.u:
            # Stored delta is h[u]-h[v].
            return self.weight_uv(hb, ha)
        raise KeyError((self.edge_id, a, b, self.u, self.v))


@dataclass
class GridProblem:
    rows: int
    cols: int
    unary_mask: list[int] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    edge_by_id: dict[str, Edge] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.rows * self.cols
        if not self.unary_mask:
            self.unary_mask = [ALL_HEIGHT_MASK] * n
        if len(self.unary_mask) != n:
            raise ValueError("unary_mask length does not match grid")
        if not self.edge_by_id:
            self.edge_by_id = {e.edge_id: e for e in self.edges}

    @property
    def n_vertices(self) -> int:
        return self.rows * self.cols

    def allowed_heights(self, vertex: int) -> list[int]:
        mask = self.unary_mask[vertex]
        return [h for h in HEIGHTS if mask & (1 << h)]

    def score(self, heights: tuple[int, ...] | list[int]) -> float:
        if len(heights) != self.n_vertices:
            raise ValueError("wrong assignment length")
        total = 0.0
        for v, h in enumerate(heights):
            if not (self.unary_mask[v] & (1 << h)):
                return NEG_INF
        for e in self.edges:
            w = e.weight_uv(heights[e.u], heights[e.v])
            if w == NEG_INF:
                return NEG_INF
            total += w
        return total

    def cycle_checks(self, heights: tuple[int, ...] | list[int]) -> list[dict]:
        """Return closure errors for every elementary square (should be empty)."""
        errors: list[dict] = []
        c = self.cols
        for r in range(self.rows - 1):
            for col in range(self.cols - 1):
                tl = r * c + col
                tr = tl + 1
                bl = (r + 1) * c + col
                br = bl + 1
                # Clockwise: TL->TR->BR->BL->TL.  Height differences sum to zero.
                s = (
                    heights[tr] - heights[tl]
                    + heights[br] - heights[tr]
                    + heights[bl] - heights[br]
                    + heights[tl] - heights[bl]
                )
                if s != 0:
                    errors.append({"row": r + 1, "col": col + 1, "sum": s})
        return errors


def constrain_exact(mask: int, height: int) -> int:
    return mask & (1 << height)


def constrain_allowed(mask: int, heights: Iterable[int]) -> int:
    m = 0
    for h in heights:
        m |= 1 << h
    return mask & m
