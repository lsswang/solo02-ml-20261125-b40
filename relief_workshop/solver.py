from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

from .grid import HEIGHTS, NEG_INF, GridProblem


@dataclass
class SolveResult:
    feasible: bool
    score: float | None
    heights: tuple[int, ...] | None
    reason: str | None = None


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def solve_map(problem: GridProblem) -> SolveResult:
    """Exact max-product DP along the longer grid axis.

    The line width is ``min(rows, cols)``; for the promised <=3x6 scans it is
    at most 3, so each line has only 4**3=64 states.
    """
    r, c = problem.rows, problem.cols
    n = problem.n_vertices
    pair_edge = {_pair_key(e.u, e.v): e for e in problem.edges}

    if c >= r:
        line_count, line_width = c, r
        axis = "col"
    else:
        line_count, line_width = r, c
        axis = "row"

    states = list(product(HEIGHTS, repeat=line_width))
    state_index = {s: i for i, s in enumerate(states)}
    k = len(states)

    def vertex_at(line: int, pos: int) -> int:
        if axis == "col":
            return pos * c + line
        return line * c + pos

    # Edges wholly inside one line and edges between consecutive lines.
    within: list[list[tuple[int, int, object, int, int]]] = [[] for _ in range(line_count)]
    between: list[list[tuple[int, object, int, int]]] = [[] for _ in range(line_count - 1)]

    for li in range(line_count):
        for p in range(line_width - 1):
            a = vertex_at(li, p)
            b = vertex_at(li, p + 1)
            e = pair_edge.get(_pair_key(a, b))
            if e is not None:
                within[li].append((p, p + 1, e, a, b))
        if li < line_count - 1:
            for p in range(line_width):
                a = vertex_at(li, p)
                b = vertex_at(li + 1, p)
                e = pair_edge.get(_pair_key(a, b))
                if e is not None:
                    between[li].append((p, e, a, b))

    line_scores: list[list[float]] = []
    for li in range(line_count):
        scores = [NEG_INF] * k
        for j, st in enumerate(states):
            total = 0.0
            ok = True
            for p, h in enumerate(st):
                v = vertex_at(li, p)
                if not (problem.unary_mask[v] & (1 << h)):
                    ok = False
                    break
            if not ok:
                scores[j] = NEG_INF
                continue
            for p, q, e, a, b in within[li]:
                w = e.weight_pair(a, b, st[p], st[q])
                if w == NEG_INF:
                    ok = False
                    break
                total += w
            scores[j] = total if ok else NEG_INF
        line_scores.append(scores)

    boundary_matrices: list[list[list[float]]] = []
    for li in range(line_count - 1):
        mat = [[NEG_INF] * k for _ in range(k)]
        edges_here = between[li]
        for i, prev in enumerate(states):
            for j, nxt in enumerate(states):
                total = 0.0
                ok = True
                for p, e, a, b in edges_here:
                    # a is in prev line at p and b in next line at p.
                    w = e.weight_pair(a, b, prev[p], nxt[p])
                    if w == NEG_INF:
                        ok = False
                        break
                    total += w
                mat[i][j] = total if ok else NEG_INF
        boundary_matrices.append(mat)

    dp = line_scores[0][:]
    backpointers: list[list[int]] = []
    for li in range(1, line_count):
        mat = boundary_matrices[li - 1]
        nxt_dp = [NEG_INF] * k
        bp = [-1] * k
        for j in range(k):
            base = line_scores[li][j]
            if base == NEG_INF:
                continue
            best = NEG_INF
            best_i = -1
            col = mat
            for i in range(k):
                if dp[i] == NEG_INF or col[i][j] == NEG_INF:
                    continue
                cand = dp[i] + col[i][j] + base
                if cand > best:
                    best = cand
                    best_i = i
            nxt_dp[j] = best
            bp[j] = best_i
        dp = nxt_dp
        backpointers.append(bp)

    final_j = -1
    best_score = NEG_INF
    for j, val in enumerate(dp):
        if val > best_score:
            best_score = val
            final_j = j

    if final_j < 0 or not math.isfinite(best_score):
        return SolveResult(False, None, None, "no height assignment satisfies holes, cuts, gauge and 0..3 domain")

    chosen_states = [0] * line_count
    j = final_j
    chosen_states[-1] = j
    for li in range(line_count - 1, 0, -1):
        i = backpointers[li - 1][j]
        if i < 0:
            return SolveResult(False, None, None, "backtracking failed")
        j = i
        chosen_states[li - 1] = j

    heights = [0] * n
    for li, state_idx in enumerate(chosen_states):
        st = states[state_idx]
        for p, h in enumerate(st):
            heights[vertex_at(li, p)] = h

    heights_t = tuple(heights)
    exact_score = problem.score(heights_t)
    if not math.isfinite(exact_score):
        return SolveResult(False, None, None, "DP returned an infeasible assignment")
    return SolveResult(True, exact_score, heights_t)


def brute_force_best(problem: GridProblem) -> SolveResult:
    """Independent 4**N enumerator used by tests and very small checks."""
    best_score = NEG_INF
    best_h: tuple[int, ...] | None = None
    for h in product(HEIGHTS, repeat=problem.n_vertices):
        s = problem.score(h)
        if math.isfinite(s) and s > best_score:
            best_score = s
            best_h = h
    if best_h is None:
        return SolveResult(False, None, None, "enumeration found no feasible assignment")
    return SolveResult(True, best_score, best_h)


def brute_force_optima(problem: GridProblem, tolerance: float = 1e-9) -> list[tuple[int, ...]]:
    best = brute_force_best(problem)
    if not best.feasible or best.score is None:
        return []
    out: list[tuple[int, ...]] = []
    for h in product(HEIGHTS, repeat=problem.n_vertices):
        s = problem.score(h)
        if math.isfinite(s) and abs(s - best.score) <= tolerance:
            out.append(h)
    return out
