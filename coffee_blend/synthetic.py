from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Sequence, Tuple

from .utils import write_csv, write_json

JIA = "JIA-2409"
YI_GOOD = "YI-2409"
YI_NEW = "YI-2418"
BING = "BING-2409"

FIELDNAMES = [
    "vote_id",
    "roast_id",
    "round_no",
    "formula_id",
    "judge_id",
    "composition",
    "vote",
]


def all_quarter_compositions(n_batches: int = 3, total: int = 4) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []

    def rec(pos: int, rem: int, cur: List[int]) -> None:
        if pos == n_batches - 1:
            out.append(tuple(cur + [rem]))
            return
        for v in range(rem + 1):
            rec(pos + 1, rem - v, cur + [v])

    rec(0, total, [])
    return out


def comp_string(batches: Sequence[str], parts: Sequence[int]) -> str:
    return "|".join(f"{b}:{p}" for b, p in zip(batches, parts) if p > 0)


def true_logit(
    parts: Sequence[int],
    *,
    alpha: float,
    batch_beta: Dict[int, float],
    pair_gamma: Dict[Tuple[int, int], float],
    roast_offset: float = 0.0,
) -> float:
    total = sum(parts)
    x = [p / total for p in parts]
    eta = alpha + roast_offset
    for i, xi in enumerate(x):
        eta += xi * batch_beta.get(i, 0.0)
    for (i, j), g in pair_gamma.items():
        eta += x[i] * x[j] * g
    return eta


def _add_formula_records(
    rows: List[Dict[str, Any]],
    rng: random.Random,
    *,
    batches: Sequence[str],
    parts: Tuple[int, ...],
    formula_id: str,
    roast_id: str,
    rounds: int,
    judges: int,
    params: Dict[str, Any],
    vote_counter: List[int],
) -> None:
    roast_offsets = params["roast_offsets"]
    for round_no in range(1, rounds + 1):
        for j in range(1, judges + 1):
            vid = f"V{vote_counter[0]:05d}"
            vote_counter[0] += 1
            eta = true_logit(
                parts,
                alpha=params["alpha"],
                batch_beta=params["batch_beta"],
                pair_gamma=params["pair_gamma"],
                roast_offset=roast_offsets.get(roast_id, 0.0),
            )
            # Small stable judge noise; no judge identity is needed by the
            # batch/ratio model, but IDs demonstrate grouped splitting.
            judge_offset = (((j * 37 + len(roast_id)) % 7) - 3) * 0.035
            p = 1.0 / (1.0 + math.exp(-(eta + judge_offset)))
            vote = 1 if rng.random() < p else 0
            rows.append(
                {
                    "vote_id": vid,
                    "roast_id": roast_id,
                    "round_no": str(round_no),
                    "formula_id": formula_id,
                    "judge_id": f"J{j:02d}",
                    "composition": comp_string(batches, parts),
                    "vote": vote,
                }
            )


def generate_votes(path: str, scenario: str = "good", seed: int = 231125) -> Dict[str, Any]:
    """Generate a small reproducible judge-level CSV.

    ``good`` has a strong JIA+YI half/half synergy.  ``bad_b`` substitutes a
    weak new YI batch whose half/half cup stays below the 0.8 shop gate.
    """
    rng = random.Random(seed)
    if scenario == "good":
        batches = [JIA, YI_GOOD, BING]
        # A strong balanced-pair term makes 2+2 pass while 3+1 / 1+3 stay
        # below the gate and both single-origin cups remain far from standard.
        alpha = -2.80
        batch_beta = {0: 0.25, 1: 0.15, 2: -0.15}
        pair_gamma = {(0, 1): 17.5, (0, 2): 2.2, (1, 2): 1.4}
        holdout_roast = "R-BLIND-11"
    elif scenario == "bad_b":
        batches = [JIA, YI_NEW, BING]
        alpha = -2.10
        batch_beta = {0: 0.20, 1: -0.85, 2: -0.20}
        pair_gamma = {(0, 1): 5.2, (0, 2): 2.0, (1, 2): 0.5}
        holdout_roast = "R-BLIND-N1"
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    comps = all_quarter_compositions(3, 4)
    # Stable formula IDs by composition.
    def fid(parts: Tuple[int, ...]) -> str:
        tags = []
        names = ["J", "Y", "C"]
        for n, p in zip(names, parts):
            if p:
                tags.append(f"{n}{p}")
        return "F-" + "-".join(tags)

    # The blind connected component is the famous 2+2 formula and its second
    # re-roast.  Pure singles stay in fit so batch quality is identifiable;
    # their low fitted probabilities demonstrate the synergy separately.
    blind_parts = {(2, 2, 0)}
    duplicate_train = {(4, 0, 0), (0, 4, 0), (3, 1, 0), (1, 3, 0), (2, 0, 2)}

    roast_offsets: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    counter = [1]
    roast_serial = 1

    roast = holdout_roast
    roast_offsets[roast] = rng.gauss(0.0, 0.10)
    ab22 = (2, 2, 0)
    _add_formula_records(
        rows, rng,
        batches=batches, parts=ab22, formula_id=fid(ab22),
        roast_id=roast, rounds=3, judges=9,
        params={"alpha": alpha, "batch_beta": batch_beta,
                "pair_gamma": pair_gamma, "roast_offsets": roast_offsets},
        vote_counter=counter,
    )
    blind2 = "R-BLIND-12" if scenario == "good" else "R-BLIND-N2"
    roast_offsets[blind2] = rng.gauss(0.0, 0.10)
    _add_formula_records(
        rows, rng,
        batches=batches, parts=ab22, formula_id=fid(ab22),
        roast_id=blind2, rounds=2, judges=9,
        params={"alpha": alpha, "batch_beta": batch_beta,
                "pair_gamma": pair_gamma, "roast_offsets": roast_offsets},
        vote_counter=counter,
    )

    # Training roasts for all other quarter recipes.
    for parts in comps:
        if parts in blind_parts:
            continue
        roast = f"R-TRAIN-{roast_serial:02d}"
        roast_serial += 1
        roast_offsets[roast] = rng.gauss(0.0, 0.10)
        rounds = 2
        _add_formula_records(
            rows, rng,
            batches=batches, parts=parts, formula_id=fid(parts),
            roast_id=roast, rounds=rounds, judges=12,
            params={"alpha": alpha, "batch_beta": batch_beta,
                    "pair_gamma": pair_gamma, "roast_offsets": roast_offsets},
            vote_counter=counter,
        )
        if parts in duplicate_train:
            roast2 = f"R-REROAST-{roast_serial:02d}"
            roast_serial += 1
            roast_offsets[roast2] = rng.gauss(0.0, 0.10)
            _add_formula_records(
                rows, rng,
                batches=batches, parts=parts, formula_id=fid(parts),
                roast_id=roast2, rounds=2, judges=12,
                params={"alpha": alpha, "batch_beta": batch_beta,
                        "pair_gamma": pair_gamma, "roast_offsets": roast_offsets},
                vote_counter=counter,
            )

    write_csv(path, rows, FIELDNAMES)
    meta = {
        "scenario": scenario,
        "seed": seed,
        "records": len(rows),
        "batches": batches,
        "holdout_roast": holdout_roast,
        "connected_extra_holdout_roast": blind2,
        "note": "The blind component is F-J2-Y2 across its requested roast and connected re-roast.",
    }
    write_json(path + ".meta.json", meta)
    return meta
