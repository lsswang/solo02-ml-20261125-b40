from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .model import TastingVote
from .utils import parse_composition, read_csv


def _as_binary(value: str) -> int:
    s = str(value).strip().lower()
    if s in {"1", "yes", "y", "true", "similar", "像", "是", "好", "合格"}:
        return 1
    if s in {"0", "no", "n", "false", "different", "不像", "否", "不合格"}:
        return 0
    try:
        return 1 if int(float(s)) > 0 else 0
    except ValueError as e:
        raise ValueError(f"cannot interpret vote {value!r}") from e


def _composition_from_row(row: Dict[str, str]) -> Dict[str, int]:
    for col in ("composition", "composition_parts", "recipe_composition", "ratio", "mix"):
        if row.get(col):
            return parse_composition(row[col])
    parts: Dict[str, int] = {}
    for key, value in row.items():
        if value in (None, ""):
            continue
        for prefix in ("parts_", "part_", "dose_", "quarter_", "q_", "frac_", "fraction_"):
            if key.startswith(prefix):
                batch = key[len(prefix) :]
                try:
                    fv = float(value)
                    # Fraction columns in 0..1 are converted to quarter parts
                    # only when they exactly line up with the quarter format.
                    if prefix in {"frac_", "fraction_"} and 0 < fv <= 1:
                        iv = int(round(fv * 4))
                    else:
                        iv = int(float(value))
                except ValueError:
                    continue
                if iv:
                    parts[batch] = iv
                break
    if not parts:
        raise ValueError(
            "row has neither composition nor parts_<BATCH> columns: " + repr(row)
        )
    return parts


def load_tasting_votes(path: str) -> List[TastingVote]:
    """Load the canonical judge-level CSV or a small aggregate-count variant.

    Canonical columns:
    vote_id,roast_id,round_no,formula_id,judge_id,composition,vote

    Aggregate variant may replace judge_id/vote with similar_votes and
    dissimilar_votes.  Aggregate counts become weighted Bernoulli observations.
    """
    rows = read_csv(path)
    votes: List[TastingVote] = []
    basename = os.path.basename(path)
    for idx, row in enumerate(rows, start=2):
        comp = _composition_from_row(row)
        roast = (row.get("roast_id") or row.get("roast") or row.get("roast_batch_id") or row.get("roast_batch") or "").strip()
        formula = (row.get("formula_id") or row.get("recipe_id") or row.get("formula") or "").strip()
        round_no = str(row.get("round_no") or row.get("round_id") or row.get("round") or "1").strip()
        if not roast or not formula:
            raise ValueError(f"{path}:{idx}: roast_id and formula_id are required")
        vote_id = (row.get("vote_id") or row.get("record_id") or f"{basename}:row{idx}").strip()

        sim_s = row.get("similar_votes") or row.get("yes_votes") or row.get("similar")
        diff_s = row.get("dissimilar_votes") or row.get("no_votes") or row.get("different")
        if sim_s not in (None, "") or diff_s not in (None, ""):
            n_yes = int(float(sim_s or 0))
            n_no = int(float(diff_s or 0))
            for kind, count, y in (("yes", n_yes, 1), ("no", n_no, 0)):
                if count < 0:
                    raise ValueError(f"{path}:{idx}: negative vote count")
                if count:
                    votes.append(
                        TastingVote(
                            vote_id=f"{vote_id}:{kind}",
                            roast_id=roast,
                            round_no=round_no,
                            formula_id=formula,
                            judge_id="aggregate",
                            composition=comp,
                            y=y,
                            weight=float(count),
                        )
                    )
            continue

        judge = (row.get("judge_id") or row.get("judge") or row.get("panelist_id") or "anonymous").strip()
        vote_col = row.get("vote") or row.get("similar") or row.get("judgement") or row.get("y")
        if vote_col in (None, ""):
            raise ValueError(f"{path}:{idx}: missing vote/similar column")
        votes.append(
            TastingVote(
                vote_id=vote_id,
                roast_id=roast,
                round_no=round_no,
                formula_id=formula,
                judge_id=judge,
                composition=comp,
                y=_as_binary(vote_col),
                weight=1.0,
            )
        )
    if not votes:
        raise ValueError(f"no tasting records found in {path}")
    return votes


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _audit_partition(votes: Sequence[TastingVote], train_idx: Sequence[int], test_idx: Sequence[int]) -> Dict[str, Any]:
    test_set = set(test_idx)
    test_roasts = sorted({votes[i].roast_id for i in test_idx})
    test_formulas = sorted({votes[i].formula_id for i in test_idx})
    train_roasts = sorted({votes[i].roast_id for i in train_idx})
    train_formulas = sorted({votes[i].formula_id for i in train_idx})
    test_roast_set = set(test_roasts)
    test_formula_set = set(test_formulas)

    roast_violations = sorted(
        {
            v.roast_id
            for i, v in enumerate(votes)
            if (v.roast_id in test_roast_set) != (i in test_set)
        }
    )
    formula_violations = sorted(
        {
            v.formula_id
            for i, v in enumerate(votes)
            if (v.formula_id in test_formula_set) != (i in test_set)
        }
    )

    event_sides: Dict[Tuple[str, str, str], Set[bool]] = {}
    for i, v in enumerate(votes):
        key = (v.roast_id, v.formula_id, str(v.round_no))
        event_sides.setdefault(key, set()).add(i in test_set)
    event_violations = sorted(
        f"{r}/{f}/round-{rn}" for (r, f, rn), sides in event_sides.items() if len(sides) > 1
    )
    return {
        "train_rows": len(train_idx),
        "blind_rows": len(test_idx),
        "train_roasts": train_roasts,
        "blind_roasts": test_roasts,
        "train_formulas": train_formulas,
        "blind_formulas": test_formulas,
        "whole_roast_violations": roast_violations,
        "same_formula_split_violations": formula_violations,
        "same_tasting_event_group_violations": event_violations,
        "whole_roast_ok": not roast_violations,
        "same_tasting_event_group_ok": not event_violations,
        "same_formula_globally_ok": not formula_violations,
        "has_both_partitions": bool(train_idx) and bool(test_idx),
    }


def grouped_blind_split(
    votes: Sequence[TastingVote],
    *,
    holdout_roast: Optional[str] = None,
    test_component_index: int = -1,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """Split with the strongest available grouping guarantee.

    Preferred strategy is a connected roast/formula component split.  Joining
    each roast node to its formula node means the blind component contains
    *all* votes of each held roast and *all* judge votes of each held formula.

    If a swapped input connects every roast/formula into one component, that
    strict split has no training data.  The function then falls back to a
    whole-roast split; this still keeps every judge in a tasting event in one
    partition, while the audit reports any global formula re-roast overlap.
    """
    if not votes:
        raise ValueError("cannot split empty records")

    uf = UnionFind()
    roast_nodes: Set[str] = set()
    for v in votes:
        r = f"roast::{v.roast_id}"
        f = f"formula::{v.formula_id}"
        uf.add(r)
        uf.add(f)
        uf.union(r, f)
        roast_nodes.add(v.roast_id)

    components: Dict[str, List[int]] = {}
    for i, v in enumerate(votes):
        root = uf.find(f"roast::{v.roast_id}")
        components.setdefault(root, []).append(i)
    roots = sorted(components.keys(), key=lambda r: (-len(components[r]), r))

    requested_found = (not holdout_roast) or (f"roast::{holdout_roast}" in uf.parent)
    strict_error: Optional[str] = None
    train_idx: List[int] = []
    test_idx: List[int] = []
    chosen_root: Optional[str] = None

    if requested_found:
        if holdout_roast:
            chosen_root = uf.find(f"roast::{holdout_roast}")
        else:
            chosen_root = roots[test_component_index]
        test_idx = sorted(components[chosen_root])
        test_set = set(test_idx)
        train_idx = [i for i in range(len(votes)) if i not in test_set]
        if len(train_idx) < 2 or not test_idx:
            strict_error = "connected component left fewer than two training rows"
    else:
        strict_error = f"requested holdout roast {holdout_roast!r} was not found"

    if not strict_error:
        audit = _audit_partition(votes, train_idx, test_idx)
        audit.update(
            {
                "strategy": "connected_roast_formula_components",
                "holdout_roast_requested": holdout_roast,
                "requested_roast_found": requested_found,
                "n_components": len(roots),
                "selected_component_root": chosen_root,
                "fallback_reason": None,
                "strict_formula_and_roast_ok": (
                    audit["whole_roast_ok"] and audit["same_formula_globally_ok"]
                ),
                "invariants_ok": (
                    audit["has_both_partitions"]
                    and audit["whole_roast_ok"]
                    and audit["same_tasting_event_group_ok"]
                    and audit["same_formula_globally_ok"]
                ),
            }
        )
        return train_idx, test_idx, audit

    # Robust fallback 1: leave one whole roast out.
    roasts = sorted(roast_nodes)
    selected_roast: Optional[str] = None
    if holdout_roast and holdout_roast in roast_nodes and len(roasts) > 1:
        selected_roast = holdout_roast
    elif len(roasts) > 1:
        # Deterministically choose a medium/large roast but never the only one.
        counts = {r: sum(1 for v in votes if v.roast_id == r) for r in roasts}
        selected_roast = sorted(roasts, key=lambda r: (-counts[r], r))[0]
        # If the largest is nearly all data, choose the second largest.
        if counts[selected_roast] > 0.8 * len(votes) and len(roasts) > 1:
            selected_roast = sorted(roasts, key=lambda r: (-counts[r], r))[1]

    if selected_roast is not None:
        test_idx = [i for i, v in enumerate(votes) if v.roast_id == selected_roast]
        test_set = set(test_idx)
        train_idx = [i for i in range(len(votes)) if i not in test_set]
        audit = _audit_partition(votes, train_idx, test_idx)
        audit.update(
            {
                "strategy": "whole_roast_fallback",
                "holdout_roast_requested": holdout_roast,
                "requested_roast_found": requested_found,
                "n_components": len(roots),
                "selected_component_root": None,
                "selected_roast": selected_roast,
                "fallback_reason": strict_error,
                "strict_formula_and_roast_ok": False,
                "invariants_ok": (
                    audit["has_both_partitions"]
                    and audit["whole_roast_ok"]
                    and audit["same_tasting_event_group_ok"]
                ),
            }
        )
        return train_idx, test_idx, audit

    # Robust fallback 2: one roast only, so group globally by formula.
    formula_nodes = sorted({v.formula_id for v in votes})
    if len(formula_nodes) > 1:
        test_formula = formula_nodes[-1]
        test_idx = [i for i, v in enumerate(votes) if v.formula_id == test_formula]
        test_set = set(test_idx)
        train_idx = [i for i in range(len(votes)) if i not in test_set]
        audit = _audit_partition(votes, train_idx, test_idx)
        audit.update(
            {
                "strategy": "formula_group_fallback_no_whole_roast_available",
                "holdout_roast_requested": holdout_roast,
                "requested_roast_found": requested_found,
                "n_components": len(roots),
                "selected_formula": test_formula,
                "fallback_reason": strict_error,
                "strict_formula_and_roast_ok": audit["same_formula_globally_ok"],
                "invariants_ok": (
                    audit["has_both_partitions"]
                    and audit["same_formula_globally_ok"]
                    and audit["same_tasting_event_group_ok"]
                ),
            }
        )
        return train_idx, test_idx, audit

    # Pathological one-roast/one-formula input: keep all same-recipe judges
    # together and report an explicitly in-sample metric rather than crashing.
    all_idx = list(range(len(votes)))
    audit = _audit_partition(votes, all_idx, all_idx)
    audit.update(
        {
            "strategy": "in_sample_only_not_enough_groups",
            "holdout_roast_requested": holdout_roast,
            "requested_roast_found": requested_found,
            "n_components": len(roots),
            "fallback_reason": strict_error,
            "strict_formula_and_roast_ok": True,
            "invariants_ok": False,
            "warning": "Input has only one roast/formula; log loss is in-sample.",
        }
    )
    return all_idx, all_idx, audit

