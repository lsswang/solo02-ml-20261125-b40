"""Exact adaptive workbench planner for small deterministic optical trays."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .core import (
    Hypothesis,
    TrayModel,
    best_material,
    money,
    posterior_from_mask,
)

FRONT = 0
BACK = 1
SIDE_CODE = {"front": FRONT, "back": BACK}
SIDE_NAME = {FRONT: "front", BACK: "back"}


@dataclass(frozen=True)
class Choice:
    kind: str  # test | flip | checkout | stop | inconsistent
    action_id: Optional[str] = None
    material_id: Optional[str] = None
    test_index: Optional[int] = None
    incremental_cost: Fraction = Fraction(0)
    reason: str = ""


class WorkbenchPlanner:
    """Finite-horizon Bayesian dynamic program.

    A state contains only quantities that affect future choices: surviving
    joint hypotheses, current face, flips left, used tests, paid instrument
    set, and remaining budget.  With at most five binary tests and a few
    orientations per material the full state graph is tiny on CPU.
    """

    def __init__(self, model: TrayModel):
        self.model = model
        self.active = model.active_tests
        self.full_mask = (1 << len(model.hypotheses)) - 1
        self.initial_paid_mask = self._paid_mask(model.initial_paid)
        self.initial_side_code = SIDE_CODE[model.initial_side]
        self.memo: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def _paid_mask(self, ids: Sequence[str]) -> int:
        mask = 0
        for iid in ids:
            mask |= self.model.instrument_bit(iid)
        return mask

    def initial_key(self, budget: Optional[Fraction | float | str] = None) -> Tuple[Any, ...]:
        b = self.model.budget if budget is None else money(budget)
        return (
            self.full_mask,
            self.initial_side_code,
            self.model.max_flips,
            0,
            self.initial_paid_mask,
            b,
        )

    def setup_fee(self, instrument_id: str, paid_mask: int) -> Fraction:
        bit = self.model.instrument_bit(instrument_id)
        if paid_mask & bit:
            return Fraction(0)
        return self.model.instruments[instrument_id].setup_fee

    def test_incremental_cost(self, test_index: int, paid_mask: int) -> Fraction:
        t = self.active[test_index]
        return t.cost + self.setup_fee(t.instrument, paid_mask)

    def available_test_indices(
        self, side_code: int, done_mask: int
    ) -> List[int]:
        out = []
        for i, t in enumerate(self.active):
            if done_mask & (1 << i):
                continue
            if t.side == "either" or SIDE_CODE.get(t.side) == side_code:
                out.append(i)
        return out

    def _state_posterior(self, alive_mask: int):
        return posterior_from_mask(self.model, alive_mask)

    def solve(self, key: Tuple[Any, ...]) -> Dict[str, Any]:
        if key in self.memo:
            return self.memo[key]

        alive, side_code, flips_left, done_mask, paid_mask, remaining = key
        posteriors, _, total_mass = self._state_posterior(alive)

        if alive == 0 or total_mass <= 0:
            value = {
                "success": Fraction(0),
                "expected_cost": Fraction(0),
                "expected_correct": Fraction(0),
                "choice": Choice("inconsistent", reason="observations contradict all templates"),
            }
            self.memo[key] = value
            return value

        best_mid, best_p, eligible = best_material(posteriors, self.model.threshold)
        if eligible and best_mid is not None:
            choice = Choice(
                "checkout",
                material_id=best_mid,
                reason=f"posterior {float(best_p):.6f} >= threshold {float(self.model.threshold):.2f}",
            )
            value = {
                "success": Fraction(1),
                "expected_cost": Fraction(0),
                "expected_correct": best_p,
                "choice": choice,
            }
            self.memo[key] = value
            return value

        # Stopping as undecided is always available.
        best: Dict[str, Any] = {
            "success": Fraction(0),
            "expected_cost": Fraction(0),
            "expected_correct": Fraction(0),
            "choice": Choice("stop", reason="no remaining action improves the optimal objective"),
        }

        def consider(candidate: Dict[str, Any]) -> None:
            nonlocal best
            c_choice: Choice = candidate["choice"]
            # Lexicographic objective:
            # 1) maximize probability of a threshold-valid checkout,
            # 2) minimize expected additional spend,
            # 3) among exact ties prefer the more certain call.
            if candidate["success"] > best["success"]:
                best = candidate
            elif candidate["success"] == best["success"]:
                if candidate["expected_cost"] < best["expected_cost"]:
                    best = candidate
                elif candidate["expected_cost"] == best["expected_cost"]:
                    if candidate["expected_correct"] > best["expected_correct"]:
                        best = candidate

        # Optical binary tests.
        for i in self.available_test_indices(side_code, done_mask):
            t = self.active[i]
            inc = self.test_incremental_cost(i, paid_mask)
            if inc > remaining:
                continue

            outcome_masks = []
            probs = []
            next_values = []
            for y in (1, 0):  # bright first for stable presentation
                next_alive = 0
                numerator = Fraction(0)
                for h in self.model.hypotheses:
                    if not ((alive >> h.index) & 1):
                        continue
                    if h.outcomes[i] == y:
                        next_alive |= 1 << h.index
                        numerator += h.joint_prior
                p = numerator / total_mass if total_mass else Fraction(0)
                if p == 0 or next_alive == 0:
                    continue
                next_paid = paid_mask | self.model.instrument_bit(t.instrument)
                next_key = (
                    next_alive,
                    side_code,
                    flips_left,
                    done_mask | (1 << i),
                    next_paid,
                    remaining - inc,
                )
                nv = self.solve(next_key)
                outcome_masks.append((y, next_alive))
                probs.append(p)
                next_values.append(nv)

            if not probs:
                continue
            success = sum(p * nv["success"] for p, nv in zip(probs, next_values))
            future_cost = sum(p * nv["expected_cost"] for p, nv in zip(probs, next_values))
            expected_cost = inc + future_cost
            expected_correct = sum(
                p * nv["expected_correct"] for p, nv in zip(probs, next_values)
            )
            consider(
                {
                    "success": success,
                    "expected_cost": expected_cost,
                    "expected_correct": expected_correct,
                    "choice": Choice(
                        "test",
                        action_id=t.id,
                        test_index=i,
                        incremental_cost=inc,
                        reason=f"binary {t.name}; instrument {t.instrument}",
                    ),
                    "branches": list(
                        zip([y for y, _ in outcome_masks], probs, next_values)
                    ),
                }
            )

        # Flip to the other face.  It has no bright/dark outcome.
        if flips_left > 0 and self.model.max_flips > 0:
            inc = self.model.flip_cost
            if inc <= remaining:
                next_side = BACK if side_code == FRONT else FRONT
                next_key = (
                    alive,
                    next_side,
                    flips_left - 1,
                    done_mask,
                    paid_mask,
                    remaining - inc,
                )
                nv = self.solve(next_key)
                consider(
                    {
                        "success": nv["success"],
                        "expected_cost": inc + nv["expected_cost"],
                        "expected_correct": nv["expected_correct"],
                        "choice": Choice(
                            "flip",
                            action_id="flip",
                            incremental_cost=inc,
                            reason="turn the slice once; face-specific actions change",
                        ),
                        "flip_next": nv,
                    }
                )

        self.memo[key] = best
        return best

    def recommend(
        self,
        evidence: Mapping[str, Any],
        *,
        side: Optional[str] = None,
        flips_remaining: Optional[int] = None,
        used_tests: Optional[Sequence[str]] = None,
        paid_instruments: Optional[Sequence[str]] = None,
        spent: Fraction | float | str = 0,
        budget: Optional[Fraction | float | str] = None,
    ) -> Dict[str, Any]:
        from .core import alive_mask_from_evidence, binary_reading

        current_side = self.model.initial_side if side is None else side
        side_code = SIDE_CODE[str(current_side).lower()]
        flips = self.model.max_flips if flips_remaining is None else int(flips_remaining)
        used = set(used_tests or [])
        done_mask = 0
        for tid in used:
            if tid in self.model.active_test_index:
                done_mask |= 1 << self.model.active_test_index[tid]
        if paid_instruments is None:
            paid_mask = self.initial_paid_mask
        else:
            paid_mask = self._paid_mask(paid_instruments)
        b = self.model.budget if budget is None else money(budget)
        spent_f = money(spent)
        remaining = b - spent_f
        alive = alive_mask_from_evidence(self.model, evidence)
        key = (alive, side_code, flips, done_mask, paid_mask, remaining)
        value = self.solve(key)
        posteriors, _, total = posterior_from_mask(self.model, alive)
        best_mid, best_p, eligible = best_material(posteriors, self.model.threshold)

        choice: Choice = value["choice"]
        action_payload: Dict[str, Any]
        if choice.kind == "test":
            t = self.active[choice.test_index]  # type: ignore[index]
            setup = self.setup_fee(t.instrument, paid_mask)
            action_payload = {
                "type": "run_test",
                "test_id": t.id,
                "name": t.name,
                "instrument": t.instrument,
                "cost_now": float(t.cost),
                "setup_fee_now": float(setup),
                "total_charge_now": float(t.cost + setup),
                "side": t.side,
                "instruction": f"做「{t.name}」，录入 bright(明/亮) 或 dark(暗/不透)",
            }
        elif choice.kind == "flip":
            action_payload = {
                "type": "flip",
                "action_id": "flip",
                "cost_now": float(self.model.flip_cost),
                "instruction": "把矿片翻到另一面；当前正面专属试验之后将不可用",
            }
        elif choice.kind == "checkout":
            action_payload = {
                "type": "checkout",
                "material_id": choice.material_id,
                "material_name": self.model.material_names.get(
                    choice.material_id or "", choice.material_id
                ),
                "instruction": f"概率已达 {float(self.model.threshold):.0%}，签出 {choice.material_id}",
            }
        else:
            action_payload = {
                "type": "undecided",
                "instruction": "预算内没有可提高最优认出率的动作；保留未定判断",
            }

        return {
            "key": self.key_to_string(key),
            "alive_mask": alive,
            "alive_hypotheses": [
                {"material": h.material, "orientation": h.orientation}
                for h in self.model.hypotheses
                if (alive >> h.index) & 1
            ],
            "posterior": {m: float(p) for m, p in posteriors.items()},
            "best_material": best_mid,
            "best_probability": float(best_p),
            "checkout_eligible": eligible,
            "consistent": alive != 0 and total > 0,
            "remaining_budget": float(max(Fraction(0), remaining)),
            "actual_remaining_budget": float(remaining),
            "optimal_recognition_probability_from_here": float(value["success"]),
            "optimal_expected_additional_cost": float(value["expected_cost"]),
            "recommended": action_payload,
            "reason": choice.reason,
        }

    @staticmethod
    def key_to_string(key: Tuple[Any, ...]) -> str:
        alive, side, flips, done, paid, rem = key
        rem_s = f"{rem.numerator}/{rem.denominator}" if isinstance(rem, Fraction) else str(rem)
        return f"alive{alive}|side{SIDE_NAME.get(side, side)}|flips{flips}|done{done}|paid{paid}|budget{rem_s}"

    def plan_summary(self, budget: Optional[Fraction | float | str] = None) -> Dict[str, Any]:
        key = self.initial_key(budget)
        value = self.solve(key)
        choice = value["choice"]
        first: Dict[str, Any]
        if choice.kind == "test":
            t = self.active[choice.test_index]  # type: ignore[index]
            first = {
                "type": "run_test",
                "test_id": t.id,
                "name": t.name,
                "instrument": t.instrument,
                "charge": float(choice.incremental_cost),
            }
        elif choice.kind == "flip":
            first = {"type": "flip", "charge": float(choice.incremental_cost)}
        elif choice.kind == "checkout":
            first = {"type": "checkout", "material_id": choice.material_id}
        else:
            first = {"type": "undecided"}

        paths = self.enumerate_hypothesis_paths(budget=budget)
        return {
            "route": "adaptive_workbench",
            "budget": float(money(self.model.budget if budget is None else budget)),
            "threshold": float(self.model.threshold),
            "currency": self.model.currency,
            "best_recognition_probability": float(value["success"]),
            "minimum_expected_cost_at_best_rate": float(value["expected_cost"]),
            "expected_correct_recognition_probability": float(value["expected_correct"]),
            "first_action": first,
            "hypothesis_paths": paths,
            "state_count": len(self.memo),
        }

    def budget_curve(self, max_budget: Optional[int] = None) -> List[Dict[str, Any]]:
        if max_budget is None:
            max_budget = math.ceil(float(self.model.budget)) + 2
        rows = []
        for b in range(0, max_budget + 1):
            summary = self.plan_summary(budget=Fraction(b))
            rows.append(
                {
                    "budget": b,
                    "best_recognition_probability": summary["best_recognition_probability"],
                    "expected_cost": summary["minimum_expected_cost_at_best_rate"],
                    "expected_correct_probability": summary[
                        "expected_correct_recognition_probability"
                    ],
                    "first_action": summary["first_action"],
                }
            )
        return rows

    def enumerate_hypothesis_paths(
        self, budget: Optional[Fraction | float | str] = None
    ) -> List[Dict[str, Any]]:
        """Follow the optimal policy once for every known deterministic slice."""
        results = []
        for h in self.model.hypotheses:
            key = self.initial_key(budget)
            alive, side_code, flips_left, done_mask, paid_mask, remaining = key
            spent = Fraction(0)
            actions: List[Dict[str, Any]] = []
            terminal = "undecided"
            called_material: Optional[str] = None

            # Five tests + flips gives a conservative hard stop.
            for _ in range(len(self.active) + self.model.max_flips + 2):
                value = self.solve(key)
                choice: Choice = value["choice"]
                if choice.kind == "checkout":
                    terminal = "checkout"
                    called_material = choice.material_id
                    actions.append({"type": "checkout", "material_id": called_material})
                    break
                if choice.kind in {"stop", "inconsistent"}:
                    terminal = choice.kind
                    actions.append({"type": choice.kind})
                    break
                if choice.kind == "test":
                    i = choice.test_index
                    assert i is not None
                    t = self.active[i]
                    inc = self.test_incremental_cost(i, paid_mask)
                    y = h.outcomes[i]
                    actions.append(
                        {
                            "type": "test",
                            "test_id": t.id,
                            "result": "bright" if y else "dark",
                            "charge": float(inc),
                        }
                    )
                    next_alive = 0
                    for hh in self.model.hypotheses:
                        if ((alive >> hh.index) & 1) and hh.outcomes[i] == y:
                            next_alive |= 1 << hh.index
                    alive = next_alive
                    done_mask |= 1 << i
                    paid_mask |= self.model.instrument_bit(t.instrument)
                    spent += inc
                    remaining -= inc
                    key = (alive, side_code, flips_left, done_mask, paid_mask, remaining)
                elif choice.kind == "flip":
                    inc = self.model.flip_cost
                    actions.append({"type": "flip", "charge": float(inc)})
                    side_code = BACK if side_code == FRONT else FRONT
                    flips_left -= 1
                    spent += inc
                    remaining -= inc
                    key = (alive, side_code, flips_left, done_mask, paid_mask, remaining)
                else:
                    terminal = "undecided"
                    break

            results.append(
                {
                    "material": h.material,
                    "orientation": h.orientation,
                    "joint_prior": float(h.joint_prior),
                    "terminal": terminal,
                    "called_material": called_material,
                    "correct": called_material == h.material if called_material else False,
                    "spent": float(spent),
                    "actions": actions,
                }
            )
        return results

    def decision_table(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        rows = []
        for key, value in list(self.memo.items())[:limit]:
            alive, side, flips, done, paid, rem = key
            post, _, _ = self._state_posterior(alive)
            choice = value["choice"]
            rows.append(
                {
                    "state": self.key_to_string(key),
                    "side": SIDE_NAME[side],
                    "flips_remaining": flips,
                    "done_mask": done,
                    "paid_mask": paid,
                    "remaining_budget": float(rem),
                    "posterior": {m: float(p) for m, p in post.items()},
                    "recommendation": choice.kind
                    if choice.action_id is None
                    else f"{choice.kind}:{choice.action_id}",
                    "material_id": choice.material_id,
                    "optimal_recognition_probability": float(value["success"]),
                    "expected_cost": float(value["expected_cost"]),
                }
            )
        return rows
