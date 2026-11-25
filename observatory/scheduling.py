from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import MAX_CANDIDATE_NIGHTS
from .calibration import write_json


BIT_DIM = 1
BIT_BRIGHT = 0


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是", "可"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


@dataclass(frozen=True)
class Ephemeris:
    id: str
    star: str
    pattern: tuple[int, ...]
    transit_time: str = ""
    note: str = ""

    def dims_on(self, night: int) -> bool:
        return 1 <= night <= len(self.pattern) and self.pattern[night - 1] == BIT_DIM

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "star": self.star,
            "transit_time": self.transit_time,
            "pattern": "".join(str(b) for b in self.pattern),
            "dimming_nights": [i + 1 for i, b in enumerate(self.pattern) if b == 1],
            "note": self.note,
        }


@dataclass(frozen=True)
class Shift:
    night: int
    cost: float = 1.0
    date: str = ""
    first_eligible: bool = False
    voucher_eligible: bool = False
    fixed_eligible: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "night": self.night,
            "date": self.date,
            "cost": self.cost,
            "first_eligible": self.first_eligible,
            "voucher_eligible": self.voucher_eligible,
            "fixed_eligible": self.fixed_eligible,
            "note": self.note,
        }


def pattern_from_row(row: dict[str, Any], horizon: int | None = None) -> tuple[int, ...]:
    if row.get("pattern"):
        s = str(row["pattern"]).strip()
        pattern = tuple(int(ch) for ch in s)
    elif row.get("future_pattern"):
        s = str(row["future_pattern"]).strip()
        pattern = tuple(int(ch) for ch in s)
    else:
        nights_text = str(row.get("dimming_nights", "")).replace(",", ";")
        nights = {int(x) for x in nights_text.split(";") if x.strip()}
        if horizon is None:
            horizon = max(nights) if nights else 0
        pattern = tuple(1 if i + 1 in nights else 0 for i in range(horizon))
    if any(b not in (0, 1) for b in pattern):
        raise ValueError("patterns may contain only 0 and 1")
    if horizon is not None and len(pattern) != horizon:
        # Allow a shorter explicit dimming_nights construction only; for an
        # explicit pattern the same-format contract requires exact horizon.
        if row.get("pattern") or row.get("future_pattern"):
            raise ValueError(f"pattern length {len(pattern)} != horizon_nights {horizon}")
        pattern = pattern + (0,) * (horizon - len(pattern))
    return pattern


def ephemerides_from_rows(rows: Iterable[dict[str, Any]], horizon: int) -> list[Ephemeris]:
    result: list[Ephemeris] = []
    for i, row in enumerate(rows):
        eid = str(row.get("id") or row.get("ephemeris_id") or f"E{i+1:03d}")
        star = str(row.get("star") or row.get("star_name") or "未注明")
        pattern = pattern_from_row(row, horizon)
        result.append(
            Ephemeris(
                id=eid,
                star=star,
                pattern=pattern,
                transit_time=str(row.get("transit_time", row.get("transit_at", ""))),
                note=str(row.get("note", "")),
            )
        )
    ids = [e.id for e in result]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ephemeris ids in application: {ids}")
    return result


def load_ephemerides_csv(path: str | Path, horizon: int) -> list[Ephemeris]:
    return ephemerides_from_rows(_read_csv(Path(path)), horizon)


def load_shifts(path: str | Path, horizon: int | None = None) -> dict[int, Shift]:
    p = Path(path)
    shifts: dict[int, Shift] = {}
    for row in _read_csv(p):
        night = int(row["night"])
        if night < 1 or night > MAX_CANDIDATE_NIGHTS:
            raise ValueError(f"night {night} exceeds one-order limit of {MAX_CANDIDATE_NIGHTS}")
        if horizon is not None and night > horizon:
            raise ValueError(f"shift night {night} is outside horizon_nights={horizon}")
        shift = Shift(
            night=night,
            cost=float(row.get("cost", 1.0) or 1.0),
            date=str(row.get("date", "")),
            first_eligible=parse_bool(row.get("first_eligible"), default=night == 1),
            voucher_eligible=parse_bool(row.get("voucher_eligible"), default=False),
            fixed_eligible=parse_bool(row.get("fixed_eligible"), default=True),
            note=str(row.get("note", "")),
        )
        if night in shifts:
            raise ValueError(f"duplicate shift row for night {night}")
        shifts[night] = shift
    if not shifts:
        raise ValueError(f"shift table contains no rows: {p}")
    return shifts


def load_clear_records(path: str | Path) -> set[int]:
    nights: set[int] = set()
    for row in _read_csv(Path(path)):
        result = str(row.get("result", row.get("status", "bright"))).strip().lower()
        if result in {"bright", "clear", "0", "未变暗", "晴"}:
            nights.add(int(row["night"]))
    return nights


@dataclass
class Application:
    order_id: str
    horizon: int
    ephemerides: list[Ephemeris]
    known_clear_nights: set[int]
    voucher_fee: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Application":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        horizon = int(data.get("horizon_nights", data.get("nights", 0)))
        if not 1 <= horizon <= MAX_CANDIDATE_NIGHTS:
            raise ValueError(f"horizon_nights must be between 1 and {MAX_CANDIDATE_NIGHTS}")

        rows: list[dict[str, Any]] = list(data.get("ephemerides", []))
        ephemeris_file = data.get("ephemerides_file")
        if ephemeris_file:
            epath = (p.parent / ephemeris_file).resolve()
            rows.extend(_read_csv(epath))
        ephemerides = ephemerides_from_rows(rows, horizon)

        clear_nights = {int(n) for n in data.get("known_clear_nights", [])}
        clear_file = data.get("clear_records_file")
        if clear_file:
            cpath = (p.parent / clear_file).resolve()
            clear_nights |= load_clear_records(cpath)

        return cls(
            order_id=str(data.get("order_id", p.stem)),
            horizon=horizon,
            ephemerides=ephemerides,
            known_clear_nights=clear_nights,
            voucher_fee=float(data.get("voucher_fee", 0.0) or 0.0),
            raw=data,
            source_path=p.resolve(),
        )


def filter_ephemerides(
    ephemerides: list[Ephemeris],
    supported_stars: list[str] | set[str] | None = None,
    clear_nights: Iterable[int] | None = None,
) -> tuple[list[Ephemeris], list[dict[str, Any]]]:
    clear = set(clear_nights or [])
    star_set = None if supported_stars is None else set(supported_stars)
    kept: list[Ephemeris] = []
    excluded: list[dict[str, Any]] = []
    for e in ephemerides:
        reasons: list[str] = []
        if star_set is not None and e.star not in star_set:
            reasons.append("star_excluded_by_photometric_calibration")
        conflict_nights = [n for n in clear if e.dims_on(n)]
        if conflict_nights:
            reasons.append("predicts_dim_on_known_clear_night")
        if reasons:
            excluded.append({"ephemeris": e.to_dict(), "reasons": reasons, "conflict_nights": conflict_nights})
        else:
            kept.append(e)
    return kept, excluded


def groups_for_nights(candidates: list[Ephemeris], nights: tuple[int, ...]) -> dict[tuple[int, ...], list[Ephemeris]]:
    groups: dict[tuple[int, ...], list[Ephemeris]] = {}
    for e in candidates:
        key = tuple(e.pattern[n - 1] for n in nights)
        groups.setdefault(key, []).append(e)
    return groups


def worst_group_size(candidates: list[Ephemeris], nights: tuple[int, ...]) -> int:
    if not candidates:
        return 0
    return max(len(g) for g in groups_for_nights(candidates, nights).values())


def _group_report(groups: dict[tuple[int, ...], list[Ephemeris]]) -> list[dict[str, Any]]:
    return [
        {"observed_bits": list(key), "count": len(items), "ephemeris_ids": [e.id for e in items]}
        for key, items in sorted(groups.items(), key=lambda kv: kv[0])
    ]


def plan_fixed(candidates: list[Ephemeris], shifts: dict[int, Shift]) -> dict[str, Any]:
    eligible = [s for s in shifts.values() if s.fixed_eligible]
    plans = []
    for a, b in itertools.combinations(sorted(eligible, key=lambda s: s.night), 2):
        nights = (a.night, b.night)
        groups = groups_for_nights(candidates, nights)
        worst = max((len(v) for v in groups.values()), default=0)
        cost = a.cost + b.cost
        plans.append(
            {
                "method": "fixed_buyout",
                "nights": list(nights),
                "worst_remaining": worst,
                "worst_cost": cost,
                "cost_breakdown": {"night_costs": [a.cost, b.cost], "total": cost},
                "groups": _group_report(groups),
            }
        )
    if not plans:
        return {"method": "fixed_buyout", "feasible": False, "reason": "need two fixed-eligible shifts"}
    plans.sort(key=lambda p: (p["worst_remaining"], p["worst_cost"], tuple(p["nights"])))
    best = dict(plans[0])
    best.update(
        {
            "feasible": True,
            "ranked_plans": [
                {
                    "nights": p["nights"],
                    "worst_remaining": p["worst_remaining"],
                    "worst_cost": p["worst_cost"],
                }
                for p in plans
            ],
        }
    )
    return best


def plan_voucher(
    candidates: list[Ephemeris],
    shifts: dict[int, Shift],
    voucher_fee: float = 0.0,
    allow_skip_redemption: bool = False,
) -> dict[str, Any]:
    first_shifts = sorted((s for s in shifts.values() if s.first_eligible), key=lambda s: s.night)
    all_plans: list[dict[str, Any]] = []

    for first in first_shifts:
        f = first.night
        redeem_shifts = sorted(
            (s for s in shifts.values() if s.voucher_eligible and s.night > f),
            key=lambda s: s.night,
        )
        branches: dict[str, dict[str, Any]] = {}
        branch_minima: list[int] = []
        for bit in (BIT_BRIGHT, BIT_DIM):
            branch_candidates = [e for e in candidates if e.pattern[f - 1] == bit]
            if not branch_candidates:
                branches[str(bit)] = {"possible": False, "candidate_ids": [], "actions": []}
                continue
            action_rows = []
            if allow_skip_redemption:
                action_rows.append(
                    {
                        "redeem_night": None,
                        "action": "skip_redemption",
                        "cost": 0.0,
                        "residual": len(branch_candidates),
                        "groups_after_redeem": [
                            {
                                "observed_bits": [],
                                "count": len(branch_candidates),
                                "ephemeris_ids": [e.id for e in branch_candidates],
                            }
                        ],
                    }
                )
            for rs in redeem_shifts:
                groups = groups_for_nights(branch_candidates, (rs.night,))
                # The first bit is fixed in this branch, so residual is just
                # the largest group after the redemption bit.
                residual = max((len(v) for v in groups.values()), default=0)
                action_rows.append(
                    {
                        "redeem_night": rs.night,
                        "action": "redeem",
                        "cost": rs.cost,
                        "residual": residual,
                        "groups_after_redeem": _group_report(groups),
                    }
                )
            if not action_rows:
                branches[str(bit)] = {
                    "possible": True,
                    "candidate_ids": [e.id for e in branch_candidates],
                    "actions": [],
                }
                branch_minima.append(len(branch_candidates))
                continue
            min_residual = min(a["residual"] for a in action_rows)
            branch_minima.append(min_residual)
            branches[str(bit)] = {
                "possible": True,
                "first_result": "dim" if bit == 1 else "bright",
                "candidate_ids": [e.id for e in branch_candidates],
                "candidate_count": len(branch_candidates),
                "minimum_residual": min_residual,
                "actions": sorted(
                    action_rows,
                    key=lambda a: (
                        a["residual"],
                        a["cost"],
                        a["redeem_night"] is None,
                        a["redeem_night"] or 0,
                    ),
                ),
            }

        if not redeem_shifts and not allow_skip_redemption:
            # No voucher shift after this first night.  The booked voucher
            # product assumes one redemption, so this first night is infeasible.
            all_plans.append(
                {
                    "method": "flexible_voucher",
                    "feasible": False,
                    "first_night": f,
                    "reason": "no voucher-eligible night after first night",
                }
            )
            continue

        worst_remaining = max(branch_minima, default=0)

        policy: dict[str, Any] = {}
        branch_costs: list[float] = []
        listed_nights: set[int] = set()
        for bit in (BIT_BRIGHT, BIT_DIM):
            b = branches[str(bit)]
            if not b.get("possible"):
                policy[str(bit)] = {"possible": False}
                continue
            feasible_actions = [a for a in b["actions"] if a["residual"] <= worst_remaining]
            # Lexicographic second objective: cheapest acceptable redemption,
            # then earliest night for deterministic reproducibility.
            chosen = min(feasible_actions, key=lambda a: (a["cost"], a["redeem_night"] is None, a["redeem_night"] or 0))
            branch_costs.append(float(chosen["cost"]))
            if chosen["redeem_night"] is not None:
                listed_nights.add(chosen["redeem_night"])
            policy[str(bit)] = {
                "possible": True,
                "first_result": "dim" if bit == 1 else "bright",
                "after_first_ids": b["candidate_ids"],
                "action": chosen.get("action", "redeem"),
                "redeem_night": chosen["redeem_night"],
                "redeem_cost": chosen["cost"],
                "remaining_after_redeem_worst": chosen["residual"],
                "groups_after_redeem": chosen["groups_after_redeem"],
                "other_acceptable_nights": [
                    a["redeem_night"]
                    for a in feasible_actions
                    if a.get("action") != "skip_redemption"
                    and (a["cost"], a["redeem_night"]) != (chosen["cost"], chosen["redeem_night"])
                ],
            }

        worst_redeem_cost = max(branch_costs, default=0.0)
        worst_cost = first.cost + voucher_fee + worst_redeem_cost
        plan = {
            "method": "flexible_voucher",
            "feasible": True,
            "allow_skip_redemption": allow_skip_redemption,
            "first_night": f,
            "first_night_cost": first.cost,
            "voucher_fee": voucher_fee,
            "voucher_listed_nights": sorted(listed_nights),
            "worst_remaining": worst_remaining,
            "worst_cost": worst_cost,
            "cost_breakdown": {
                "first_shift": first.cost,
                "voucher_fee": voucher_fee,
                "worst_redeem_shift": worst_redeem_cost,
                "total": worst_cost,
            },
            "policy_by_first_bit": policy,
            "branches": branches,
        }
        all_plans.append(plan)

    feasible = [p for p in all_plans if p.get("feasible")]
    if not feasible:
        return {
            "method": "flexible_voucher",
            "feasible": False,
            "reason": "no first night with a later voucher-eligible shift",
            "attempted_plans": all_plans,
        }
    feasible.sort(key=lambda p: (p["worst_remaining"], p["worst_cost"], p["first_night"]))
    best = feasible[0]
    best["ranked_first_nights"] = [
        {
            "first_night": p["first_night"],
            "worst_remaining": p["worst_remaining"],
            "worst_cost": p["worst_cost"],
            "voucher_listed_nights": p["voucher_listed_nights"],
        }
        for p in feasible
    ]
    return best


def summarize_candidates(candidates: list[Ephemeris]) -> list[dict[str, Any]]:
    return [e.to_dict() for e in candidates]


def save_json(obj: Any, path: str | Path) -> Path:
    return write_json(obj, path)
