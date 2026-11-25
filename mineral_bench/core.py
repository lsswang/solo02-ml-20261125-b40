"""Core input normalization and Bayesian hypothesis model.

The project deliberately uses only the Python standard library.  Inputs are
small JSON documents; another museum tray in the same schema can be passed to
``run.py all --input ...`` without changing the program.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "mineral-optical-tray/v1"

_BINARY_WORDS = {
    "0": 0,
    "1": 1,
    "false": 0,
    "true": 1,
    "no": 0,
    "yes": 1,
    "dark": 0,
    "light": 1,
    "bright": 1,
    "dim": 0,
    "opaque": 0,
    "transparent": 1,
    "translucent": 1,
    "暗": 0,
    "黑暗": 0,
    "黑": 0,
    "不透": 0,
    "不透明": 0,
    "明": 1,
    "亮": 1,
    "明亮": 1,
    "透光": 1,
    "透明": 1,
}


class InputFormatError(ValueError):
    """Raised when a tray input does not satisfy the documented schema."""


def money(value: Any) -> Fraction:
    """Parse JSON money/cost values without binary-float surprises."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(str(value))
    if isinstance(value, str):
        return Fraction(value.strip())
    raise InputFormatError(f"invalid number/money value: {value!r}")


def probability(value: Any) -> Fraction:
    p = money(value)
    if p < 0 or p > 1:
        raise InputFormatError(f"probability must be in [0,1], got {value!r}")
    return p


def binary_reading(value: Any) -> int:
    """Normalize a bright/dark reading.

    JSON native ``0/1`` and ``false/true`` are accepted, as are English and
    Chinese result words.  The canonical on-disk form in generated files is
    ``0`` for dark and ``1`` for bright.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and float(value).is_integer() and int(value) in (0, 1):
        return int(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _BINARY_WORDS:
            return _BINARY_WORDS[key]
        # Chinese phrases may contain extra characters.
        if any(word in value for word in ("不透", "黑暗", "暗", "黑")):
            return 0
        if any(word in value for word in ("透光", "透明", "明亮", "亮", "明")):
            return 1
    raise InputFormatError(f"cannot normalize binary optical reading: {value!r}")


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise InputFormatError("input JSON root must be an object")
    return data


def dump_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")


def fnum(x: Fraction | float | int | None, ndigits: int = 6) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, Fraction):
        x = float(x)
    return round(float(x), ndigits)


@dataclass(frozen=True)
class TestSpec:
    id: str
    name: str
    instrument: str
    side: str  # front | back | either
    cost: Fraction
    sealed: bool = False
    description: str = ""

    @property
    def active(self) -> bool:
        return not self.sealed


@dataclass(frozen=True)
class InstrumentSpec:
    id: str
    name: str
    setup_fee: Fraction


@dataclass(frozen=True)
class Hypothesis:
    index: int
    material: str
    orientation: str
    joint_prior: Fraction
    outcomes: Tuple[int, ...]


@dataclass
class TrayModel:
    raw: Dict[str, Any]
    tests: List[TestSpec]
    instruments: Dict[str, InstrumentSpec]
    materials: List[str]
    material_names: Dict[str, str]
    material_prior: Dict[str, Fraction]
    templates: List[Dict[str, Any]]
    hypotheses: List[Hypothesis]
    threshold: Fraction
    budget: Fraction
    max_flips: int
    initial_side: str
    flip_cost: Fraction
    initial_paid: frozenset[str]
    currency: str

    @property
    def active_tests(self) -> List[TestSpec]:
        return [t for t in self.tests if t.active]

    @property
    def active_test_index(self) -> Dict[str, int]:
        return {t.id: i for i, t in enumerate(self.active_tests)}

    @property
    def instrument_ids(self) -> List[str]:
        return list(self.instruments.keys())

    def instrument_bit(self, instrument_id: str) -> int:
        return 1 << self.instrument_ids.index(instrument_id)

    def test_by_id(self, test_id: str) -> TestSpec:
        for t in self.tests:
            if t.id == test_id:
                return t
        raise KeyError(f"unknown test/action {test_id!r}")

    def active_test_by_id(self, test_id: str) -> TestSpec:
        t = self.test_by_id(test_id)
        if not t.active:
            raise KeyError(f"test {test_id!r} is sealed/unavailable")
        return t


def _as_list(data: Mapping[str, Any], key: str) -> List[Any]:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise InputFormatError(f"{key!r} must be a list")
    return value


def validate_and_build(
    data: Dict[str, Any],
    orientation_weights: Optional[Mapping[str, Mapping[str, float | Fraction]]] = None,
) -> TrayModel:
    """Validate an input document and build the joint material/orientation space."""

    if not isinstance(data, dict):
        raise InputFormatError("input must be a JSON object")

    materials_raw = _as_list(data, "materials")
    if not materials_raw:
        raise InputFormatError("at least one material is required")

    materials: List[str] = []
    material_names: Dict[str, str] = {}
    raw_priors: Dict[str, Fraction] = {}
    for m in materials_raw:
        if not isinstance(m, dict) or "id" not in m:
            raise InputFormatError("each material must be an object with an id")
        mid = str(m["id"])
        if mid in materials:
            raise InputFormatError(f"duplicate material id {mid!r}")
        materials.append(mid)
        material_names[mid] = str(m.get("name", mid))
        raw_priors[mid] = probability(m.get("prior", m.get("probability", 1)))
    prior_sum = sum(raw_priors.values(), Fraction(0))
    if prior_sum <= 0:
        raise InputFormatError("material priors must have positive sum")
    material_prior = {m: raw_priors[m] / prior_sum for m in materials}

    # Instruments.  A single default instrument exists if the input omits them.
    instruments: Dict[str, InstrumentSpec] = {}
    second_fee = data.get(
        "second_instrument_setup_fee", data.get("second_instrument_setup_cost")
    )
    for instrument_index, inst in enumerate(_as_list(data, "instruments")):
        if not isinstance(inst, dict) or "id" not in inst:
            raise InputFormatError("each instrument must have an id")
        iid = str(inst["id"])
        if iid in instruments:
            raise InputFormatError(f"duplicate instrument id {iid!r}")
        if "setup_fee" in inst or "setup_cost" in inst:
            fee = money(inst.get("setup_fee", inst.get("setup_cost", 0)))
        elif instrument_index == 1 and second_fee is not None:
            # Story-level alias: the second workbench instrument has this
            # one-time setup/调机 fee on its first use in a case.
            fee = money(second_fee)
        else:
            fee = Fraction(0)
        instruments[iid] = InstrumentSpec(
            id=iid,
            name=str(inst.get("name", iid)),
            setup_fee=fee,
        )

    tests: List[TestSpec] = []
    test_ids = set()
    for t in _as_list(data, "tests"):
        if not isinstance(t, dict) or "id" not in t:
            raise InputFormatError("each test must have an id")
        tid = str(t["id"])
        if tid in test_ids:
            raise InputFormatError(f"duplicate test id {tid!r}")
        test_ids.add(tid)
        side = str(t.get("side", "either")).lower()
        if side not in {"front", "back", "either", "both"}:
            raise InputFormatError(f"test {tid!r} has invalid side {side!r}")
        if side == "both":
            side = "either"
        iid = str(t.get("instrument", "default"))
        if iid not in instruments:
            # Same-format files may define costs directly without an explicit
            # instrument list; create a zero-fee instrument on demand.
            instruments[iid] = InstrumentSpec(iid, iid, Fraction(0))
        tests.append(
            TestSpec(
                id=tid,
                name=str(t.get("name", tid)),
                instrument=iid,
                side=side,
                cost=money(t.get("cost", t.get("cost_per_use", 1))),
                sealed=bool(t.get("sealed", t.get("available", True) is False)),
                description=str(t.get("description", "")),
            )
        )

    optical_tests = [t for t in tests if not t.id.startswith("flip")]
    if len(optical_tests) > 5:
        raise InputFormatError("a template may prescribe at most five optical tests")

    if "default" not in instruments:
        # Ensure deterministic instrument ordering.
        instruments.setdefault("default", InstrumentSpec("default", "默认工具", Fraction(0)))

    templates = _as_list(data, "templates")
    if not templates:
        raise InputFormatError("at least one orientation template is required")

    seen_template: set[Tuple[str, str]] = set()
    normalized_templates: List[Dict[str, Any]] = []
    theta: Dict[str, Dict[str, Fraction]] = {m: {} for m in materials}

    for tp in templates:
        if not isinstance(tp, dict):
            raise InputFormatError("each template must be an object")
        mid = str(tp.get("material", tp.get("material_id", "")))
        oid = str(tp.get("orientation", tp.get("orientation_id", tp.get("id", ""))))
        if mid not in material_prior:
            raise InputFormatError(f"template {oid!r} refers to unknown material {mid!r}")
        if not oid:
            raise InputFormatError("each template needs an orientation id")
        key = (mid, oid)
        if key in seen_template:
            raise InputFormatError(f"duplicate orientation template {key}")
        seen_template.add(key)

        raw_readings = tp.get("readings", {})
        if not isinstance(raw_readings, dict):
            raise InputFormatError(f"template {mid}/{oid} readings must be an object")
        readings: Dict[str, int] = {}
        for tid, val in raw_readings.items():
            if tid not in test_ids:
                raise InputFormatError(f"template {mid}/{oid} uses unknown test {tid!r}")
            readings[str(tid)] = binary_reading(val)

        active_ids = [t.id for t in tests if t.active]
        missing = [tid for tid in active_ids if tid not in readings]
        if missing:
            raise InputFormatError(
                f"template {mid}/{oid} is missing active test readings: {', '.join(missing)}"
            )

        if orientation_weights and mid in orientation_weights and oid in orientation_weights[mid]:
            w = money(orientation_weights[mid][oid])
        elif tp.get("weight") is not None:
            w = money(tp["weight"])
        else:
            w = Fraction(1)
        if w < 0:
            raise InputFormatError(f"negative orientation weight for {mid}/{oid}")
        theta[mid][oid] = w
        normalized_templates.append(
            {"material": mid, "orientation": oid, "readings": readings, "_raw": tp}
        )

    for m in materials:
        if not theta[m]:
            raise InputFormatError(f"material {m!r} has no orientation templates")
        total = sum(theta[m].values(), Fraction(0))
        if total <= 0:
            # All-zero learned weights can happen only with pathological data;
            # fall back to uniform so the workbench remains operable.
            n = len(theta[m])
            theta[m] = {o: Fraction(1, n) for o in theta[m]}
        else:
            theta[m] = {o: w / total for o, w in theta[m].items()}

    active = [t for t in tests if t.active]
    hypotheses: List[Hypothesis] = []
    idx = 0
    for tp in normalized_templates:
        m = tp["material"]
        o = tp["orientation"]
        joint = material_prior[m] * theta[m][o]
        outcomes = tuple(tp["readings"][t.id] for t in active)
        hypotheses.append(Hypothesis(idx, m, o, joint, outcomes))
        idx += 1

    initial_paid_raw = data.get(
        "initial_paid_instruments",
        data.get("ready_instruments", []),
    )
    if initial_paid_raw is None:
        initial_paid: frozenset[str] = frozenset()
    elif isinstance(initial_paid_raw, str):
        initial_paid = frozenset([initial_paid_raw])
    else:
        initial_paid = frozenset(str(x) for x in initial_paid_raw)
    unknown_paid = [i for i in initial_paid if i not in instruments]
    if unknown_paid:
        raise InputFormatError(f"unknown initially paid instruments: {unknown_paid}")

    flip_raw = data.get("flip_action", {})
    if not isinstance(flip_raw, dict):
        raise InputFormatError("flip_action must be an object")

    initial_side = str(data.get("initial_side", "front")).lower()
    if initial_side not in {"front", "back"}:
        raise InputFormatError("initial_side must be front or back")

    return TrayModel(
        raw=data,
        tests=tests,
        instruments=instruments,
        materials=materials,
        material_names=material_names,
        material_prior=material_prior,
        templates=normalized_templates,
        hypotheses=hypotheses,
        threshold=probability(data.get("confidence_threshold", 0.9)),
        budget=money(data.get("budget", data.get("test_budget", 3))),
        max_flips=int(data.get("max_flips", 1)),
        initial_side=initial_side,
        flip_cost=money(flip_raw.get("cost", 0) if isinstance(flip_raw, dict) else 0),
        initial_paid=initial_paid,
        currency=str(data.get("currency", data.get("cost_unit", "预算元"))),
    )


def initial_weights(model: TrayModel) -> List[Fraction]:
    return [h.joint_prior for h in model.hypotheses]


def alive_mask_from_evidence(
    model: TrayModel,
    evidence: Mapping[str, int | str | bool],
) -> int:
    """Return a bit mask of hypotheses compatible with observed readings."""
    active_index = model.active_test_index
    mask = (1 << len(model.hypotheses)) - 1
    for test_id, raw_value in evidence.items():
        if test_id in {"flip", "action"}:
            continue
        if test_id not in active_index:
            # Sealed/unknown observed keys do not affect inference.
            continue
        tid = str(test_id)
        y = binary_reading(raw_value)
        i = active_index[tid]
        for h in model.hypotheses:
            if ((mask >> h.index) & 1) and h.outcomes[i] != y:
                mask &= ~(1 << h.index)
    return mask


def posterior_from_mask(
    model: TrayModel, alive_mask: int
) -> Tuple[Dict[str, Fraction], List[Fraction], Fraction]:
    """Return material posteriors, alive joint weights and normalizing sum."""
    weights: List[Fraction] = []
    total = Fraction(0)
    material_mass = {m: Fraction(0) for m in model.materials}
    for h in model.hypotheses:
        if (alive_mask >> h.index) & 1:
            w = h.joint_prior
        else:
            w = Fraction(0)
        weights.append(w)
        total += w
        material_mass[h.material] += w
    if total <= 0:
        return {m: Fraction(0) for m in model.materials}, weights, Fraction(0)
    return {m: material_mass[m] / total for m in model.materials}, weights, total


def best_material(
    posteriors: Mapping[str, Fraction], threshold: Fraction
) -> Tuple[Optional[str], Fraction, bool]:
    if not posteriors:
        return None, Fraction(0), False
    mid = max(posteriors, key=lambda m: posteriors[m])
    p = posteriors[mid]
    return mid, p, p >= threshold
