"""Deterministic small synthetic museum trays.

Run ``python run.py generate`` to regenerate every example.  All randomness is
seeded, so the same JSON inputs and policy reports can be reproduced on an
ordinary CPU.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .core import SCHEMA_VERSION, dump_json


def _record_set(
    rng: random.Random,
    materials: Sequence[Dict[str, Any]],
    templates: Sequence[Dict[str, Any]],
    n_train: int,
    n_holdout: int,
    n_new: int,
    partial_min: int = 2,
    partial_max: int = 4,
) -> Dict[str, List[Dict[str, Any]]]:
    mat_ids = [m["id"] for m in materials]
    priors = [float(m.get("prior", 1)) for m in materials]
    ps = sum(priors)
    priors = [p / ps for p in priors]

    by_mat: Dict[str, List[Dict[str, Any]]] = {m: [] for m in mat_ids}
    for tp in templates:
        by_mat[tp["material"]].append(tp)

    def sample_hypothesis():
        m = rng.choices(mat_ids, weights=priors, k=1)[0]
        tps = by_mat[m]
        weights = [float(tp.get("weight", 1)) for tp in tps]
        tp = rng.choices(tps, weights=weights, k=1)[0]
        return m, tp

    all_test_ids = list(templates[0]["readings"].keys())

    train: List[Dict[str, Any]] = []
    holdout: List[Dict[str, Any]] = []
    new_specimens: List[Dict[str, Any]] = []

    all_test_ids = list(templates[0]["readings"].keys())

    for i in range(n_train):
        m, tp = sample_hypothesis()
        k = rng.randint(partial_min, min(partial_max, len(all_test_ids)))
        observed_ids = rng.sample(all_test_ids, k=k)
        train.append(
            {
                "id": f"train-{i+1:03d}",
                "material": m,
                # Orientation is intentionally absent: the learner must infer
                # the within-material distribution from partial readings.
                "readings": {tid: tp["readings"][tid] for tid in observed_ids},
            }
        )

    for i in range(n_holdout):
        m, tp = sample_hypothesis()
        # Designate a small evidence subset before scoring.  Every key in
        # check_readings is excluded from the Bayesian update and is used only
        # to audit the predictive distribution.
        n_evidence = min(2, len(all_test_ids) - 1)
        evidence_ids = rng.sample(all_test_ids, k=n_evidence)
        check_ids = [tid for tid in all_test_ids if tid not in evidence_ids]
        readings = dict(tp["readings"])
        holdout.append(
            {
                "id": f"heldout-{i+1:03d}",
                "material": m,
                # Truth is retained only for scoring; prediction code excludes
                # check_readings from its conditioning evidence.
                "orientation": tp["orientation"],
                "readings": readings,
                "evidence_readings": {tid: readings[tid] for tid in evidence_ids},
                "check_readings": {tid: readings[tid] for tid in check_ids},
            }
        )

    for i in range(n_new):
        m, tp = sample_hypothesis()
        new_specimens.append(
            {
                "id": f"new-{i+1:03d}",
                "material": m,
                "orientation": tp["orientation"],
                "actual_readings": dict(tp["readings"]),
            }
        )

    return {
        "training_records": train,
        "heldout_records": holdout,
        "new_specimens": new_specimens,
    }


def perfect_transmission() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": "先验各半：甲总透光、乙总不透",
        "description": "一次透光试验即可在两个明暗分支分别签出甲、乙。",
        "random_seed": 101,
        "confidence_threshold": 0.9,
        "budget": 1,
        "currency": "试验元",
        "max_flips": 1,
        "initial_side": "front",
        "orientation_smoothing": 0.5,
        "instruments": [{"id": "lamp", "name": "透光灯", "setup_fee": 0}],
        "initial_paid_instruments": ["lamp"],
        "tests": [
            {
                "id": "trans",
                "name": "透光试验",
                "instrument": "lamp",
                "side": "front",
                "cost": 1,
            }
        ],
        "flip_action": {"id": "flip", "name": "翻面", "cost": 0},
        "materials": [
            {"id": "A", "name": "甲", "prior": 0.5},
            {"id": "B", "name": "乙", "prior": 0.5},
        ],
        "templates": [
            {"material": "A", "orientation": "A-fixed", "weight": 1, "readings": {"trans": 1}},
            {"material": "B", "orientation": "B-fixed", "weight": 1, "readings": {"trans": 0}},
        ],
        "training_records": [
            {"id": f"tr-a-{i}", "material": "A", "readings": {"trans": 1}} for i in range(1, 21)
        ]
        + [
            {"id": f"tr-b-{i}", "material": "B", "readings": {"trans": 0}} for i in range(1, 21)
        ],
        "heldout_records": [
            {"id": "ho-a", "material": "A", "orientation": "A-fixed", "readings": {"trans": 1}},
            {"id": "ho-b", "material": "B", "orientation": "B-fixed", "readings": {"trans": 0}},
        ],
    }


def half_b_transmits() -> Dict[str, Any]:
    train = []
    for i in range(1, 41):
        train.append({"id": f"tr-a-{i:02d}", "material": "A", "readings": {"trans": 1}})
    for i in range(1, 21):
        train.append(
            {"id": f"tr-b-dark-{i:02d}", "material": "B", "readings": {"trans": 0}}
        )
    for i in range(1, 21):
        train.append(
            {"id": f"tr-b-bright-{i:02d}", "material": "B", "readings": {"trans": 1}}
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": "训练改样：乙有一半取向也透光",
        "description": "明分支 P(A|明)=2/3，未达 0.90，必须留未定；暗分支可签乙。",
        "random_seed": 202,
        "confidence_threshold": 0.9,
        "budget": 1,
        "currency": "试验元",
        "max_flips": 0,
        "initial_side": "front",
        "orientation_smoothing": 0.5,
        "instruments": [{"id": "lamp", "name": "透光灯", "setup_fee": 0}],
        "initial_paid_instruments": ["lamp"],
        "tests": [
            {
                "id": "trans",
                "name": "透光试验",
                "instrument": "lamp",
                "side": "front",
                "cost": 1,
            }
        ],
        "materials": [
            {"id": "A", "name": "甲", "prior": 0.5},
            {"id": "B", "name": "乙", "prior": 0.5},
        ],
        "templates": [
            {"material": "A", "orientation": "A-fixed", "weight": 1.0, "readings": {"trans": 1}},
            {"material": "B", "orientation": "B-opaque", "weight": 0.5, "readings": {"trans": 0}},
            {"material": "B", "orientation": "B-bright", "weight": 0.5, "readings": {"trans": 1}},
        ],
        "training_records": train,
        "heldout_records": [
            {"id": "ho-a", "material": "A", "orientation": "A-fixed", "readings": {"trans": 1}},
            {"id": "ho-b-dark", "material": "B", "orientation": "B-opaque", "readings": {"trans": 0}},
            {"id": "ho-b-bright", "material": "B", "orientation": "B-bright", "readings": {"trans": 1}},
        ],
    }


def upside_down_sealed() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": "原展盘倒扣并封住透光试验",
        "description": "唯一余下读数对甲乙相同，更新后仍为先验各半，必须留未定。",
        "random_seed": 303,
        "confidence_threshold": 0.9,
        "budget": 1,
        "currency": "试验元",
        "max_flips": 1,
        "initial_side": "back",
        "orientation_smoothing": 0.5,
        "instruments": [{"id": "loupe", "name": "边缘读数台", "setup_fee": 0}],
        "initial_paid_instruments": ["loupe"],
        "tests": [
            {
                "id": "trans",
                "name": "已封住的透光试验",
                "instrument": "loupe",
                "side": "front",
                "cost": 1,
                "sealed": True,
            },
            {
                "id": "rim",
                "name": "余下的边缘明暗项",
                "instrument": "loupe",
                "side": "back",
                "cost": 1,
            },
        ],
        "flip_action": {"id": "flip", "name": "翻回正面", "cost": 0},
        "materials": [
            {"id": "A", "name": "甲", "prior": 0.5},
            {"id": "B", "name": "乙", "prior": 0.5},
        ],
        "templates": [
            {
                "material": "A",
                "orientation": "A-fixed",
                "weight": 1,
                "readings": {"trans": 1, "rim": 1},
            },
            {
                "material": "B",
                "orientation": "B-fixed",
                "weight": 1,
                "readings": {"trans": 0, "rim": 1},
            },
        ],
        "training_records": [
            {"id": f"tr-{i}", "material": "A" if i % 2 else "B", "readings": {"rim": 1}}
            for i in range(1, 41)
        ],
        "heldout_records": [
            {
                "id": "ho-a",
                "material": "A",
                "orientation": "A-fixed",
                "readings": {"trans": 1, "rim": 1},
            },
            {
                "id": "ho-b",
                "material": "B",
                "orientation": "B-fixed",
                "readings": {"trans": 0, "rim": 1},
            },
        ],
    }


def minimal_two_sides() -> Dict[str, Any]:
    train = []
    # Balanced material labels; B's two orientations are equally frequent.
    for i in range(1, 31):
        train.append({"id": f"tr-a-{i:02d}", "material": "A", "readings": {"front_signal": 1, "back_signal": 1}})
    for i in range(1, 16):
        train.append(
            {
                "id": f"tr-b-opaque-{i:02d}",
                "material": "B",
                "readings": {"front_signal": 0, "back_signal": 1},
            }
        )
    for i in range(1, 16):
        train.append(
            {
                "id": f"tr-b-bright-{i:02d}",
                "material": "B",
                "readings": {"front_signal": 1, "back_signal": 0},
            }
        )
    # Make two thirds of records partial while preserving balanced labels.
    for idx, rec in enumerate(train):
        readings = rec["readings"]
        if idx % 3 == 0:
            rec["readings"] = {"front_signal": readings["front_signal"]}
        elif idx % 3 == 1:
            rec["readings"] = {"back_signal": readings["back_signal"]}
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": "正反各一项辨别读数的极小展盘",
        "description": (
            "正面暗可签乙；正面明后甲为2/3，翻面做背面项可区分甲明与乙亮取向。"
            "第二件仪器调机费1，故走完整条分支花费3。"
        ),
        "random_seed": 404,
        "confidence_threshold": 0.9,
        "budget": 3,
        "currency": "试验元",
        "max_flips": 1,
        "initial_side": "front",
        "orientation_smoothing": 0.5,
        "second_instrument_setup_fee": 1,
        "instruments": [
            {"id": "front_eye", "name": "正面目镜", "setup_fee": 0},
            {"id": "back_bench", "name": "第二件仪器/背面台", "setup_fee": 1},
        ],
        "initial_paid_instruments": ["front_eye"],
        "tests": [
            {
                "id": "front_signal",
                "name": "正面辨别试验",
                "instrument": "front_eye",
                "side": "front",
                "cost": 1,
            },
            {
                "id": "back_signal",
                "name": "背面辨别试验",
                "instrument": "back_bench",
                "side": "back",
                "cost": 1,
            },
        ],
        "flip_action": {"id": "flip", "name": "翻面", "cost": 0},
        "materials": [
            {"id": "A", "name": "甲", "prior": 0.5},
            {"id": "B", "name": "乙", "prior": 0.5},
        ],
        "templates": [
            {
                "material": "A",
                "orientation": "A-clear",
                "weight": 1.0,
                "readings": {"front_signal": 1, "back_signal": 1},
            },
            {
                "material": "B",
                "orientation": "B-opaque",
                "weight": 0.5,
                "readings": {"front_signal": 0, "back_signal": 1},
            },
            {
                "material": "B",
                "orientation": "B-bright",
                "weight": 0.5,
                "readings": {"front_signal": 1, "back_signal": 0},
            },
        ],
        "training_records": train,
        "heldout_records": [
            {
                "id": "ho-a",
                "material": "A",
                "orientation": "A-clear",
                "readings": {"front_signal": 1, "back_signal": 1},
            },
            {
                "id": "ho-b-opaque",
                "material": "B",
                "orientation": "B-opaque",
                "readings": {"front_signal": 0, "back_signal": 1},
            },
            {
                "id": "ho-b-bright",
                "material": "B",
                "orientation": "B-bright",
                "readings": {"front_signal": 1, "back_signal": 0},
            },
        ],
        "new_specimens": [
            {
                "id": "demo-A",
                "material": "A",
                "orientation": "A-clear",
                "actual_readings": {"front_signal": 1, "back_signal": 1},
            }
        ],
    }


def canonical_tray() -> Dict[str, Any]:
    rng = random.Random(606)
    materials = [
        {"id": "A", "name": "甲质矿片", "prior": 0.5},
        {"id": "B", "name": "乙质矿片", "prior": 0.5},
    ]
    tests = [
        {"id": "cleavage", "name": "正面解理明暗", "instrument": "loupe", "side": "front", "cost": 1},
        {"id": "luster", "name": "正面反光强弱", "instrument": "loupe", "side": "front", "cost": 1},
        {"id": "trans", "name": "透光试验", "instrument": "optical_bench", "side": "front", "cost": 1},
        {"id": "polar", "name": "背面偏光花样", "instrument": "optical_bench", "side": "back", "cost": 1},
        {"id": "asterism", "name": "背面星线", "instrument": "loupe", "side": "back", "cost": 1},
    ]
    templates = [
        {"material": "A", "orientation": "A1", "weight": 0.45,
         "readings": {"cleavage": 1, "luster": 1, "trans": 1, "polar": 1, "asterism": 0}},
        {"material": "A", "orientation": "A2", "weight": 0.35,
         "readings": {"cleavage": 0, "luster": 1, "trans": 1, "polar": 0, "asterism": 1}},
        {"material": "A", "orientation": "A3", "weight": 0.20,
         "readings": {"cleavage": 0, "luster": 0, "trans": 1, "polar": 1, "asterism": 1}},
        {"material": "B", "orientation": "B1", "weight": 0.40,
         "readings": {"cleavage": 0, "luster": 0, "trans": 0, "polar": 0, "asterism": 0}},
        {"material": "B", "orientation": "B2", "weight": 0.35,
         "readings": {"cleavage": 1, "luster": 0, "trans": 1, "polar": 0, "asterism": 0}},
        {"material": "B", "orientation": "B3", "weight": 0.25,
         "readings": {"cleavage": 1, "luster": 1, "trans": 0, "polar": 1, "asterism": 1}},
    ]
    sets = _record_set(
        rng, materials, templates, n_train=420, n_holdout=80, n_new=36,
        partial_min=2, partial_max=4,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": "展台五读数光学矿片识别工作台",
        "description": (
            "每种材质三个已知取向模板；训练记录只含部分读数且有材质标签。"
            "工作台按当前联合概率自适应选择试验，并对第二件仪器收取一次调机费。"
        ),
        "random_seed": 606,
        "confidence_threshold": 0.9,
        "budget": 4,
        "currency": "试验元",
        "max_flips": 1,
        "initial_side": "front",
        "orientation_smoothing": 0.5,
        "second_instrument_setup_fee": 1,
        "instruments": [
            {"id": "loupe", "name": "首选目镜台", "setup_fee": 0},
            {"id": "optical_bench", "name": "第二件仪器：透光/偏光台", "setup_fee": 1},
        ],
        "initial_paid_instruments": ["loupe"],
        "tests": tests,
        "flip_action": {"id": "flip", "name": "翻面", "cost": 0},
        "materials": materials,
        "templates": templates,
        **sets,
    }


def generate_all(out_dir: str | Path = "examples") -> List[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sample_input.json": canonical_tray(),
        "perfect_transmission.json": perfect_transmission(),
        "half_b_transmits.json": half_b_transmits(),
        "upside_down_sealed.json": upside_down_sealed(),
        "minimal_tray.json": minimal_two_sides(),
    }
    paths = []
    for name, doc in outputs.items():
        p = out / name
        dump_json(p, doc)
        paths.append(p)
    return paths
