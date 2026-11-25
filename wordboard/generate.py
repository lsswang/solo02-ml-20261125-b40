"""Deterministic small synthetic AAC corpora in the documented JSON format."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

WORDS = ["甲", "乙", "丙", "丁", "戊", "己"]


def _normalize(row: dict[str, float]) -> dict[str, float]:
    total = sum(row.values())
    return {k: v / total for k, v in row.items()}


HOME_START = _normalize({"甲": 0.68, "乙": 0.24, "丙": 0.08})
SCHOOL_START = _normalize({"甲": 0.42, "丁": 0.40, "戊": 0.10, "乙": 0.08})

HOME_TRANS = {
    "甲": _normalize({"乙": 0.76, "丙": 0.14, "甲": 0.05, "丁": 0.05}),
    "乙": _normalize({"丙": 0.82, "甲": 0.10, "乙": 0.04, "丁": 0.04}),
    "丙": _normalize({"甲": 0.58, "乙": 0.24, "丙": 0.10, "戊": 0.08}),
    "丁": _normalize({"乙": 0.45, "甲": 0.25, "丙": 0.20, "戊": 0.10}),
    "戊": _normalize({"丙": 0.45, "甲": 0.30, "乙": 0.25}),
    "己": _normalize({"甲": 0.45, "乙": 0.35, "丙": 0.20}),
}

SCHOOL_TRANS = {
    "甲": _normalize({"丁": 0.78, "戊": 0.10, "乙": 0.07, "甲": 0.05}),
    "丁": _normalize({"戊": 0.82, "己": 0.10, "甲": 0.05, "丁": 0.03}),
    "戊": _normalize({"己": 0.76, "丁": 0.12, "甲": 0.07, "戊": 0.05}),
    "己": _normalize({"戊": 0.48, "丁": 0.28, "甲": 0.16, "乙": 0.08}),
    "乙": _normalize({"丁": 0.48, "戊": 0.28, "甲": 0.14, "己": 0.10}),
    "丙": _normalize({"丁": 0.42, "戊": 0.32, "己": 0.16, "甲": 0.10}),
}


def _choose(rng: random.Random, probs: dict[str, float]) -> str:
    words = list(probs)
    weights = list(probs.values())
    return rng.choices(words, weights=weights, k=1)[0]


def _sample_sentence(rng: random.Random, start: dict[str, float], trans: dict[str, dict[str, float]]) -> list[str]:
    # Corpus utterances are small; task phrases themselves stay 1-3 words.
    length = rng.choices([2, 3, 4, 5], weights=[2, 5, 3, 1], k=1)[0]
    word = _choose(rng, start)
    out = [word]
    for _ in range(length - 1):
        word = _choose(rng, trans[word])
        out.append(word)
    return out


def _build_scene(seed: int, start, trans, n_train: int = 42, n_heldout: int = 10):
    rng = random.Random(seed)
    train_tuples: set[tuple[str, ...]] = set()
    sentences: list[list[str]] = []
    while len(sentences) < n_train:
        sent = _sample_sentence(rng, start, trans)
        key = tuple(sent)
        # Keep some duplicate natural frequency, but not infinitely.
        duplicate_allowed = rng.random() < 0.35
        if key in train_tuples and not duplicate_allowed:
            continue
        train_tuples.add(key)
        sentences.append(sent)

    groups = [
        ["甲", "乙"],
        ["乙", "丙"],
        ["甲", "丁"],
        ["丁", "戊"],
        ["丁", "戊", "己"],
    ]
    # Remove exact ordered duplicates of group-only evidence? Not necessary:
    # whole sentences and group records are both allowed.
    heldout: list[list[str]] = []
    heldout_keys: set[tuple[str, ...]] = set()
    local_rng = random.Random(seed + 10_000)
    while len(heldout) < n_heldout:
        sent = _sample_sentence(local_rng, start, trans)
        key = tuple(sent)
        if key in train_tuples or key in heldout_keys:
            continue
        heldout_keys.add(key)
        heldout.append(sent)
    return {"sentences": sentences, "groups": groups}, heldout


def build_sample_input(seed: int = 2026) -> dict[str, Any]:
    home_corpus, home_heldout = _build_scene(seed + 1, HOME_START, HOME_TRANS)
    school_corpus, school_heldout = _build_scene(seed + 2, SCHOOL_START, SCHOOL_TRANS)
    return {
        "schema": "wordboard-input-v1",
        "name": f"synthetic-aac-seed-{seed}",
        "seed": seed,
        "description": "小型合成辅助沟通词板：家庭与课堂有不同一阶词序。",
        "config": {
            "alpha": 1.0,
            "ambiguity_weighting": "uniform",
            "page_turn_cost": 3,
            "select_cost": 1,
            "wait_cost": 1,
            "toggle_cost": 0.5,
            "card_swap_cost": 4,
            "slots_per_page": 2,
            "max_words": 6,
            "page_turn_resets_slot": True,
            "route": "fixed_preset",
        },
        "start_page": 1,
        "scene_order": ["home", "school"],
        "board": {
            "num_pages": 3,
            "cards": [
                {"word": "甲", "page": 1, "slot": 1, "familiar": True, "fixed": True,
                 "note": "锁定在第一页首格的熟词"}
            ],
        },
        "corpus": {
            "home": home_corpus,
            "school": school_corpus,
        },
        "heldout": {
            "home": home_heldout,
            "school": school_heldout,
        },
        "tasks": {
            "home": ["甲", "乙", "丙"],
            "school": ["丁", "戊", "己"],
        },
        "query": {"scene": "home", "current": "甲", "top_k": 6},
        "probability_scenarios": [
            {
                "id": "given-page1-high-yi",
                "num_pages": 2,
                "start_page": 1,
                "locked_word": "甲",
                "locked_page": 1,
                "locked_slot": 1,
                "distribution": {"乙": 0.8, "丙": 0.2},
                "expected_position": {"word": "乙", "page": 1, "slot": 2},
            },
            {
                "id": "given-page1-swapped",
                "num_pages": 2,
                "start_page": 1,
                "locked_word": "甲",
                "distribution": {"乙": 0.2, "丙": 0.8},
                "expected_position": {"word": "丙", "page": 1, "slot": 2},
            },
            {
                "id": "given-page2-start",
                "num_pages": 2,
                "start_page": 2,
                "locked_word": "甲",
                "distribution": {"乙": 0.8, "丙": 0.2},
                "expected_position": {"word": "乙", "page": 2, "slot": 1},
            },
        ],
        "replays": [
            {
                "session_id": "one-word-from-page-1",
                "scene": "home",
                "start_page": 1,
                "sentence": ["乙"],
                "close_after": 1,
            },
            {
                "session_id": "carry-page-demo",
                "scene": "home",
                "start_page": 1,
                "sentence": ["甲", "乙", "丙"],
                "close_after": 3,
            },
            {
                "session_id": "carry-page-demo",
                "scene": "school",
                "switch_scene": True,
                "sentence": ["丁", "戊", "己"],
            },
            {
                "session_id": "mid-sentence-restore",
                "scene": "home",
                "start_page": 1,
                "sentence": ["乙", "丙"],
                "close_after": 1,
            },
        ],
    }


def write_sample(path: str | Path, seed: int = 2026) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build_sample_input(seed)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path
