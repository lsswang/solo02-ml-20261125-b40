"""小型合成材料：成对生成低清读数与精检真图。

真值用一阶马尔可夫链生成（相邻格的好坏带相关性），低清读头跨相邻两格
取一次观测，每格读数以 flip 概率独立翻转；每次完整观测再以 missing_rate
被整体记成缺读（null）。全部随机数来自带种子的 random.Random，普通 CPU 可重跑。
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

from .common import dump_json


def generate_truth(rng: random.Random, n: int, p0: float, stay: float) -> str:
    bits: List[str] = []
    cur = 1 if rng.random() < p0 else 0
    bits.append(str(cur))
    for _ in range(n - 1):
        keep = rng.random() < stay
        cur = cur if keep else 1 - cur
        bits.append(str(cur))
    return "".join(bits)


def generate_reading(rng: random.Random,
                     truth_pair: str,
                     flip: float,
                     missing_rate: float) -> Optional[List[int]]:
    if rng.random() < missing_rate:
        return None  # 缺读：该次观测整体丢失
    return [b if rng.random() >= flip else 1 - b for b in (int(c) for c in truth_pair)]


def make_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    n = config["n"]
    rng = random.Random(config["seed"])
    fabrics: List[Dict[str, Any]] = []
    for idx in range(config["n_train"] + config["n_holdout"]):
        truth = generate_truth(rng, n, config["p0"], config["stay"])
        readings: List[Optional[List[int]]] = []
        for i in range(n - 1):
            readings.append(generate_reading(
                rng, truth[i:i + 2], config["flip"], config["missing_rate"]))
        split = "train" if idx < config["n_train"] else "holdout"
        fabrics.append({
            "fabric_id": f"{config['name']}-{idx:03d}",
            "split": split,
            "n": n,
            "truth": truth,
            "readings": readings,
        })
    return {"config": config, "fabrics": fabrics}


def split_fabrics(dataset: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "train": [f for f in dataset["fabrics"] if f["split"] == "train"],
        "holdout": [f for f in dataset["fabrics"] if f["split"] == "holdout"],
    }


MATERIALS: Dict[str, Dict[str, Any]] = {
    # 材料 A：相邻格相关性较强，坏布约三成
    "material_a": {
        "name": "material_a", "n": 6, "seed": 20261125,
        "p0": 0.3, "stay": 0.8, "flip": 0.1, "missing_rate": 0.08,
        "n_train": 40, "n_holdout": 8,
    },
    # 材料 B：换一种相关性与噪声，验题人可整份换入
    "material_b": {
        "name": "material_b", "n": 5, "seed": 777001,
        "p0": 0.4, "stay": 0.6, "flip": 0.15, "missing_rate": 0.12,
        "n_train": 32, "n_holdout": 6,
    },
}


def write_dataset(dataset: Dict[str, Any], path: str) -> None:
    dump_json(dataset, path)
