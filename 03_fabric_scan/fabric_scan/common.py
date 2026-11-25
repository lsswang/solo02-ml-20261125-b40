"""通用数据结构与后验推断。

整幅疵点图用 0/1 位串表示：0 = 好布，1 = 坏布。
例如 6 格布的图 "010010" 表示第 2、5 格坏。
低清读数是相邻两格的观测对，每个样布有 n-1 次观测；
某次观测为 null 即缺读，推断时只略去该次观测，不做任何插补。
"""

from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Bit = int
Map = Tuple[Bit, ...]


def all_maps(n: int) -> List[Map]:
    """n 格布的全部 2^n 幅候选图（n<=6，枚举可行）。"""
    return [tuple((i >> (n - 1 - j)) & 1 for j in range(n)) for i in range(1 << n)]


def map_to_str(m: Sequence[Bit]) -> str:
    return "".join(str(b) for b in m)


def str_to_map(s: str) -> Map:
    return tuple(int(c) for c in s)


def normalize(weights: Dict[str, float]) -> Dict[str, float]:
    """把未归一化权重归一化为概率分布；全零时回退均匀分布。"""
    w = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(w.values())
    if total <= 0.0:
        k = max(len(w), 1)
        return {key: 1.0 / k for key in w} if w else {}
    return {key: v / total for key, v in w.items()}


def emission_prob(emission: Dict[str, Dict[str, float]],
                  true_pair: Sequence[Bit], read_pair: Sequence[Bit]) -> float:
    """相邻两格真值给定时，读到某对低清读数的条件概率 P(read | true)。"""
    return emission[map_to_str(true_pair)].get(map_to_str(read_pair), 0.0)


def observation_likelihood(m: Map,
                           emission: Optional[Dict[str, Dict[str, float]]],
                           readings: Sequence[Optional[Sequence[Optional[int]]]]) -> float:
    """一幅候选图对整条低清读数串的似然（缺读项直接跳过）。"""
    p = 1.0
    if emission is None or not readings:
        return p
    for i, read in enumerate(readings):
        if read is None:  # 缺读：略去该次观测
            continue
        true_pair = (m[i], m[i + 1])
        read_pair = tuple(read)  # type: ignore[arg-type]
        p *= emission_prob(emission, true_pair, read_pair)  # type: ignore[arg-type]
        if p == 0.0:
            return 0.0
    return p


def posterior(prior: Dict[str, float],
              emission: Optional[Dict[str, Dict[str, float]]] = None,
              readings: Optional[Sequence[Optional[Sequence[Optional[int]]]]] = None,
              precision: Optional[Dict[int, Bit]] = None) -> Dict[str, float]:
    """整幅图后验 P(map | 低清读数, 精检结果)。

    先验与误读分布都显式传入，因此同一套推断代码既可以跑学来的模型，
    也可以跑题目给定的两格玩具模型。精检结果是无噪真值，按位条件化。
    """
    readings = readings or []
    precision = precision or {}
    weights: Dict[str, float] = {}
    for key, pr in prior.items():
        m = str_to_map(key)
        if any(m[c] != b for c, b in precision.items()):
            continue  # 与精检真值冲突的候选图直接剔除
        weights[key] = pr * observation_likelihood(m, emission, readings)
    return normalize(weights)


def marginals(post: Dict[str, float]) -> List[float]:
    """后验逐格坏布概率。"""
    if not post:
        return []
    n = len(next(iter(post)))
    out = [0.0] * n
    for key, p in post.items():
        for j, c in enumerate(key):
            if c == "1":
                out[j] += p
    return out


def map_map(post: Dict[str, float]) -> Tuple[str, float]:
    """最大后验图及其概率（并列时按位串字典序，保证可复跑）。"""
    key = max(post, key=lambda k: (post[k], tuple(-ord(c) for c in k)))
    return key, post[key]


def entropy(post: Dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in post.values() if p > 0.0)


def condition(post: Dict[str, float], cell: int, bit: Bit) -> Dict[str, float]:
    """已知某格精检真值后做条件化（便携头第二次决策时使用）。"""
    return posterior(post, precision={cell: bit})


def load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: object, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def iter_cells(n: int) -> Iterable[int]:
    return range(n)
