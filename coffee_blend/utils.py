from __future__ import annotations

import csv
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

Parts = Dict[str, int]
Composition = Mapping[str, int]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any, *, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, sort_keys=False)
        f.write("\n")


def append_jsonl(path: str, obj: Mapping[str, Any]) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_composition(text: str | Mapping[str, Any]) -> Parts:
    """Parse ``BATCH-A:2|BATCH-B:2`` or a mapping into positive integer parts."""
    if isinstance(text, Mapping):
        out: Parts = {}
        for k, v in text.items():
            if v is None or v == "":
                continue
            iv = int(v)
            if iv < 0:
                raise ValueError(f"negative dose for {k}: {iv}")
            if iv:
                out[str(k)] = iv
        return out

    out = {}
    s = str(text).strip()
    if not s:
        return out
    for chunk in re.split(r"[|;]", s):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"bad composition component {chunk!r}; expected BATCH:parts")
        batch, val = chunk.split(":", 1)
        batch = batch.strip()
        val = val.strip()
        if not batch:
            raise ValueError("empty batch id in composition")
        parts = int(float(val))
        if parts < 0:
            raise ValueError(f"negative dose for {batch}")
        if parts:
            out[batch] = parts
    if not out:
        raise ValueError("composition has no positive components")
    return out


def composition_to_string(comp: Composition) -> str:
    return "|".join(f"{b}:{int(comp[b])}" for b in sorted(comp))


def composition_key(comp: Composition) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((b, int(p)) for b, p in comp.items() if int(p) > 0))


def total_parts(comp: Composition) -> int:
    return sum(int(v) for v in comp.values())


def fractions(comp: Composition, parts_per_cup: int = 4) -> Dict[str, float]:
    total = total_parts(comp)
    if total <= 0:
        raise ValueError("empty composition")
    # Inputs are quarter recipes.  Normalization by actual total also makes
    # the estimator tolerant of a swapped file that uses fractions summing to 1.
    denom = float(total if total != parts_per_cup else parts_per_cup)
    return {b: int(p) / denom for b, p in comp.items()}


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 709.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -709.0))
    return z / (1.0 + z)


def safe_prob(p: float) -> float:
    return min(1.0 - 1e-15, max(1e-15, float(p)))


def log_loss(y: Sequence[float], p: Sequence[float]) -> float:
    if len(y) != len(p) or not y:
        raise ValueError("log_loss needs equal non-empty sequences")
    total = 0.0
    for yi, pi in zip(y, p):
        pi = safe_prob(pi)
        total -= yi * math.log(pi) + (1.0 - yi) * math.log(1.0 - pi)
    return total / len(y)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False))
