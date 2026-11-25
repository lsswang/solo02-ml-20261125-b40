"""从成对的（低清读数, 精检真图）训练材料学习模型。

学两件事：
1. 整幅疵点图先验 P(map)：对全部 2^n 幅图做计数 + Dirichlet 加性平滑；
2. 相邻格给定时的误读分布 P(read | 两格真值)：4 类真值、每类 4 种读数，
   逐类计数 + 加性平滑。
缺读不参与误读计数。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import (all_maps, dump_json, emission_prob, load_json, map_to_str,
                     marginals, normalize, posterior, str_to_map)


def _pair_counts() -> Dict[str, Dict[str, int]]:
    pairs = ["00", "01", "10", "11"]
    return {a: {b: 0 for b in pairs} for a in pairs}


def learn_model(fabrics: Sequence[Dict[str, Any]],
                n: int,
                alpha: float = 1.0,
                beta: float = 0.5) -> Dict[str, Any]:
    """在训练布上计数并平滑，返回可直接给 common.posterior 使用的模型。"""
    map_counts = {map_to_str(m): 0 for m in all_maps(n)}
    emit_counts = _pair_counts()
    used_readings = 0
    skipped_missing = 0

    for fab in fabrics:
        truth = str_to_map(fab["truth"])
        if len(truth) != n:
            raise ValueError(f"布 {fab.get('fabric_id')} 格数 {len(truth)} 与 n={n} 不一致")
        map_counts[map_to_str(truth)] += 1
        for i, read in enumerate(fab["readings"]):
            if read is None:
                skipped_missing += 1
                continue
            true_pair = map_to_str((truth[i], truth[i + 1]))
            read_pair = "".join(str(b) for b in read)
            emit_counts[true_pair][read_pair] += 1
            used_readings += 1

    # Dirichlet(alpha) 平滑：未出现过的图也保留非零先验
    prior = normalize({k: v + alpha for k, v in map_counts.items()})
    emission: Dict[str, Dict[str, float]] = {}
    for true_pair, row in emit_counts.items():
        emission[true_pair] = normalize({k: v + beta for k, v in row.items()})

    return {
        "n": n,
        "alpha": alpha,
        "beta": beta,
        "n_train": len(fabrics),
        "n_readings_used": used_readings,
        "n_missing_skipped": skipped_missing,
        "prior": prior,
        "emission": emission,
        "map_counts": map_counts,
    }


def save_model(model: Dict[str, Any], path: str) -> None:
    dump_json(model, path)


def load_model(path: str) -> Dict[str, Any]:
    model = load_json(path)
    # 推断只需要先验与误读分布，附带字段保留用于审计
    return model  # type: ignore[return-value]


def posterior_of_fabric(model: Dict[str, Any],
                        readings: Sequence[Optional[Sequence[Optional[int]]]],
                        precision: Optional[Dict[int, int]] = None) -> Dict[str, float]:
    return posterior(model["prior"], model.get("emission"), readings, precision)


def evaluate(model: Dict[str, Any], holdout: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """在留出的整条布上评估：真值图概率、对数损失、MAP 命中、逐格 Brier。"""
    n = model["n"]
    rows: List[Dict[str, Any]] = []
    log_loss_sum = 0.0
    map_hits = 0
    brier_sum = 0.0
    for fab in holdout:
        truth = fab["truth"]
        post = posterior_of_fabric(model, fab["readings"])
        p_truth = post.get(truth, 0.0)
        log_loss = -math.log(max(p_truth, 1e-300))
        marg = marginals(post)
        brier = sum((marg[j] - int(truth[j])) ** 2 for j in range(n)) / n
        map_key = max(post, key=lambda k: post[k])
        hit = map_key == truth
        map_hits += int(hit)
        log_loss_sum += log_loss
        brier_sum += brier
        rows.append({
            "fabric_id": fab["fabric_id"],
            "truth": truth,
            "probability_of_truth": p_truth,
            "log_loss": log_loss,
            "map_estimate": map_key,
            "map_hit": hit,
            "posterior": {k: post[k] for k in sorted(post, key=lambda k: -post[k])},
        })
    k = max(len(holdout), 1)
    return {
        "n_holdout": len(holdout),
        "mean_log_loss": log_loss_sum / k,
        "map_accuracy": map_hits / k,
        "mean_brier": brier_sum / k,
        "fabrics": rows,
    }
