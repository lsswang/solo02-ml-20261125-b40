"""Optional runnable preview of the other venue's sealed-evidence route.

The main project chooses the retained-readings archive in ``archive.py``.
This module exists to make the difference concrete: after destroying the raw
paired readings, export one joint emission likelihood factor per hidden joint
state per frame.  Those factors are sufficient to rerun the route DP and to
score arbitrary complete frame-by-frame contact testimonies under the sealed
model.
"""
from __future__ import annotations

import math
from typing import Any

from .inference import aggregate_route_probabilities_from_log_obs
from .model import JointModel, pair_key

SEALED_FORMAT = "boulder-sealed-joint-evidence/v1"


def build_sealed_bundle(
    model: JointModel, clip: dict[str, Any], route: dict[str, Any]
) -> dict[str, Any]:
    factors = []
    for t, fr in enumerate(clip["frames"]):
        obs_j = model.observation_index(fr["left_reading"], fr["right_reading"])
        factors.append(
            {
                "frame": t,
                # Joint, not wrist-independent: one likelihood per (L,R) state.
                "joint_emission_likelihood": {
                    pair_key(model.states[s]): float(model.emission[s][obs_j])
                    for s in range(len(model.states))
                },
            }
        )
    return {
        "format": SEALED_FORMAT,
        "appeal_number": None,
        "clip_id": clip.get("clip_id"),
        "raw_wrist_readings_included": False,
        "retention_rule": "paired readings must be destroyed same day; only joint factors remain",
        "route_card": route,
        "state_order": model.states,
        "initial": model.initial,
        "transition": model.transition,
        "sealed_model_version": model.model_version,
        "sealed_model_fingerprint": model.fingerprint(),
        "frame_factors": factors,
    }


def bundle_log_observations(bundle: dict[str, Any]) -> list[list[float]]:
    states = bundle["state_order"]
    logs = []
    for factor in bundle["frame_factors"]:
        row = []
        table = factor["joint_emission_likelihood"]
        for state in states:
            p = float(table[pair_key(state)])
            row.append(math.log(p) if p > 0.0 else float("-inf"))
        logs.append(row)
    return logs


def rejudge_sealed(bundle: dict[str, Any]) -> dict[str, Any]:
    return aggregate_route_probabilities_from_log_obs(
        bundle["state_order"],
        [math.log(x) if x > 0 else float("-inf") for x in bundle["initial"]],
        [
            [math.log(x) if x > 0 else float("-inf") for x in row]
            for row in bundle["transition"]
        ],
        bundle_log_observations(bundle),
        bundle["route_card"],
    )


def _testimony_path(testimony: dict[str, Any], bundle: dict[str, Any]) -> list[int]:
    index = {pair_key(s): i for i, s in enumerate(bundle["state_order"])}
    frames = testimony.get("frames", [])
    if len(frames) != len(bundle["frame_factors"]):
        raise ValueError(
            f"testimony has {len(frames)} frames, evidence has {len(bundle['frame_factors'])}"
        )
    path = []
    for t, fr in enumerate(frames):
        key = pair_key((fr["left_contact"], fr["right_contact"]))
        if key not in index:
            raise KeyError(f"frame {t}: unknown joint contact state {key!r}")
        path.append(index[key])
    return path


def _class_for_path(path_states: list[list[str]], route: dict[str, Any]) -> str:
    # Local deterministic rule, kept compatible with evaluate.labelled_route_class.
    top, zone, k = route["top_vertex"], route["zone_vertex"], int(route["stable_frames"])
    deadline = route.get("deadline_frame")
    run = 0
    topped = zoned = False
    for t, state in enumerate(path_states):
        active = deadline is None or t <= int(deadline)
        if active and state[0] == top and state[1] == top:
            run += 1
            topped = topped or run >= k
        else:
            run = 0
        if active and (state[0] == zone or state[1] == zone):
            zoned = True
    return "TOP" if topped else ("ZONE" if zoned else "NO_ZONE")


def score_testimony(
    bundle: dict[str, Any], testimony: dict[str, Any]
) -> dict[str, Any]:
    path_idx = _testimony_path(testimony, bundle)
    log_obs = bundle_log_observations(bundle)
    log_initial = [math.log(x) if x > 0 else float("-inf") for x in bundle["initial"]]
    log_trans = [
        [math.log(x) if x > 0 else float("-inf") for x in row]
        for row in bundle["transition"]
    ]
    log_weight = log_initial[path_idx[0]] + log_obs[0][path_idx[0]]
    frame_terms = [
        {
            "frame": 0,
            "initial_log_probability": log_initial[path_idx[0]],
            "emission_log_likelihood": log_obs[0][path_idx[0]],
        }
    ]
    for t in range(1, len(path_idx)):
        prev, cur = path_idx[t - 1], path_idx[t]
        trans_term = log_trans[prev][cur]
        emit_term = log_obs[t][cur]
        log_weight += trans_term + emit_term
        frame_terms.append(
            {
                "frame": t,
                "transition_log_probability": trans_term,
                "emission_log_likelihood": emit_term,
            }
        )
    total_log = rejudge_sealed(bundle)["log_likelihood"]
    path_states = [bundle["state_order"][i] for i in path_idx]
    return {
        "testimony_id": testimony.get("testimony_id", testimony.get("clip_id", "testimony")),
        "source": testimony.get("source", "appellant"),
        "route_class_if_testimony_exact": _class_for_path(path_states, bundle["route_card"]),
        "state_path": path_states,
        "log_joint_weight": log_weight,
        "posterior_path_probability_under_sealed_model": math.exp(log_weight - total_log)
        if log_weight != float("-inf")
        else 0.0,
        "frame_terms": frame_terms,
    }


def compare_testimonies(
    bundle: dict[str, Any],
    testimony_a: dict[str, Any],
    testimony_b: dict[str, Any],
) -> dict[str, Any]:
    a = score_testimony(bundle, testimony_a)
    b = score_testimony(bundle, testimony_b)
    idx_a = _testimony_path(testimony_a, bundle)
    idx_b = _testimony_path(testimony_b, bundle)
    log_obs = bundle_log_observations(bundle)
    log_trans = [
        [math.log(x) if x > 0 else float("-inf") for x in row]
        for row in bundle["transition"]
    ]
    conflicts = []
    for t, (ia, ib) in enumerate(zip(idx_a, idx_b)):
        if ia == ib:
            continue
        emit_log_ratio = log_obs[t][ia] - log_obs[t][ib]
        if t == 0:
            trans_log_ratio = (
                math.log(float(bundle["initial"][ia]))
                - math.log(float(bundle["initial"][ib]))
                if bundle["initial"][ia] > 0 and bundle["initial"][ib] > 0
                else float("inf")
            )
            prev_note = "initial-state prior"
        else:
            pa, pb = idx_a[t - 1], idx_b[t - 1]
            trans_log_ratio = log_trans[pa][ia] - log_trans[pb][ib]
            prev_note = f"transition from {pair_key(bundle['state_order'][pa])} vs {pair_key(bundle['state_order'][pb])}"
        conflicts.append(
            {
                "frame": t,
                "testimony_a_state": bundle["state_order"][ia],
                "testimony_b_state": bundle["state_order"][ib],
                "emission_likelihood_ratio_a_over_b": _safe_exp(emit_log_ratio),
                "transition_ratio_a_over_b": _safe_exp(trans_log_ratio),
                "joint_frame_path_ratio_a_over_b": _safe_exp(emit_log_ratio + trans_log_ratio),
                "transition_context": prev_note,
                "factor_favors": "A" if emit_log_ratio > 0 else "B" if emit_log_ratio < 0 else "tie",
            }
        )
    log_bayes_factor = a["log_joint_weight"] - b["log_joint_weight"]
    return {
        "format": "boulder-sealed-testimony-comparison/v1",
        "sealed_model_version": bundle["sealed_model_version"],
        "sealed_model_fingerprint": bundle["sealed_model_fingerprint"],
        "rejudged_probabilities": rejudge_sealed(bundle)["probabilities"],
        "testimony_a": {k: v for k, v in a.items() if k != "frame_terms"},
        "testimony_b": {k: v for k, v in b.items() if k != "frame_terms"},
        "conflict_frames": conflicts,
        "log_bayes_factor_a_over_b": log_bayes_factor,
        "bayes_factor_a_over_b": _safe_exp(log_bayes_factor),
        "interpretation": (
            "A is better supported by sealed joint evidence"
            if log_bayes_factor > 0
            else "B is better supported by sealed joint evidence"
            if log_bayes_factor < 0
            else "equal support"
        ),
        "venue_warning": (
            "Raw readings are absent. A later emission calibration cannot be applied to these "
            "factors unless the new likelihoods were also sealed before readings destruction."
        ),
    }


def _safe_exp(x: float) -> float | None:
    try:
        if x > 709:
            return None  # report as overflow/>1e308 rather than crashing JSON
        return math.exp(x)
    except OverflowError:
        return None


def render_sealed_markdown(comparison: dict[str, Any]) -> str:
    a, b = comparison["testimony_a"], comparison["testimony_b"]
    probs = comparison["rejudged_probabilities"]
    lines = [
        "# 替代路线预览：销毁读数、封存联合因子",
        "",
        f"- 封存模型：`{comparison['sealed_model_version']}`",
        f"- 模型指纹：`{comparison['sealed_model_fingerprint'][:16]}…`",
        f"- 封存因子重判概率：TOP={probs['TOP']:.4f}, ZONE={probs['ZONE']:.4f}, NO_ZONE={probs['NO_ZONE']:.4f}",
        f"- 证词 A：`{a['testimony_id']}`，若逐帧为真则类别 `{a['route_class_if_testimony_exact']}`",
        f"- 证词 B：`{b['testimony_id']}`，若逐帧为真则类别 `{b['route_class_if_testimony_exact']}`",
        f"- log Bayes factor A/B：{comparison['log_bayes_factor_a_over_b']:.3f}",
        f"- 解释：**{comparison['interpretation']}**",
        "",
        "## 相抵触帧",
        "",
        "| 帧 | 证词 A | 证词 B | 发射似然比 A/B | 转移比 A/B | 联合帧路径比 A/B | 因子更支持 |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for c in comparison["conflict_frames"]:
        def fmt(x: float | None) -> str:
            return ">1e308" if x is None else f"{x:.3g}"

        lines.append(
            f"| {c['frame']} | {pair_key(c['testimony_a_state'])} | "
            f"{pair_key(c['testimony_b_state'])} | {fmt(c['emission_likelihood_ratio_a_over_b'])} | "
            f"{fmt(c['transition_ratio_a_over_b'])} | {fmt(c['joint_frame_path_ratio_a_over_b'])} | "
            f"{c['factor_favors']} |"
        )
    lines.extend(["", f"> {comparison['venue_warning']}", ""])
    return "\n".join(lines)
