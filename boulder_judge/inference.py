"""Exact posterior and trajectory-score calculations."""
from __future__ import annotations

import math
from typing import Any

from .mathutils import NEG_INF, exp_normalize, logsumexp
from .model import JointModel, pair_key

CLASS_ORDER = ["TOP", "ZONE", "NO_ZONE"]
CLASS_LABELS = {"TOP": "完攀", "ZONE": "区点", "NO_ZONE": "无区点"}


def clip_observations(model: JointModel, clip: dict[str, Any]) -> list[list[float]]:
    vectors = []
    for fr in clip["frames"]:
        vectors.append(model.log_observation_vector(fr["left_reading"], fr["right_reading"]))
    return vectors


def posterior_marginals(
    model: JointModel, clip: dict[str, Any]
) -> tuple[list[list[float]], float]:
    """Forward-backward posterior P(state_t | all paired readings)."""
    obs = clip_observations(model, clip)
    n = len(obs)
    s_count = len(model.states)
    if n == 0:
        return [], 0.0

    log_alpha: list[list[float]] = [[NEG_INF] * s_count for _ in range(n)]
    for s in range(s_count):
        log_alpha[0][s] = model.log_initial[s] + obs[0][s]
    for t in range(1, n):
        for s in range(s_count):
            log_alpha[t][s] = obs[t][s] + logsumexp(
                log_alpha[t - 1][p] + model.log_transition[p][s] for p in range(s_count)
            )

    log_beta: list[list[float]] = [[NEG_INF] * s_count for _ in range(n)]
    log_beta[n - 1] = [0.0] * s_count
    for t in range(n - 2, -1, -1):
        for p in range(s_count):
            log_beta[t][p] = logsumexp(
                model.log_transition[p][s] + obs[t + 1][s] + log_beta[t + 1][s]
                for s in range(s_count)
            )

    log_likelihood = logsumexp(log_alpha[n - 1])
    gamma: list[list[float]] = []
    for t in range(n):
        logs = [log_alpha[t][s] + log_beta[t][s] - log_likelihood for s in range(s_count)]
        gamma.append([math.exp(x) if x > -745.0 else 0.0 for x in logs])
        # Protect tiny rounding drift by renormalizing directly.
        total = sum(gamma[t])
        if total > 0:
            gamma[t] = [x / total for x in gamma[t]]
    return gamma, log_likelihood


def viterbi_path(model: JointModel, clip: dict[str, Any]) -> tuple[list[list[str]], float]:
    """Most probable complete joint contact trajectory."""
    obs = clip_observations(model, clip)
    n = len(obs)
    s_count = len(model.states)
    if n == 0:
        return [], NEG_INF
    dp = [model.log_initial[s] + obs[0][s] for s in range(s_count)]
    backpointers: list[list[int]] = []
    for t in range(1, n):
        new_dp = [NEG_INF] * s_count
        backs = [0] * s_count
        for s in range(s_count):
            candidates = [
                (dp[p] + model.log_transition[p][s] + obs[t][s], p) for p in range(s_count)
            ]
            best_val, best_p = max(candidates, key=lambda x: x[0])
            new_dp[s] = best_val
            backs[s] = best_p
        dp = new_dp
        backpointers.append(backs)
    last = max(range(s_count), key=lambda s: dp[s])
    score = dp[last]
    idx_path = [last]
    for t in range(n - 1, 0, -1):
        last = backpointers[t - 1][last]
        idx_path.append(last)
    idx_path.reverse()
    return [model.states[i] for i in idx_path], score


def _is_zone_state(state: list[str], zone_vertex: str, requirement: str) -> bool:
    if requirement == "both_hands":
        return state[0] == zone_vertex and state[1] == zone_vertex
    return state[0] == zone_vertex or state[1] == zone_vertex


def aggregate_route_probabilities_from_log_obs(
    states: list[list[str]],
    log_initial: list[float],
    log_transition: list[list[float]],
    obs: list[list[float]],
    route: dict[str, Any],
) -> dict[str, Any]:
    """Sum exact HMM path masses into TOP / ZONE / NO_ZONE score classes.

    The DP tracks the current consecutive ``(top, top)`` run and two route
    flags.  Readings after the inclusive deadline still inform retrospective
    state estimates, but cannot set a score flag.  Accepting log emission
    factors directly lets the sealed-evidence venue rerun the same arithmetic
    after raw readings have been destroyed.
    """
    n = len(obs)
    if n == 0:
        raise ValueError("clip has no frames")

    top_vertex = route["top_vertex"]
    zone_vertex = route["zone_vertex"]
    stable = int(route["stable_frames"])
    deadline = route.get("deadline_frame", None)
    requirement = route.get("zone_requires", "any_hand")
    s_count = len(states)

    # key: (state, capped_top_run, valid_top, reached_zone) -> log mass
    dp: dict[tuple[int, int, bool, bool], float] = {}
    for s in range(s_count):
        state = states[s]
        is_top_pair = state[0] == top_vertex and state[1] == top_vertex
        run = min(stable, 1) if is_top_pair else 0
        active = deadline is None or 0 <= int(deadline)
        topped = bool(is_top_pair and run >= stable and active)
        zoned = bool(_is_zone_state(state, zone_vertex, requirement) and active)
        key = (s, run, topped, zoned)
        val = log_initial[s] + obs[0][s]
        dp[key] = logsumexp([dp.get(key, NEG_INF), val])

    for t in range(1, n):
        active = deadline is None or t <= int(deadline)
        nxt: dict[tuple[int, int, bool, bool], float] = {}
        for (prev_s, run, topped, zoned), weight in dp.items():
            for s in range(s_count):
                state = states[s]
                is_top_pair = state[0] == top_vertex and state[1] == top_vertex
                new_run = min(stable, run + 1) if is_top_pair else 0
                new_topped = topped or bool(active and new_run >= stable)
                new_zoned = zoned or bool(
                    active and _is_zone_state(state, zone_vertex, requirement)
                )
                key = (s, new_run, new_topped, new_zoned)
                val = weight + log_transition[prev_s][s] + obs[t][s]
                nxt[key] = logsumexp([nxt.get(key, NEG_INF), val])
        dp = nxt

    class_logs = {code: NEG_INF for code in CLASS_ORDER}
    for (_, _, topped, zoned), weight in dp.items():
        code = "TOP" if topped else ("ZONE" if zoned else "NO_ZONE")
        class_logs[code] = logsumexp([class_logs[code], weight])

    logs = [class_logs[c] for c in CLASS_ORDER]
    probs = exp_normalize(logs)
    log_likelihood = logsumexp(logs)
    return {
        "probabilities": {code: probs[i] for i, code in enumerate(CLASS_ORDER)},
        "log_joint_weights": {code: logs[i] for i, code in enumerate(CLASS_ORDER)},
        "log_likelihood": log_likelihood,
        "deadline_frame": deadline,
        "frames_processed": n,
    }


def aggregate_route_probabilities(
    model: JointModel, clip: dict[str, Any], route: dict[str, Any]
) -> dict[str, Any]:
    """Observation-backed wrapper around the factor-based route DP."""
    obs = clip_observations(model, clip)
    return aggregate_route_probabilities_from_log_obs(
        model.states, model.log_initial, model.log_transition, obs, route
    )


def simultaneous_top_event_probability(
    model: JointModel,
    clip: dict[str, Any],
    top_vertex: str,
    deadline_only: bool = False,
    deadline_frame: int | None = None,
) -> float:
    """Posterior probability of at least one exact ``(top, top)`` frame."""
    obs = clip_observations(model, clip)
    n = len(obs)
    if n == 0:
        return 0.0
    s_count = len(model.states)
    # dp[(state, seen)] = log joint mass
    dp: dict[tuple[int, bool], float] = {}
    for s in range(s_count):
        state = model.states[s]
        tt = state[0] == top_vertex and state[1] == top_vertex
        active = (not deadline_only) or deadline_frame is None or 0 <= int(deadline_frame)
        seen = bool(tt and active)
        dp[(s, seen)] = model.log_initial[s] + obs[0][s]
    for t in range(1, n):
        active = (not deadline_only) or deadline_frame is None or t <= int(deadline_frame)
        nxt: dict[tuple[int, bool], float] = {}
        for (prev_s, seen), weight in dp.items():
            for s in range(s_count):
                state = model.states[s]
                tt = state[0] == top_vertex and state[1] == top_vertex
                new_seen = seen or bool(tt and active)
                key = (s, new_seen)
                val = weight + model.log_transition[prev_s][s] + obs[t][s]
                nxt[key] = logsumexp([nxt.get(key, NEG_INF), val])
        dp = nxt
    seen_log = logsumexp([w for (_, seen), w in dp.items() if seen])
    total_log = logsumexp(list(dp.values()))
    if total_log == NEG_INF:
        return 0.0
    return float(math.exp(seen_log - total_log)) if seen_log != NEG_INF else 0.0


def evidence_band(p: float, threshold: float) -> str:
    if p + 1e-12 >= threshold:
        return "signing_strength"
    if p >= 0.75:
        return "strong_below_signing"
    if p >= 0.25:
        return "ambiguous"
    if p >= 0.05:
        return "weak"
    return "effectively_absent"


def adjudicate(
    model: JointModel, clip: dict[str, Any], route: dict[str, Any]
) -> dict[str, Any]:
    """Full deliverable: score probabilities, signature, and frame evidence."""
    route_threshold = float(route.get("threshold", 0.95))
    agg = aggregate_route_probabilities(model, clip, route)
    probs = agg["probabilities"]
    gamma, fb_loglik = posterior_marginals(model, clip)
    viterbi, viterbi_log = viterbi_path(model, clip)

    top_vertex = route["top_vertex"]
    tt_index = model.state_index.get(pair_key((top_vertex, top_vertex)))
    frame_evidence = []
    for t, row in enumerate(gamma):
        dominant = max(range(len(row)), key=lambda s: row[s])
        p_tt = float(row[tt_index]) if tt_index is not None else 0.0
        frame_evidence.append(
            {
                "frame": t,
                "wrists": ["left", "right"],
                "p_both_at_top_vertex": p_tt,
                "evidence_band": evidence_band(p_tt, route_threshold),
                "posterior_joint_state": model.states[dominant],
                "posterior_dominant_probability": float(row[dominant]),
                "joint_state_posterior": {
                    pair_key(model.states[s]): float(row[s]) for s in range(len(model.states))
                },
                "readings": {
                    "left": clip["frames"][t]["left_reading"],
                    "right": clip["frames"][t]["right_reading"],
                },
            }
        )

    signed_class = max(CLASS_ORDER, key=lambda c: probs[c])
    signed = probs[signed_class] + 1e-12 >= route_threshold
    if not signed:
        signed_class = None

    high_conf = [x["frame"] for x in frame_evidence if x["p_both_at_top_vertex"] + 1e-12 >= route_threshold]
    probable = [x["frame"] for x in frame_evidence if x["p_both_at_top_vertex"] >= 0.5]

    result = {
        "format": "boulder-decision/v1",
        "clip_id": clip.get("clip_id", "unknown-clip"),
        "route_id": route.get("route_id", "unknown-route"),
        "route_summary": {
            "top_vertex": route["top_vertex"],
            "zone_vertex": route["zone_vertex"],
            "stable_frames": int(route["stable_frames"]),
            "deadline_frame": route.get("deadline_frame"),
            "zone_requires": route.get("zone_requires", "any_hand"),
        },
        "model_version": model.model_version,
        "model_fingerprint": model.fingerprint(),
        "threshold": route_threshold,
        "score_probabilities": [
            {
                "code": code,
                "label_zh": CLASS_LABELS[code],
                "probability": float(probs[code]),
            }
            for code in CLASS_ORDER
        ],
        "signed": bool(signed),
        "signed_class": signed_class,
        "status": "signed" if signed else "appealable_no_class_reached_threshold",
        "leading_class": max(CLASS_ORDER, key=lambda c: probs[c]),
        "frame_evidence": frame_evidence,
        "high_confidence_simultaneous_top_frames": high_conf,
        "probable_simultaneous_top_frames": probable,
        "p_any_simultaneous_top_in_clip": simultaneous_top_event_probability(
            model, clip, top_vertex, False
        ),
        "p_any_simultaneous_top_before_deadline": simultaneous_top_event_probability(
            model, clip, top_vertex, True, route.get("deadline_frame")
        ),
        "most_likely_joint_path": [
            {"frame": t, "left_contact": s[0], "right_contact": s[1]}
            for t, s in enumerate(viterbi)
        ],
        "log_likelihood": agg["log_likelihood"],
        "forward_backward_log_likelihood": fb_loglik,
        "viterbi_log_score": viterbi_log,
    }
    return result
