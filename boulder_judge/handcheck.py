"""Hand-enumerable probability examples requested by the brief."""
from __future__ import annotations

from typing import Any

from .inference import adjudicate
from .model import JointModel

E, H, U = "E", "H", "U"
STATES = [[E, E], [E, H], [H, E], [H, H]]
READINGS = [E, H, U]
PAIRS = [[a, b] for a in READINGS for b in READINGS]
IDX = {f"{a}|{b}": i for i, (a, b) in enumerate(STATES)}
OBS_IDX = {f"{a}|{b}": i for i, (a, b) in enumerate(PAIRS)}


def _uniform_transition() -> list[list[float]]:
    return [[0.25] * 4 for _ in range(4)]


def _uninformative_u_emission() -> list[list[float]]:
    # Every hidden state makes paired reading U,U certain.  This is a compact
    # mathematical probe of the joint distribution, not a physical sensor.
    row = [0.0] * len(PAIRS)
    row[OBS_IDX["U|U"]] = 1.0
    return [list(row) for _ in STATES]


def _identity_emission() -> list[list[float]]:
    emis = [[0.0] * len(PAIRS) for _ in STATES]
    for i, state in enumerate(STATES):
        emis[i][OBS_IDX[f"{state[0]}|{state[1]}"]] = 1.0
    return emis


def _route(stable: int, deadline: int | None = None) -> dict[str, Any]:
    return {
        "format": "boulder-route-card/v1",
        "route_id": "hand-route",
        "top_vertex": H,
        "zone_vertex": "NEVER_ZONE",
        "none_vertex": E,
        "stable_frames": stable,
        "deadline_frame": deadline,
        "zone_requires": "any_hand",
        "threshold": 0.95,
    }


def _marginal_hands(result: dict[str, Any]) -> list[dict[str, float]]:
    out = []
    for fr in result["frame_evidence"]:
        dist = fr["joint_state_posterior"]
        p_left_h = dist["H|E"] + dist["H|H"]
        p_right_h = dist["E|H"] + dist["H|H"]
        out.append({"frame": fr["frame"], "p_left_hold": p_left_h, "p_right_hold": p_right_h})
    return out


def run_hand_examples() -> dict[str, Any]:
    # Example 1a: one frame, HH and EE each have mass 1/2.
    model_joint = JointModel(
        vertices=[E, H],
        reading_values=READINGS,
        states=STATES,
        reading_pairs=PAIRS,
        initial=[0.5, 0.0, 0.0, 0.5],
        transition=_uniform_transition(),
        emission=_uninformative_u_emission(),
        model_version="hand-joint-half",
    )
    clip_joint = {"clip_id": "hand-joint-half", "frames": [{"left_reading": U, "right_reading": U}]}
    res_joint = adjudicate(model_joint, clip_joint, _route(stable=1, deadline=0))

    # Example 1b: four frames, always exactly one hand holds; the identity of
    # the holding hand alternates deterministically from either initial state.
    trans_exact_one = [[0.0] * 4 for _ in range(4)]
    trans_exact_one[IDX["E|H"]][IDX["H|E"]] = 1.0
    trans_exact_one[IDX["H|E"]][IDX["E|H"]] = 1.0
    # Unreachable rows still need normalized probabilities.
    trans_exact_one[IDX["E|E"]] = [0.25] * 4
    trans_exact_one[IDX["H|H"]] = [0.25] * 4
    model_exact_one = JointModel(
        vertices=[E, H],
        reading_values=READINGS,
        states=STATES,
        reading_pairs=PAIRS,
        initial=[0.0, 0.5, 0.5, 0.0],
        transition=trans_exact_one,
        emission=_uninformative_u_emission(),
        model_version="hand-exactly-one",
    )
    clip_exact_one = {
        "clip_id": "hand-exactly-one",
        "frames": [{"left_reading": U, "right_reading": U} for _ in range(4)],
    }
    res_exact_one = adjudicate(model_exact_one, clip_exact_one, _route(stable=1, deadline=3))

    # Example 2: EE then HH, with exact readings and deterministic transition.
    trans_last = [[0.0] * 4 for _ in range(4)]
    trans_last[IDX["E|E"]][IDX["H|H"]] = 1.0
    trans_last[IDX["H|H"]][IDX["H|H"]] = 1.0
    trans_last[IDX["E|H"]] = [0.0, 1.0, 0.0, 0.0]
    trans_last[IDX["H|E"]] = [0.0, 0.0, 1.0, 0.0]
    model_last = JointModel(
        vertices=[E, H],
        reading_values=READINGS,
        states=STATES,
        reading_pairs=PAIRS,
        initial=[1.0, 0.0, 0.0, 0.0],
        transition=trans_last,
        emission=_identity_emission(),
        model_version="hand-last-frame-hh",
    )
    clip_last = {
        "clip_id": "hand-last-frame-hh",
        "frames": [
            {"left_reading": E, "right_reading": E},
            {"left_reading": H, "right_reading": H},
        ],
    }
    res_last_k1 = adjudicate(model_last, clip_last, _route(stable=1, deadline=1))
    res_last_k2 = adjudicate(model_last, clip_last, _route(stable=2, deadline=1))

    examples = [
        {
            "name": "half_HH_half_EE_one_frame_K1",
            "hand_path_calculation": "P(EE)=0.5，P(HH)=0.5；只有 HH 满足完攀，因此 P(TOP)=0.5。",
            "hand_marginals": _marginal_hands(res_joint),
            "score_probabilities": res_joint["score_probabilities"],
            "p_top": res_joint["score_probabilities"][0]["probability"],
            "expected_p_top": 0.5,
        },
        {
            "name": "always_exactly_one_hand_K1",
            "hand_path_calculation": (
                "P(EH,HE,EH,HE)=0.5，P(HE,EH,HE,EH)=0.5。"
                "两条路径都没有 HH，所以即使每只手腕边缘概率均为 0.5，P(TOP)=0。"
            ),
            "hand_marginals": _marginal_hands(res_exact_one),
            "score_probabilities": res_exact_one["score_probabilities"],
            "p_top": res_exact_one["score_probabilities"][0]["probability"],
            "expected_p_top": 0.0,
        },
        {
            "name": "empty_then_final_HH_K1",
            "hand_path_calculation": "唯一路径为 EE -> HH；K=1 时末帧满足，因此 P(TOP)=1。",
            "score_probabilities": res_last_k1["score_probabilities"],
            "p_top": res_last_k1["score_probabilities"][0]["probability"],
            "expected_p_top": 1.0,
        },
        {
            "name": "empty_then_final_HH_K2",
            "hand_path_calculation": (
                "唯一路径为 EE -> HH。它没有连续两帧 HH，因此把 stable_frames 改为 2 后 P(TOP)=0。"
            ),
            "score_probabilities": res_last_k2["score_probabilities"],
            "p_top": res_last_k2["score_probabilities"][0]["probability"],
            "expected_p_top": 0.0,
        },
    ]
    all_ok = all(abs(ex["p_top"] - ex["expected_p_top"]) < 1e-12 for ex in examples)
    return {"format": "boulder-hand-checks/v1", "all_expected_values_reproduced": all_ok, "examples": examples}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 手算小例核对",
        "",
        f"全部期望值由程序复现：**{'是' if report['all_expected_values_reproduced'] else '否'}**",
        "",
        "| 小例 | 手算路径 | 程序 TOP 概率 | 期望 |",
        "|---|---|---:|---:|",
    ]
    for ex in report["examples"]:
        calc = ex["hand_path_calculation"].replace("|", "\\|")
        lines.append(f"| `{ex['name']}` | {calc} | {ex['p_top']:.6g} | {ex['expected_p_top']:g} |")
    lines.extend(["", "## 手腕边缘概率（前两例）", ""])
    for ex in report["examples"][:2]:
        lines.append(f"### `{ex['name']}`")
        for m in ex.get("hand_marginals", []):
            lines.append(
                f"- frame {m['frame']}: P(左手触顶/握)={m['p_left_hold']:.3f}, "
                f"P(右手触顶/握)={m['p_right_hold']:.3f}"
            )
    lines.append("")
    return "\n".join(lines)
