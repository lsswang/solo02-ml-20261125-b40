"""Small deterministic synthetic bouldering corpus.

The generator is deliberately transparent: hidden joint contact trajectories
are made first, then *paired* wrist readings are emitted with both ordinary
confusions and correlated simultaneous occlusion.  All default clips have at
most 16 frames.
"""
from __future__ import annotations

import random
from typing import Any

VERTICES = ["NONE", "ZONE", "TOP"]
READING_VALUES = ["NONE", "ZONE", "TOP", "OCCLUDED"]

N, Z, T = VERTICES
O = "OCCLUDED"  # noqa: E741


def _blank(length: int) -> list[list[str]]:
    return [[N, N] for _ in range(length)]


def make_hidden_sequence(rng: random.Random, outcome: str, length: int) -> list[list[str]]:
    """Create a short physically plausible joint contact path."""
    if length > 16:
        raise ValueError("reference demonstrations require <= 16 frames")
    seq = _blank(length)
    z_start = rng.randint(2, min(4, length - 2))

    if outcome in {"zone", "top"}:
        current = rng.choice([[Z, Z], [Z, N], [N, Z]])
        for t in range(z_start, length):
            if rng.random() < 0.22:
                current = rng.choice([[Z, Z], [Z, N], [N, Z]])
            seq[t] = list(current)

    if outcome == "top":
        latest_start = max(z_start + 2, min(8, length - 3))
        t_start = rng.randint(max(z_start + 2, 5), latest_start)
        first_top = rng.choice([[T, Z], [Z, T], [T, N], [N, T]])
        if t_start < length:
            seq[t_start] = first_top
        tt_frames = rng.randint(2, 4)
        for k in range(1, tt_frames + 1):
            t = t_start + k
            if t < length:
                seq[t] = [T, T]
        # After matching the top, climbers may stay matched or peel one hand.
        for t in range(t_start + tt_frames + 1, length):
            seq[t] = rng.choice([[T, T], [T, Z], [Z, T], [T, N], [N, T]])

    # A few small transition wobbles make counts less degenerate without
    # destroying the intended outcome.
    for t in range(1, length - 1):
        if outcome == "none" and rng.random() < 0.10:
            seq[t] = rng.choice([[N, Z], [Z, N]])
        elif rng.random() < 0.06 and seq[t] != [T, T]:
            seq[t] = rng.choice([[N, N], [Z, N], [N, Z], [Z, Z]])
    return seq


def emit_reading_pair(
    rng: random.Random, state: list[str], correlated_occlusion: float | None = None
) -> list[str]:
    top_pair = state == [T, T]
    corr_p = correlated_occlusion
    if corr_p is None:
        corr_p = 0.20 if top_pair else 0.10
    if rng.random() < corr_p:
        return [O, O]

    out = []
    for true_vertex in state:
        u = rng.random()
        if u < 0.80:
            out.append(true_vertex)
        elif u < 0.89:
            out.append(O)
        else:
            wrong = [v for v in (N, Z, T) if v != true_vertex]
            out.append(rng.choice(wrong))
    return out


def make_clip(
    rng: random.Random,
    clip_id: str,
    outcome: str,
    length: int | None = None,
    split: str = "train",
    correlated_occlusion: float | None = None,
) -> dict[str, Any]:
    length = length or rng.randint(10, 16)
    states = make_hidden_sequence(rng, outcome, length)
    frames = []
    for t, state in enumerate(states):
        lr, rr = emit_reading_pair(rng, state, correlated_occlusion)
        frames.append(
            {
                "frame": t,
                "left_contact": state[0],
                "right_contact": state[1],
                "left_reading": lr,
                "right_reading": rr,
            }
        )
    return {
        "format": "boulder-clip/v1",
        "clip_id": clip_id,
        "split": split,
        "known_truth_outcome": outcome,
        "frames": frames,
    }


def generate_corpus(
    seed: int = 20261125,
    n_train: int = 24,
    n_heldout: int = 8,
    train_prefix: str = "train",
    heldout_prefix: str = "heldout",
) -> dict[str, Any]:
    rng = random.Random(seed)
    clips: list[dict[str, Any]] = []
    outcome_cycle = ["top", "zone", "none", "top", "zone", "none", "top", "zone"]
    for i in range(n_train):
        outcome = outcome_cycle[i % len(outcome_cycle)]
        clips.append(make_clip(rng, f"{train_prefix}-{i:03d}", outcome, split="train"))
    for i in range(n_heldout):
        outcome = outcome_cycle[(i + 1) % len(outcome_cycle)]
        clips.append(make_clip(rng, f"{heldout_prefix}-{i:03d}", outcome, split="heldout"))
    return {
        "format": "boulder-synthetic-corpus/v1",
        "seed": seed,
        "vertices": VERTICES,
        "reading_values": READING_VALUES,
        "generator": "phase-and-paired-emission/v1",
        "clips": clips,
    }


def generate_calibration_extra(seed: int = 9917, n: int = 14) -> dict[str, Any]:
    """A later routine labelled calibration batch."""
    corpus = generate_corpus(
        seed=seed,
        n_train=n,
        n_heldout=0,
        train_prefix="recal",
    )
    corpus["format"] = "boulder-calibration-batch/v1"
    corpus["note"] = "Post-event labelled calibration clips."
    return corpus


def generate_false_top_revision(n_clips: int = 18, length: int = 8) -> dict[str, Any]:
    """Calibration study exposing a correlated false TOP/TOP reading mode.

    Manual labels stay at ZONE/ZONE while the paired tracker reading is
    TOP/TOP.  This is exactly the kind of paired error that a joint emission
    table can learn and that an independence assumption would mishandle.
    """
    clips = []
    for i in range(n_clips):
        frames = []
        for t in range(length):
            frames.append(
                {
                    "frame": t,
                    "left_contact": Z,
                    "right_contact": Z,
                    "left_reading": T,
                    "right_reading": T,
                }
            )
        clips.append(
            {
                "format": "boulder-clip/v1",
                "clip_id": f"paired-false-top-{i:02d}",
                "split": "train",
                "calibration_note": "Both trackers snap to TOP together while labels remain ZONE/ZONE.",
                "frames": frames,
            }
        )
    return {
        "format": "boulder-calibration-batch/v1",
        "vertices": VERTICES,
        "reading_values": READING_VALUES,
        "note": "Post-appeal paired-error calibration: correlated false TOP/TOP readings from ZONE/ZONE.",
        "clips": clips,
    }


def default_route_card() -> dict[str, Any]:
    return {
        "format": "boulder-route-card/v1",
        "route_id": "W5-qualifier",
        "top_vertex": T,
        "zone_vertex": Z,
        "none_vertex": N,
        "stable_frames": 2,
        "deadline_frame": 11,
        "frame_index_base": 0,
        "zone_requires": "any_hand",
        "threshold": 0.95,
    }


def make_appeal_attempt() -> tuple[dict[str, Any], dict[str, Any]]:
    """Engineered attempt that changes outcome under revised paired calibration.

    Under the initial calibration, four consecutive TOP/TOP paired readings
    look strong enough to sign a TOP.  The later calibration set reveals that
    this particular tracker version can emit correlated TOP/TOP pairs while
    both labelled contacts remain ZONE/ZONE.
    """
    rows = [
        # reading L, reading R, manual truth L, manual truth R
        (N, N, N, N),
        (N, N, N, N),
        (N, Z, N, Z),
        (Z, Z, Z, Z),
        (Z, Z, Z, Z),
        (Z, Z, Z, Z),
        (T, T, Z, Z),
        (T, T, Z, Z),
        (T, T, Z, Z),
        (T, T, Z, Z),
    ]
    frames = []
    truth_frames = []
    for t, (lr, rr, lc, rc) in enumerate(rows):
        frames.append({"frame": t, "left_reading": lr, "right_reading": rr})
        truth_frames.append(
            {
                "frame": t,
                "left_contact": lc,
                "right_contact": rc,
                "left_reading": lr,
                "right_reading": rr,
            }
        )
    clip = {
        "format": "boulder-clip/v1",
        "clip_id": "attempt-A17",
        "athlete": "demo-finalist-17",
        "frames": frames,
    }
    truth = {
        "format": "boulder-clip-truth/v1",
        "clip_id": "attempt-A17",
        "known_truth_outcome": "zone",
        "note": "Oracle labels are for verification only; the judge receives readings only.",
        "frames": truth_frames,
    }
    return clip, truth


def make_sealed_testimonies() -> tuple[dict[str, Any], dict[str, Any]]:
    """Two arbitrary frame-by-frame testimonies for the alternative venue."""
    common = [[N, N], [N, N], [N, Z], [Z, Z], [Z, Z], [Z, Z]]
    top_states = common + [[T, T]] * 4
    zone_states = common + [[Z, Z]] * 4

    def make(tid: str, source: str, outcome: str, states: list[list[str]]) -> dict[str, Any]:
        return {
            "format": "boulder-contact-testimony/v1",
            "testimony_id": tid,
            "source": source,
            "claimed_outcome": outcome,
            "frames": [
                {"frame": t, "left_contact": s[0], "right_contact": s[1]}
                for t, s in enumerate(states)
            ],
        }

    return (
        make("T-A-top-witness", "witness_A", "TOP", top_states),
        make("T-B-zone-coach", "witness_B", "ZONE", zone_states),
    )
