"""Joint two-wrist hidden-state model and supervised calibration.

Hidden state is the *joint* contact pair ``(left_vertex, right_vertex)``.
Observation is the *paired* wrist reading ``(left_reading, right_reading)``.
The emission table therefore retains correlated/paired reading errors such as
simultaneous body occlusion; the two wrists are never assumed independent.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .mathutils import NEG_INF

MODEL_FORMAT = "boulder-joint-hmm/v1"


def pairs(values: Sequence[str]) -> list[list[str]]:
    return [[a, b] for a in values for b in values]


def pair_key(pair: Sequence[str]) -> str:
    return f"{pair[0]}|{pair[1]}"


@dataclass
class JointModel:
    vertices: list[str]
    reading_values: list[str]
    states: list[list[str]]
    reading_pairs: list[list[str]]
    initial: list[float]
    transition: list[list[float]]
    emission: list[list[float]]
    model_version: str = "unversioned"
    alpha: float = 0.25
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.state_index = {pair_key(s): i for i, s in enumerate(self.states)}
        self.reading_index = {pair_key(o): i for i, o in enumerate(self.reading_pairs)}
        self.log_initial = [_log(x) for x in self.initial]
        self.log_transition = [[_log(x) for x in row] for row in self.transition]
        self.log_emission = [[_log(x) for x in row] for row in self.emission]

    # ----- serialization -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MODEL_FORMAT,
            "model_version": self.model_version,
            "smoothing_alpha": self.alpha,
            "vertices": self.vertices,
            "reading_values": self.reading_values,
            "state_order": self.states,
            "reading_pair_order": self.reading_pairs,
            "initial": self.initial,
            "transition": self.transition,
            "emission": self.emission,
            "metadata": self.metadata or {},
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JointModel":
        if d.get("format") != MODEL_FORMAT:
            raise ValueError(f"unsupported model format: {d.get('format')!r}")
        return cls(
            vertices=list(d["vertices"]),
            reading_values=list(d["reading_values"]),
            states=[list(x) for x in d["state_order"]],
            reading_pairs=[list(x) for x in d["reading_pair_order"]],
            initial=[float(x) for x in d["initial"]],
            transition=[[float(x) for x in row] for row in d["transition"]],
            emission=[[float(x) for x in row] for row in d["emission"]],
            model_version=str(d.get("model_version", "unversioned")),
            alpha=float(d.get("smoothing_alpha", 0.0)),
            metadata=dict(d.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "JointModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def fingerprint(self) -> str:
        """Stable hash of the probabilistic calibration, not free-text metadata."""
        core = {
            "format": MODEL_FORMAT,
            "vertices": self.vertices,
            "reading_values": self.reading_values,
            "state_order": self.states,
            "reading_pair_order": self.reading_pairs,
            "initial": self.initial,
            "transition": self.transition,
            "emission": self.emission,
            "alpha": self.alpha,
        }
        blob = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # ----- observations --------------------------------------------------
    def observation_index(self, left_reading: str, right_reading: str) -> int:
        key = pair_key((left_reading, right_reading))
        if key not in self.reading_index:
            known = ", ".join(pair_key(p) for p in self.reading_pairs)
            raise KeyError(
                f"unknown paired reading {key!r}. Retrain on a corpus containing it. "
                f"Known pairs: {known}"
            )
        return self.reading_index[key]

    def log_observation_vector(self, left_reading: str, right_reading: str) -> list[float]:
        j = self.observation_index(left_reading, right_reading)
        return [self.log_emission[s][j] for s in range(len(self.states))]


def _log(x: float) -> float:
    x = float(x)
    return NEG_INF if x <= 0.0 else math.log(x)


def normalize_counts(counts: list[float], alpha: float) -> list[float]:
    total = sum(counts) + alpha * len(counts)
    if total <= 0:
        return [1.0 / len(counts)] * len(counts)
    return [(c + alpha) / total for c in counts]


def train_model(
    clips: Iterable[dict[str, Any]],
    vertices: Sequence[str],
    reading_values: Sequence[str],
    alpha: float = 0.25,
    model_version: str = "trained-v1",
    metadata: dict[str, Any] | None = None,
) -> JointModel:
    """Supervised count training from clips carrying both labels and readings."""
    vertices = list(vertices)
    reading_values = list(reading_values)
    states = pairs(vertices)
    obs_pairs = pairs(reading_values)
    s_index = {pair_key(s): i for i, s in enumerate(states)}
    o_index = {pair_key(o): i for i, o in enumerate(obs_pairs)}
    k_s = len(states)
    k_o = len(obs_pairs)

    init_counts = [0.0] * k_s
    trans_counts = [[0.0] * k_s for _ in range(k_s)]
    emit_counts = [[0.0] * k_o for _ in range(k_s)]

    used_clips = 0
    frame_count = 0
    for clip in clips:
        frames = clip.get("frames", [])
        labelled = []
        for fr in frames:
            lc = fr.get("left_contact")
            rc = fr.get("right_contact")
            lr = fr.get("left_reading")
            rr = fr.get("right_reading")
            if lc is None or rc is None:
                raise ValueError(f"training clip {clip.get('clip_id')!r} has unlabelled frame")
            if lr is None or rr is None:
                raise ValueError(f"training clip {clip.get('clip_id')!r} frame lacks readings")
            sk = pair_key((lc, rc))
            ok = pair_key((lr, rr))
            if sk not in s_index:
                raise ValueError(f"unknown contact pair {sk!r}; vertices={vertices}")
            if ok not in o_index:
                raise ValueError(f"unknown reading pair {ok!r}; reading_values={reading_values}")
            labelled.append((s_index[sk], o_index[ok]))
        if not labelled:
            continue
        used_clips += 1
        init_counts[labelled[0][0]] += 1.0
        for si, oi in labelled:
            emit_counts[si][oi] += 1.0
            frame_count += 1
        for (si, _), (sj, _) in zip(labelled, labelled[1:]):
            trans_counts[si][sj] += 1.0

    initial = normalize_counts(init_counts, alpha)
    transition = [normalize_counts(row, alpha) for row in trans_counts]
    emission = [normalize_counts(row, alpha) for row in emit_counts]
    meta = dict(metadata or {})
    meta.update({"trained_clips": used_clips, "labelled_frames": frame_count})
    return JointModel(
        vertices=vertices,
        reading_values=reading_values,
        states=states,
        reading_pairs=obs_pairs,
        initial=initial,
        transition=transition,
        emission=emission,
        model_version=model_version,
        alpha=alpha,
        metadata=meta,
    )


def corpus_vocabulary(corpus: dict[str, Any]) -> tuple[list[str], list[str]]:
    vertices = corpus.get("vertices")
    readings = corpus.get("reading_values")
    if vertices and readings:
        return list(vertices), list(readings)
    # Fallback: derive deterministically if a corpus omits top-level vocab.
    v: list[str] = []
    r: list[str] = []
    for clip in corpus.get("clips", []):
        for fr in clip.get("frames", []):
            for key, bucket in (("left_contact", v), ("right_contact", v)):
                x = fr.get(key)
                if x is not None and x not in bucket:
                    bucket.append(x)
            for key, bucket in (("left_reading", r), ("right_reading", r)):
                x = fr.get(key)
                if x is not None and x not in bucket:
                    bucket.append(x)
    return sorted(v), sorted(r)
