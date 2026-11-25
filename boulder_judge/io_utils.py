"""JSON IO and small schema helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CLIP_FORMATS = {"boulder-clip/v1", "boulder-attempt/v1"}
CORPUS_FORMATS = {"boulder-synthetic-corpus/v1", "boulder-corpus/v1"}
ROUTE_FORMAT = "boulder-route-card/v1"


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def clips_from_input(data: dict[str, Any], clip_id: str | None = None) -> list[dict[str, Any]]:
    """Accept either one clip object or a corpus containing a ``clips`` list."""
    if "frames" in data:
        if clip_id and data.get("clip_id") != clip_id:
            raise KeyError(f"input clip_id={data.get('clip_id')!r}, expected {clip_id!r}")
        return [data]
    clips = list(data.get("clips", []))
    if clip_id is not None:
        clips = [c for c in clips if c.get("clip_id") == clip_id]
        if not clips:
            raise KeyError(f"no clip with clip_id={clip_id!r}")
    return clips


def load_clip(path: str | Path, clip_id: str | None = None) -> dict[str, Any]:
    data = load_json(path)
    clips = clips_from_input(data, clip_id)
    if not clips:
        raise ValueError(f"no clips found in {path}")
    if clip_id is None and len(clips) == 1:
        return clips[0]
    if clip_id is None and "frames" in data:
        return data
    if clip_id is None:
        ids = [c.get("clip_id") for c in clips]
        raise ValueError(f"{path} contains multiple clips {ids}; pass --clip-id")
    return clips[0]


def route_card_from_dict(d: dict[str, Any]) -> dict[str, Any]:
    if d.get("format") != ROUTE_FORMAT:
        raise ValueError(f"unsupported route card format: {d.get('format')!r}")
    route = dict(d)
    route.setdefault("frame_index_base", 0)
    route.setdefault("zone_requires", "any_hand")
    route.setdefault("threshold", 0.95)
    if int(route["stable_frames"]) < 1:
        raise ValueError("stable_frames must be >= 1")
    if not 0.0 < float(route["threshold"]) <= 1.0:
        raise ValueError("threshold must be in (0,1]")
    return route
