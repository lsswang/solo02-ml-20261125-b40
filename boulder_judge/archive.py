"""Appeal archive for the venue allowed to retain raw wrist readings.

Each appeal number is one immutable case directory.  The directory contains the
submitted paired readings, route card, the exact model snapshot used for the
original signature, and every later calibration/decision.  Consequently an old
judgment can be reproduced before a changed calibration is allowed to alter it.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inference import adjudicate
from .io_utils import write_json
from .model import JointModel


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _probability_diff(a: dict[str, Any], b: dict[str, Any]) -> float:
    pa = {x["code"]: x["probability"] for x in a["score_probabilities"]}
    pb = {x["code"]: x["probability"] for x in b["score_probabilities"]}
    return max(abs(pa[k] - pb[k]) for k in pa)


def create_case(
    root: str | Path,
    appeal_number: str,
    clip: dict[str, Any],
    route: dict[str, Any],
    model: JointModel,
    decision: dict[str, Any],
    *,
    input_source: str | None = None,
    model_source: str | Path | None = None,
) -> Path:
    case_dir = Path(root) / appeal_number
    if case_dir.exists():
        raise FileExistsError(f"appeal case already exists: {case_dir}")
    (case_dir / "models").mkdir(parents=True)
    (case_dir / "decisions").mkdir()
    (case_dir / "appeals").mkdir()
    write_json(case_dir / "submission" / "clip.json", clip)
    write_json(case_dir / "route_card.json", route)
    model.save(case_dir / "models" / "initial_model.json")
    write_json(case_dir / "decisions" / "initial_decision.json", decision)
    if model_source is not None and Path(model_source).exists():
        shutil.copyfile(model_source, case_dir / "models" / "initial_model.source-copy.json")

    dossier = {
        "format": "boulder-appeal-dossier/v1",
        "appeal_number": appeal_number,
        "archive_mode": "retain_wrist_readings",
        "created_at": _now(),
        "clip_id": clip.get("clip_id"),
        "route_id": route.get("route_id"),
        "raw_readings_retained": True,
        "input_source": str(input_source) if input_source else None,
        "initial": {
            "signed": decision["signed"],
            "signed_class": decision["signed_class"],
            "status": decision["status"],
            "score_probabilities": decision["score_probabilities"],
            "model_version": model.model_version,
            "model_fingerprint": model.fingerprint(),
            "decision_file": "decisions/initial_decision.json",
        },
        "appeals": [],
        "current_status": decision["status"],
        "current_signed_class": decision["signed_class"],
    }
    write_json(case_dir / "dossier.json", dossier)
    (case_dir / "DOSSIER.md").write_text(render_dossier(dossier, decision), encoding="utf-8")
    return case_dir


def load_dossier(case_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(case_dir) / "dossier.json").read_text(encoding="utf-8"))


def _reproduce(
    model: JointModel, recorded: dict[str, Any], recomputed: dict[str, Any]
) -> dict[str, Any]:
    max_diff = _probability_diff(recorded, recomputed)
    return {
        "reproduced_at": _now(),
        "model_fingerprint": model.fingerprint(),
        "recorded_model_fingerprint": recorded.get("model_fingerprint"),
        "fingerprint_match": model.fingerprint() == recorded.get("model_fingerprint"),
        "max_abs_probability_difference": max_diff,
        "signed_class_match": recomputed["signed_class"] == recorded["signed_class"],
        "signed_match": recomputed["signed"] == recorded["signed"],
        "matches_recorded_judgment": (
            model.fingerprint() == recorded.get("model_fingerprint")
            and recomputed["signed_class"] == recorded["signed_class"]
            and recomputed["signed"] == recorded["signed"]
            and max_diff < 1e-12
        ),
    }


def file_calibration_appeal(
    case_dir: str | Path,
    new_model: JointModel,
    reason: str,
    *,
    appellant: str = "head_judge_review",
    appeal_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_dir = Path(case_dir)
    dossier = load_dossier(case_dir)
    clip = json.loads((case_dir / "submission" / "clip.json").read_text(encoding="utf-8"))
    route = json.loads((case_dir / "route_card.json").read_text(encoding="utf-8"))

    initial_model = JointModel.load(case_dir / "models" / "initial_model.json")
    initial_recorded = json.loads(
        (case_dir / "decisions" / "initial_decision.json").read_text(encoding="utf-8")
    )
    initial_recomputed = adjudicate(initial_model, clip, route)
    initial_reproduction = _reproduce(initial_model, initial_recorded, initial_recomputed)
    # Backward-compatible key for the first appeal and existing tests.
    initial_reproduction["matches_initial_judgment"] = initial_reproduction[
        "matches_recorded_judgment"
    ]

    if dossier["appeals"]:
        last = dossier["appeals"][-1]
        prior_model = JointModel.load(case_dir / last["model_file"])
        prior_recorded = json.loads((case_dir / last["decision_file"]).read_text(encoding="utf-8"))
        prior_recomputed = adjudicate(prior_model, clip, route)
        prior_reproduction = _reproduce(prior_model, prior_recorded, prior_recomputed)
    else:
        prior_model = initial_model
        prior_recorded = initial_recorded
        prior_reproduction = dict(initial_reproduction)
    reproduction = prior_reproduction

    new_decision = adjudicate(new_model, clip, route)
    prior_class = dossier.get("current_signed_class")
    prior_status = dossier.get("current_status")
    if new_decision["signed"] and prior_class != new_decision["signed_class"]:
        disposition = "revised"
    elif new_decision["signed"] and prior_class == new_decision["signed_class"]:
        disposition = "upheld"
    elif not new_decision["signed"] and prior_status == new_decision["status"]:
        disposition = "upheld_unsigned"
    else:
        disposition = "unsigned_after_review"

    n = len(dossier["appeals"]) + 1
    appeal_id = appeal_id or f"{dossier['appeal_number']}-R{n:02d}"
    new_model_path = case_dir / "models" / f"appeal-{n:02d}-model.json"
    new_decision_path = case_dir / "decisions" / f"appeal-{n:02d}-decision.json"
    new_model.save(new_model_path)
    write_json(new_decision_path, new_decision)

    record = {
        "appeal_id": appeal_id,
        "filed_at": _now(),
        "appellant": appellant,
        "reason": reason,
        "review_type": "post_event_recalibration_from_retained_readings",
        "raw_readings_used": "submission/clip.json",
        "initial_judgment_reproduction": initial_reproduction,
        "old_judgment_reproduction": reproduction,
        "prior_status": prior_status,
        "prior_signed_class": prior_class,
        "new_model_version": new_model.model_version,
        "new_model_fingerprint": new_model.fingerprint(),
        "new_score_probabilities": new_decision["score_probabilities"],
        "new_signed": new_decision["signed"],
        "new_signed_class": new_decision["signed_class"],
        "new_status": new_decision["status"],
        "disposition": disposition,
        "changed_signed_result": bool(
            reproduction["matches_recorded_judgment"]
            and initial_reproduction["matches_initial_judgment"]
            and (prior_class != new_decision["signed_class"] or prior_status != new_decision["status"])
        ),
        "model_file": f"models/{new_model_path.name}",
        "decision_file": f"decisions/{new_decision_path.name}",
    }
    dossier["appeals"].append(record)
    dossier["current_status"] = new_decision["status"]
    dossier["current_signed_class"] = new_decision["signed_class"]
    write_json(case_dir / "dossier.json", dossier)
    (case_dir / "DOSSIER.md").write_text(
        render_dossier(dossier, latest_decision=new_decision), encoding="utf-8"
    )
    write_json(case_dir / "appeals" / f"appeal-{n:02d}.json", record)
    return dossier, record


def _fmt_probs(prob_list: list[dict[str, Any]]) -> str:
    return ", ".join(f"{p['label_zh']} {p['probability']:.4f}" for p in prob_list)


def render_dossier(dossier: dict[str, Any], latest_decision: dict[str, Any] | None = None) -> str:
    initial = dossier["initial"]
    lines = [
        f"# 申诉档案 {dossier['appeal_number']}",
        "",
        f"- 档案模式：{dossier['archive_mode']}（保留双腕离散读数）",
        f"- 片段：`{dossier.get('clip_id')}`；路线卡：`{dossier.get('route_id')}`",
        f"- 建档时间：{dossier.get('created_at')}",
        f"- 当前状态：**{dossier.get('current_status')}**；当前签单类别：**{dossier.get('current_signed_class') or '无'}**",
        "",
        "## 最初签了什么",
        "",
        f"- 初始模型：`{initial['model_version']}` (`{initial['model_fingerprint'][:12]}…`)",
        f"- 初始概率：{_fmt_probs(initial['score_probabilities'])}",
        f"- 初始签单：**{initial['signed_class'] or '未签单'}**（{initial['status']}）",
        "",
    ]
    if dossier.get("appeals"):
        lines.append("## 申诉与改判链")
        lines.append("")
        for rec in dossier["appeals"]:
            rep = rec["old_judgment_reproduction"]
            reproduced = rep.get(
                "matches_initial_judgment", rep.get("matches_recorded_judgment", False)
            )
            lines.extend(
                [
                    f"### {rec['appeal_id']} — {rec['disposition']}",
                    "",
                    f"- 理由：{rec['reason']}",
                    f"- 上一判复现：{'成功' if reproduced else '失败'}；"
                    f"最大概率差 {rep['max_abs_probability_difference']:.3g}",
                    f"- 最初判单复现：{'成功' if rec['initial_judgment_reproduction']['matches_initial_judgment'] else '失败'}",
                    f"- 新校准：`{rec['new_model_version']}` (`{rec['new_model_fingerprint'][:12]}…`)",
                    f"- 新概率：{_fmt_probs(rec['new_score_probabilities'])}",
                    f"- 新签单：**{rec['new_signed_class'] or '未达 0.95，可继续申诉'}**",
                    f"- 是否改变结果：**{'是' if rec['changed_signed_result'] else '否'}**",
                    "",
                ]
            )
    if latest_decision is not None:
        lines.extend(
            [
                "## 最新逐帧触顶证据",
                "",
                "| 帧 | P(左,右均在顶点) | 证据档 | 读数 |",
                "|---:|---:|---|---|",
            ]
        )
        for fr in latest_decision["frame_evidence"]:
            lines.append(
                f"| {fr['frame']} | {fr['p_both_at_top_vertex']:.4f} | {fr['evidence_band']} | "
                f"{fr['readings']['left']}/{fr['readings']['right']} |"
            )
        lines.append("")
    return "\n".join(lines)
