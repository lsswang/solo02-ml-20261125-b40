from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from .archive import create_case, file_calibration_appeal
from .evaluate import evaluate_corpus, render_heldout_markdown
from .handcheck import run_hand_examples, render_markdown as render_hand_markdown
from .inference import adjudicate
from .io_utils import load_clip, load_json, route_card_from_dict, write_json
from .model import JointModel, corpus_vocabulary, train_model
from .report import render_decision_markdown
from .sealed import (
    build_sealed_bundle,
    compare_testimonies,
    render_sealed_markdown,
    rejudge_sealed,
)
from .synthetic import (
    default_route_card,
    generate_calibration_extra,
    generate_corpus,
    generate_false_top_revision,
    make_appeal_attempt,
    make_clip,
    make_sealed_testimonies,
)


def _print(path: Path, what: str) -> None:
    print(f"wrote {what}: {path}")


def cmd_generate_sample(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    data_dir = outdir / "data" if args.layout else outdir
    data_dir.mkdir(parents=True, exist_ok=True)

    corpus = generate_corpus(seed=args.seed, n_train=args.train, n_heldout=args.heldout)
    corpus_path = data_dir / "synthetic_corpus.json"
    write_json(corpus_path, corpus)
    _print(corpus_path, "synthetic labeled corpus")

    extra = generate_calibration_extra(seed=args.calibration_seed)
    extra_path = data_dir / "calibration_extra.json"
    write_json(extra_path, extra)
    _print(extra_path, "later calibration batch")

    revision = generate_false_top_revision()
    revision_path = data_dir / "calibration_false_top_revision.json"
    write_json(revision_path, revision)
    _print(revision_path, "paired false-TOP calibration revision")

    route = default_route_card()
    route_path = data_dir / "route_card.json"
    write_json(route_path, route)
    _print(route_path, "route card")

    rng = random.Random(args.new_clip_seed)
    new_clip_full = make_clip(rng, "public-new-W5-014", "top", length=12, split="new")
    new_clip = {
        "format": "boulder-clip/v1",
        "clip_id": new_clip_full["clip_id"],
        "athlete": "demo-athlete-14",
        "frames": [
            {"frame": i, "left_reading": f["left_reading"], "right_reading": f["right_reading"]}
            for i, f in enumerate(new_clip_full["frames"])
        ],
    }
    new_path = data_dir / "new_attempt.json"
    write_json(new_path, new_clip)
    _print(new_path, "new readings-only clip")
    truth_path = data_dir / "new_attempt.oracle-truth.json"
    write_json(truth_path, new_clip_full)
    _print(truth_path, "oracle truth for verification only")

    appeal_clip, appeal_truth = make_appeal_attempt()
    appeal_path = data_dir / "appeal_attempt.json"
    write_json(appeal_path, appeal_clip)
    _print(appeal_path, "appeal readings-only clip")
    appeal_truth_path = data_dir / "appeal_attempt.oracle-truth.json"
    write_json(appeal_truth_path, appeal_truth)
    _print(appeal_truth_path, "appeal oracle truth for verification only")

    testimony_a, testimony_b = make_sealed_testimonies()
    testimony_a_path = data_dir / "sealed_testimony_a_top.json"
    testimony_b_path = data_dir / "sealed_testimony_b_zone.json"
    write_json(testimony_a_path, testimony_a)
    write_json(testimony_b_path, testimony_b)
    _print(testimony_a_path, "example frame-by-frame testimony A")
    _print(testimony_b_path, "example frame-by-frame testimony B")

    template = {
        "format": "boulder-clip/v1",
        "clip_id": "replace-me",
        "frames": [
            {"frame": 0, "left_reading": "NONE", "right_reading": "NONE"},
            {"frame": 1, "left_reading": "TOP", "right_reading": "TOP"},
        ],
        "_allowed_readings": corpus["reading_values"],
    }
    template_path = data_dir / "same_format_template.json"
    write_json(template_path, template)
    _print(template_path, "same-format input template")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    merged_clips: list[dict[str, Any]] = []
    vertices = None
    readings = None
    sources = []
    for raw_path in args.data:
        path = Path(raw_path)
        data = load_json(path)
        sources.append(str(path))
        v, r = corpus_vocabulary(data)
        vertices = vertices or v
        readings = readings or r
        merged_clips.extend(data.get("clips", []))
        if "frames" in data and ("left_contact" in data["frames"][0] if data.get("frames") else False):
            merged_clips.append(data)
    if not merged_clips:
        raise ValueError("no labeled clips found in --data inputs")
    if args.all_clips:
        train_clips = merged_clips
    else:
        train_clips = [c for c in merged_clips if c.get("split", "train") == "train"]
    model = train_model(
        train_clips,
        vertices or ["NONE", "ZONE", "TOP"],
        readings or ["NONE", "ZONE", "TOP", "OCCLUDED"],
        alpha=args.alpha,
        model_version=args.version,
        metadata={"sources": sources, "all_clips": bool(args.all_clips)},
    )
    out = Path(args.out)
    model.save(out)
    _print(out, f"model {model.model_version} fingerprint={model.fingerprint()[:12]}")
    return 0


def _load_route(path: str | None) -> dict[str, Any]:
    if path is None:
        return default_route_card()
    return route_card_from_dict(load_json(path))


def cmd_handcheck(args: argparse.Namespace) -> int:
    report = run_hand_examples()
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    write_json(out_json, report)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_hand_markdown(report), encoding="utf-8")
    _print(out_json, "hand-check JSON")
    _print(out_md, "hand-check Markdown")
    if args.strict and not report["all_expected_values_reproduced"]:
        return 2
    return 0


def cmd_heldout(args: argparse.Namespace) -> int:
    corpus = load_json(args.data)
    model = JointModel.load(args.model)
    route = _load_route(args.route)
    report = evaluate_corpus(model, corpus, route, split=args.split)
    write_json(args.out_json, report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_heldout_markdown(report), encoding="utf-8")
    _print(Path(args.out_json), "held-out JSON")
    _print(Path(args.out_md), "held-out Markdown")
    print(
        f"TV={report['joint_distribution_check']['total_variation']:.4f} "
        f"signed_coverage={report['signed_coverage']:.2f}"
    )
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    clip = load_clip(args.clip, args.clip_id)
    model = JointModel.load(args.model)
    route = _load_route(args.route)
    decision = adjudicate(model, clip, route)
    write_json(args.out_json, decision)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_decision_markdown(decision), encoding="utf-8")
    _print(Path(args.out_json), "decision JSON")
    _print(Path(args.out_md), "decision Markdown")

    if args.case_root and args.appeal_number:
        case_dir = create_case(
            args.case_root,
            args.appeal_number,
            clip,
            route,
            model,
            decision,
            input_source=args.clip,
            model_source=args.model,
        )
        _print(case_dir / "dossier.json", "appeal case dossier")
        _print(case_dir / "DOSSIER.md", "appeal case Markdown")
    elif args.case_root or args.appeal_number:
        raise ValueError("both --case-root and --appeal-number are required to archive a case")

    probs = ", ".join(f"{x['code']}={x['probability']:.4f}" for x in decision["score_probabilities"])
    print(f"{decision['clip_id']}: signed={decision['signed_class']} ({probs})")
    return 0


def cmd_appeal(args: argparse.Namespace) -> int:
    new_model = JointModel.load(args.new_model)
    dossier, record = file_calibration_appeal(
        args.case_dir,
        new_model,
        reason=args.reason,
        appellant=args.appellant,
        appeal_id=args.appeal_id,
    )
    case_dir = Path(args.case_dir)
    out_json = Path(args.out_json) if args.out_json else case_dir / "latest_appeal_result.json"
    write_json(out_json, {"dossier": dossier, "latest_record": record})
    _print(out_json, "appeal result JSON")
    _print(case_dir / "DOSSIER.md", "updated same-number dossier")
    print(
        f"{dossier['appeal_number']}: disposition={record['disposition']} "
        f"new_signed={record['new_signed_class']} reproduced={record['old_judgment_reproduction']['matches_initial_judgment']}"
    )
    return 0


def cmd_sealed_preview(args: argparse.Namespace) -> int:
    clip = load_clip(args.clip, args.clip_id)
    model = JointModel.load(args.model)
    route = _load_route(args.route)
    bundle = build_sealed_bundle(model, clip, route)
    if args.appeal_number:
        bundle["appeal_number"] = args.appeal_number
    bundle_path = Path(args.bundle_json)
    write_json(bundle_path, bundle)
    _print(bundle_path, "sealed joint evidence bundle (raw readings omitted)")

    result: dict[str, Any] = {"sealed_rejudgment": rejudge_sealed(bundle)}
    if args.testimony_a and args.testimony_b:
        testimony_a = load_json(args.testimony_a)
        testimony_b = load_json(args.testimony_b)
        comparison = compare_testimonies(bundle, testimony_a, testimony_b)
        result["testimony_comparison"] = comparison
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_sealed_markdown(comparison), encoding="utf-8")
        _print(out_md, "sealed testimony comparison Markdown")
    write_json(args.out_json, result)
    _print(Path(args.out_json), "sealed preview JSON")
    probs = result["sealed_rejudgment"]["probabilities"]
    print("sealed rejudgment:", {k: round(v, 6) for k, v in probs.items()})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="boulder-judge", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate-sample", help="generate deterministic small sample bundle")
    g.add_argument("--outdir", default=".")
    g.add_argument("--seed", type=int, default=20261125)
    g.add_argument("--calibration-seed", type=int, default=9917)
    g.add_argument("--new-clip-seed", type=int, default=4242)
    g.add_argument("--train", type=int, default=24)
    g.add_argument("--heldout", type=int, default=8)
    g.add_argument("--no-layout", dest="layout", action="store_false")
    g.set_defaults(func=cmd_generate_sample, layout=True)

    t = sub.add_parser("train", help="train joint HMM from labeled synthetic JSON")
    t.add_argument("--data", action="append", required=True, help="labeled corpus; may be repeated")
    t.add_argument("--out", required=True)
    t.add_argument("--alpha", type=float, default=0.35)
    t.add_argument("--version", default="trained-v1")
    t.add_argument("--all-clips", action="store_true")
    t.set_defaults(func=cmd_train)

    h = sub.add_parser("handcheck", help="run four hand-enumerable examples")
    h.add_argument("--out-json", default="deliverables/hand_examples.json")
    h.add_argument("--out-md", default="deliverables/hand_examples.md")
    h.add_argument("--strict", action="store_true")
    h.set_defaults(func=cmd_handcheck)

    e = sub.add_parser("heldout", help="check joint distribution on whole held-out clips")
    e.add_argument("--data", default="data/synthetic_corpus.json")
    e.add_argument("--model", default="models/model_v1.json")
    e.add_argument("--route", default="data/route_card.json")
    e.add_argument("--split", default="heldout")
    e.add_argument("--out-json", default="deliverables/heldout_report.json")
    e.add_argument("--out-md", default="deliverables/heldout_report.md")
    e.set_defaults(func=cmd_heldout)

    j = sub.add_parser("judge", help="adjudicate one new clip")
    j.add_argument("--clip", required=True)
    j.add_argument("--clip-id", default=None)
    j.add_argument("--model", default="models/model_v1.json")
    j.add_argument("--route", default="data/route_card.json")
    j.add_argument("--out-json", default="deliverables/decision.json")
    j.add_argument("--out-md", default="deliverables/decision.md")
    j.add_argument("--case-root", default=None)
    j.add_argument("--appeal-number", default=None)
    j.set_defaults(func=cmd_judge)

    a = sub.add_parser("appeal", help="file a calibration appeal using retained readings")
    a.add_argument("--case-dir", required=True)
    a.add_argument("--new-model", required=True)
    a.add_argument("--reason", required=True)
    a.add_argument("--appellant", default="head_judge_review")
    a.add_argument("--appeal-id", default=None)
    a.add_argument("--out-json", default=None)
    a.set_defaults(func=cmd_appeal)

    s = sub.add_parser(
        "sealed-preview",
        help="optional alternative venue: export joint factors and compare two testimonies",
    )
    s.add_argument("--clip", required=True)
    s.add_argument("--clip-id", default=None)
    s.add_argument("--model", default="models/model_v1.json")
    s.add_argument("--route", default="data/route_card.json")
    s.add_argument("--testimony-a", default=None)
    s.add_argument("--testimony-b", default=None)
    s.add_argument("--appeal-number", default=None)
    s.add_argument("--bundle-json", default="deliverables/sealed_evidence_bundle.json")
    s.add_argument("--out-json", default="deliverables/sealed_preview.json")
    s.add_argument("--out-md", default="deliverables/sealed_preview.md")
    s.set_defaults(func=cmd_sealed_preview)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # CLI boundary: concise actionable error.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
