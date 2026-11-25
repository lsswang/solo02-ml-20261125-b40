from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .bundle import read_json
from .pipeline import run_pipeline
from .replay import format_replay_text, replay_plates
from .sample import generate_sample


def _split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="guide_studio", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate-sample", help="generate a small reproducible input bundle")
    g.add_argument("--out", default="sample_input")
    g.add_argument("--scenario", choices=["baseline", "west_lin", "high_fee", "revised_script"], default="baseline")
    g.add_argument("--seed", type=int, default=20261125)

    f = sub.add_parser("fit", help="refit referent probabilities and proofread without optimizing audio")
    f.add_argument("--input", required=True)
    f.add_argument("--output", required=True)

    r = sub.add_parser("run", help="fit model, proofread, optimize, render audio and replays")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--close", default="", help="comma-separated station IDs to close")
    r.add_argument("--devices", default="wall,pen")
    r.add_argument("--no-audio", action="store_true")

    rp = sub.add_parser("replay", help="replay one real plate string against a produced plan")
    rp.add_argument("--plan", required=True, help="path to plans/<device>_<lang>.json")
    rp.add_argument("--plates", required=True, help="comma-separated real station plate IDs")
    rp.add_argument("--json-out", default="")

    v = sub.add_parser("verify", help="print key narrative assertions from a run_summary.json")
    v.add_argument("--summary", required=True)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "generate-sample":
        path = generate_sample(args.out, scenario=args.scenario, seed=args.seed)
        print(f"generated {args.scenario} sample bundle at {path}")
        return 0

    if args.command == "fit":
        summary = run_pipeline(args.input, args.output, devices=(), render_audio=False)
        print(json.dumps({
            "output_dir": summary["output_dir"],
            "log_loss": summary["holdout"]["mean_log_loss_nats"],
            "mishearings": summary["holdout"]["selected_two_record_ids"],
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run":
        closed = _split_csv(args.close)
        devices = tuple(_split_csv(args.devices))
        summary = run_pipeline(
            args.input,
            args.output,
            closed_stations=closed,
            devices=devices,
            render_audio=not args.no_audio,
        )
        print(json.dumps({
            "output_dir": summary["output_dir"],
            "log_loss": summary["holdout"]["mean_log_loss_nats"],
            "mishearings": summary["holdout"]["selected_two_record_ids"],
            "plans": summary["plans"],
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "replay":
        plan = read_json(Path(args.plan))
        plates = _split_csv(args.plates)
        rep = replay_plates(plan, plates)
        print(format_replay_text(rep))
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0

    if args.command == "verify":
        s = read_json(Path(args.summary))
        failures = []
        ids = set(s["holdout"]["selected_two_record_ids"])
        if len(ids) < 2:
            failures.append("fewer than two highlighted mishearing records")
        for lang in ("zh", "en"):
            if lang not in s["holdout"]["per_language"]:
                failures.append(f"missing language metrics {lang}")
        if failures:
            print("FAIL: " + "; ".join(failures))
            return 1
        print("OK: two mishearing records and zh/en metrics are present")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
