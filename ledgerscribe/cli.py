"""LedgerScribe 命令行。

用法总览：
  python -m ledgerscribe.cli generate --data data
  python -m ledgerscribe.cli train  --data data --writer A --out data/model_A.json
  python -m ledgerscribe.cli line   --model data/model_A.json \
        --lines data/heldout_lines.json --line-id A-line-1 \
        --page data/pages.json --page-id A-040
  python -m ledgerscribe.cli page   --model data/model_A.json \
        --lines data/heldout_lines.json --page data/pages.json --page-id A-040
  python -m ledgerscribe.cli plan   ...      # 路线 A 最坏答复回查规划
  python -m ledgerscribe.cli reply  ... --reply reply.json   # 接收另一份新页+回复
  python -m ledgerscribe.cli check  ...      # 极短轨迹独立枚举检查
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import Model, train
from .page import (
    apply_review_dialogue,
    n_signed,
    plan_review,
    reconcile,
)
from .segments import segment_digit_responsibility, split_segments
from .synth import (
    make_new_writer_page,
    make_writer_material,
    write_heldout_pages,
)


# --------------------------------------------------------------------------
# 同格式输入装载：行可来自独立行文件，也可内嵌在 page 文件中
# --------------------------------------------------------------------------
def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_lines(path: Path) -> dict[str, dict]:
    data = _load_json(path)
    if isinstance(data, dict):
        data = data.get("lines", [])
    return {x["line_id"]: x for x in data}


def load_pages(path: Path) -> dict[str, dict]:
    data = _load_json(path)
    if isinstance(data, dict):
        data = data.get("pages", [data])
    return {p["page_id"]: p for p in data}


def load_registry(data_dir: Path) -> tuple[dict, dict]:
    """合并目录内全部 *lines*.json 行文件与 *page*.json 页文件。"""
    lines, pages = {}, {}
    for f in sorted(Path(data_dir).glob("*.json")):
        obj = _load_json(f)
        if isinstance(obj, list) and obj and "tokens" in obj[0]:
            lines.update({x["line_id"]: x for x in obj})
        elif isinstance(obj, dict) and "pages" in obj:
            pages.update({p["page_id"]: p for p in obj["pages"]})
        elif isinstance(obj, dict) and "candidates" in obj:
            pages[obj["page_id"]] = obj
            if "lines" in obj:
                lines.update({x["line_id"]: x for x in obj["lines"]})
    return lines, pages


def infer_page(model: Model, lines: dict, page: dict) -> dict:
    inferred = {}
    for lid in page["line_ids"]:
        inferred[lid] = model.infer_line(
            lines[lid]["tokens"], page["candidates"][lid]
        )
    return inferred


# --------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------
def cmd_generate(args) -> None:
    d = Path(args.data)
    d.mkdir(parents=True, exist_ok=True)
    make_writer_material(d, "A", seed=101, direction_noise=0.12)
    make_writer_material(d, "B", seed=202, direction_noise=0.34)
    write_heldout_pages(d)
    make_new_writer_page(d)
    print(json.dumps({"generated": sorted(p.name for p in d.iterdir())},
                     ensure_ascii=False, indent=2))


def cmd_train(args) -> None:
    d = Path(args.data)
    rows = [
        json.loads(line)
        for line in (d / f"{args.writer}_train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    model = train(rows, iterations=args.iterations)
    out = Path(args.out)
    model.save(out)
    print(json.dumps({"model": str(out), "train_lines": len(rows)},
                     ensure_ascii=False, indent=2))


def _line_report(model: Model, line: dict, candidates: list[str]) -> dict:
    inferred = model.infer_line(line["tokens"], candidates)
    ranking = sorted(inferred.items(), key=lambda kv: -kv[1]["prob"])
    report = {
        "line_id": line["line_id"],
        "n_segments": len(split_segments(line["tokens"])),
        "amount_ranking": [
            {"amount": a, "prob": round(e["prob"], 4),
             "n_segmentations": len(e["paths"])}
            for a, e in ranking
        ],
        "per_amount_segment_responsibility": {
            a: segment_digit_responsibility(line["tokens"], inferred, a)
            for a, _ in ranking
        },
    }
    return report


def cmd_line(args) -> None:
    model = Model.load(Path(args.model))
    lines = load_lines(Path(args.lines))
    pages = load_pages(Path(args.page)) if args.page else {}
    page = pages[args.page_id] if args.page_id else None
    candidates = (
        page["candidates"][args.line_id]
        if page else args.candidates.split(",")
    )
    print(json.dumps(
        _line_report(model, lines[args.line_id], candidates),
        ensure_ascii=False, indent=2,
    ))


def cmd_page(args) -> None:
    model = Model.load(Path(args.model))
    lines = load_lines(Path(args.lines))
    page = load_pages(Path(args.page))[args.page_id]
    inferred = infer_page(model, lines, page)
    result = reconcile(inferred, page)
    result["model"] = str(args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_plan(args) -> None:
    model = Model.load(Path(args.model))
    lines = load_lines(Path(args.lines))
    page = load_pages(Path(args.page))[args.page_id]
    inferred = infer_page(model, lines, page)
    tokens_by_line = {lid: lines[lid]["tokens"] for lid in page["line_ids"]}
    plan = plan_review(inferred, tokens_by_line, page, budget=2)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def cmd_reply(args) -> None:
    model = Model.load(Path(args.model))
    lines = load_lines(Path(args.lines))
    page = load_pages(Path(args.page))[args.page_id]
    reply_doc = _load_json(Path(args.reply))
    replies = reply_doc["replies"] if isinstance(reply_doc, dict) else reply_doc
    inferred = infer_page(model, lines, page)
    tokens_by_line = {lid: lines[lid]["tokens"] for lid in page["line_ids"]}
    result = apply_review_dialogue(inferred, tokens_by_line, page, replies)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_check(args) -> None:
    """极短轨迹的独立枚举检查：用两套独立实现互证后验与字归属。"""
    from .indep_check import independent_check

    model = Model.load(Path(args.model))
    lines = load_lines(Path(args.lines))
    page = load_pages(Path(args.page))[args.page_id]
    report = independent_check(model, lines, page)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ledgerscribe")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--data", default="data")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train")
    t.add_argument("--data", default="data")
    t.add_argument("--writer", default="A")
    t.add_argument("--out", required=True)
    t.add_argument("--iterations", type=int, default=25)
    t.set_defaults(func=cmd_train)

    l = sub.add_parser("line")
    l.add_argument("--model", required=True)
    l.add_argument("--lines", required=True)
    l.add_argument("--line-id", required=True)
    l.add_argument("--page")
    l.add_argument("--page-id")
    l.add_argument("--candidates", default="")
    l.set_defaults(func=cmd_line)

    pg = sub.add_parser("page")
    pg.add_argument("--model", required=True)
    pg.add_argument("--lines", required=True)
    pg.add_argument("--page", required=True)
    pg.add_argument("--page-id", required=True)
    pg.set_defaults(func=cmd_page)

    pl = sub.add_parser("plan")
    pl.add_argument("--model", required=True)
    pl.add_argument("--lines", required=True)
    pl.add_argument("--page", required=True)
    pl.add_argument("--page-id", required=True)
    pl.set_defaults(func=cmd_plan)

    r = sub.add_parser("reply")
    r.add_argument("--model", required=True)
    r.add_argument("--lines", required=True)
    r.add_argument("--page", required=True)
    r.add_argument("--page-id", required=True)
    r.add_argument("--reply", required=True)
    r.set_defaults(func=cmd_reply)

    c = sub.add_parser("check")
    c.add_argument("--model", required=True)
    c.add_argument("--lines", required=True)
    c.add_argument("--page", required=True)
    c.add_argument("--page-id", required=True)
    c.set_defaults(func=cmd_check)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
