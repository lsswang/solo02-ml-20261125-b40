from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

from .model import BlendModel
from .pipeline import run_pipeline
from .recommend import recommend_recipes
from .shop import ShopError, ShopService
from .synthetic import generate_votes
from .utils import read_json, write_json


def _print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False))


def _load_model(path: Optional[str]) -> Optional[BlendModel]:
    if not path:
        return None
    return BlendModel.from_dict(read_json(path))


def _default_model_path(state_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(state_dir, "model.json"),
        os.path.join(state_dir, "..", "model.json"),
        os.path.join(state_dir, "..", "..", "model.json"),
        os.path.join("deliverables", "good", "model.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def _open_shop(args: argparse.Namespace) -> ShopService:
    model_path = getattr(args, "model", None) or _default_model_path(args.state_dir)
    model = _load_model(model_path) if model_path else None
    return ShopService(args.state_dir, model=model)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="coffee-blend", description="Coffee blend learning and local shop simulator")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("generate-data", help="generate a reproducible synthetic same-format CSV")
    sp.add_argument("--out", required=True, help="output CSV path")
    sp.add_argument("--scenario", choices=["good", "bad_b"], default="good")
    sp.add_argument("--seed", type=int, default=231125)

    sp = sub.add_parser("pipeline", help="train, evaluate, recommend and execute plan.json")
    sp.add_argument("--input-dir", required=True)
    sp.add_argument("--out-dir", required=True)

    sp = sub.add_parser("init-shop", help="initialize an append-only local shop state")
    sp.add_argument("--state", required=True, dest="state_dir")
    sp.add_argument("--inventory", required=True)
    sp.add_argument("--model")
    sp.add_argument("--reset", action="store_true")

    sp = sub.add_parser("order", help="receive an order and decide bucket/trial handling")
    sp.add_argument("--state", required=True, dest="state_dir")
    sp.add_argument("--model")
    sp.add_argument("--id", required=True, dest="order_id")
    sp.add_argument("--cups", type=int, default=1)
    sp.add_argument("--composition", required=True, help="e.g. JIA-2409:2|YI-2409:2")
    sp.add_argument("--roast", required=True, dest="roast_id")
    sp.add_argument("--formula", dest="formula_id")
    sp.add_argument("--mode", choices=["bucket", "trial"], default="bucket")
    sp.add_argument("--shift", dest="shift_id")
    sp.add_argument("--require-trial", action="store_true")

    sp = sub.add_parser("trial", help="record the shift's blind like/unlike trial reading")
    sp.add_argument("--state", required=True, dest="state_dir")
    sp.add_argument("--model")
    sp.add_argument("--id", required=True, dest="order_id")
    sp.add_argument("--result", required=True, choices=["like", "unlike", "像", "不像"])

    sp = sub.add_parser("serve", help="dispense/serve an accepted order; successful direct weighing is recorded")
    sp.add_argument("--state", required=True, dest="state_dir")
    sp.add_argument("--model")
    sp.add_argument("--id", required=True, dest="order_id")

    sp = sub.add_parser("scan", help="scan an order and show actual beans/refillable half buckets")
    sp.add_argument("--state", required=True, dest="state_dir")
    sp.add_argument("--model")
    sp.add_argument("--id", required=True, dest="order_id")
    sp.add_argument("--save")

    sp = sub.add_parser("stock", help="show remaining loose portions and buckets")
    sp.add_argument("--state", required=True, dest="state_dir")
    sp.add_argument("--model")

    sp = sub.add_parser("withdraw", help="append an external stock withdrawal/recount event")
    sp.add_argument("--state", required=True, dest="state_dir")
    sp.add_argument("--model")
    sp.add_argument("--batch", required=True, dest="batch_id")
    sp.add_argument("--parts", required=True, type=int)
    sp.add_argument("--reason", default="external withdrawal")

    sp = sub.add_parser("recommend", help="list gate probabilities and feasible cup counts")
    sp.add_argument("--model", required=True)
    sp.add_argument("--inventory", required=True)
    sp.add_argument("--threshold", type=float, default=0.8)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-data":
        meta = generate_votes(args.out, args.scenario, args.seed)
        _print(meta)
        return 0
    if args.command == "pipeline":
        _print(run_pipeline(args.input_dir, args.out_dir))
        return 0
    if args.command == "init-shop":
        model = _load_model(args.model)
        shop = ShopService.initialize(args.state_dir, args.inventory, model=model, reset=args.reset)
        _print(shop.stock_report())
        return 0
    if args.command == "recommend":
        model = _load_model(args.model)
        inv = read_json(args.inventory)
        rows = recommend_recipes(model, inv.get("loose_stock_parts", {}), threshold=args.threshold)  # type: ignore[arg-type]
        _print(rows)
        return 0

    try:
        shop = _open_shop(args)
        if args.command == "order":
            res = shop.receive_order(
                args.order_id,
                cups=args.cups,
                composition=args.composition,
                roast_id=args.roast_id,
                formula_id=args.formula_id,
                mode=args.mode,
                shift_id=args.shift_id,
                require_trial=args.require_trial,
            )
        elif args.command == "trial":
            res = shop.record_trial(args.order_id, args.result)
        elif args.command == "serve":
            res = shop.serve_order(args.order_id)
        elif args.command == "scan":
            res = shop.scan_order(args.order_id)
            if args.save:
                write_json(args.save, res)
        elif args.command == "stock":
            res = shop.stock_report()
        elif args.command == "withdraw":
            res = shop.withdraw_stock(args.batch_id, args.parts, args.reason)
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except ShopError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
