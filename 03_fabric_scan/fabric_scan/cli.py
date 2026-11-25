"""命令行入口：生成材料、学习模型、演示、正常下单。

普通下单入口示例：
  python -m fabric_scan.cli place-order \
      --model out/model_a.json --order order.json \
      --precision precision.json --ledger out/ledger.json

order.json 字段：
  order_id, fabric_id, device(portable|bridge),
  losses(每格损失), retain(留样格), head0(初始头位),
  startup_fee, move_fee, bridge_fee, readings(含 null 缺读)
precision.json 字段：格号字符串 -> 0好/1坏，只给该台实际能返回的格。
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from .common import dump_json, load_json
from .datagen import MATERIALS, make_dataset, split_fabrics, write_dataset
from .learn import evaluate, learn_model, save_model
from .order import new_ledger, save_ledger
from .workflow import place_order


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def cmd_gen_data(args: argparse.Namespace) -> None:
    config = MATERIALS[args.material]
    dataset = make_dataset(config)
    write_dataset(dataset, args.out)
    print(f"已生成材料 {args.material}: {len(dataset['fabrics'])} 条布 -> {args.out}")


def cmd_learn(args: argparse.Namespace) -> None:
    dataset = load_json(args.data)
    parts = split_fabrics(dataset)
    model = learn_model(parts["train"], dataset["config"]["n"],
                        alpha=args.alpha, beta=args.beta)
    save_model(model, args.out)
    print(f"训练 {model['n_train']} 条，用读数 {model['n_readings_used']} 次，"
          f"缺读略过 {model['n_missing_skipped']} 次 -> {args.out}")
    if args.holdout_report:
        report = evaluate(model, parts["holdout"])
        dump_json(report, args.holdout_report)
        print(f"留出 {report['n_holdout']} 条：MAP 准确率 "
              f"{report['map_accuracy']:.3f}，平均对数损失 "
              f"{report['mean_log_loss']:.4f}，Brier {report['mean_brier']:.4f} "
              f"-> {args.holdout_report}")


def cmd_place_order(args: argparse.Namespace) -> None:
    model = load_json(args.model)
    order = load_json(args.order)
    precision_raw = load_json(args.precision)
    precision = {int(k): int(v) for k, v in precision_raw.items()}
    ledger = new_ledger(args.ledger)
    result = place_order(model, order, precision, ledger)
    save_ledger(ledger, args.ledger)
    _print_header(f"下单结果 {result['order_id']}（{result['device']}）")
    print(f"布号 {result['fabric_id']}")
    print(f"下单时策略: {json.dumps(result['plan_at_order_time'], ensure_ascii=False)}")
    print(f"实际复扫: {result['scan_steps']}，扫描费 {result['scan_fee']}")
    print(f"剪口区段: {result['cut_segment']}，剪去格 {result['cut_cells']}")
    print(f"期望剪口损失 {result['expected_cut_loss']:.4f}，"
          f"期望总成本 {result['expected_total_cost']:.4f}")
    print(f"档案已更新 -> {args.ledger}；当前不可再剪格 "
          f"{result['blocked_cells_after']}")


def cmd_demo(args: argparse.Namespace) -> None:
    from .demo import run_demo
    run_demo(args.out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="布面复扫与裁切软件")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gen-data", help="生成小型合成材料")
    p.add_argument("--material", choices=sorted(MATERIALS), required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_gen_data)

    p = sub.add_parser("learn", help="学习先验与误读分布并评估留出布")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--holdout-report", default=None)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("place-order", help="正常下单入口：准备复扫、收读数、开剪口")
    p.add_argument("--model", required=True)
    p.add_argument("--order", required=True)
    p.add_argument("--precision", required=True)
    p.add_argument("--ledger", required=True)
    p.set_defaults(func=cmd_place_order)

    p = sub.add_parser("demo", help="题目指定的两格玩具场景逐项核对")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: Any = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
