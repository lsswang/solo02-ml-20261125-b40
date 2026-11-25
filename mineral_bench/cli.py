"""Command-line interface for the mineral optical workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .core import dump_json, load_json
from .generate import generate_all
from .pipeline import run_pipeline
from .workbench import (
    checkout as case_checkout,
    create_case,
    flip as case_flip,
    load_case,
    observe as case_observe,
    status as case_status,
)


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _render_status(st: Dict[str, Any]) -> None:
    cost = st["cost"]
    judgment = st["judgment"]
    print(f"病例/片号: {st['case_id']}    状态: {st['status']}")
    print(
        f"实际花费: {cost['spent']:.2f}/{cost['budget']:.2f} {cost['currency']}    "
        f"剩余预算: {cost['remaining']:.2f}    当前面: {st['current_face']}    "
        f"可翻次数: {st['flips_remaining']}"
    )
    print(f"已录入读数: {json.dumps(st['evidence'], ensure_ascii=False)}")
    post = "  ".join(f"{m}={p:.1%}" for m, p in st["material_posterior"].items())
    print(f"材质后验: {post}")

    possible_joint = [
        j for j in st["joint_material_orientation_posterior"] if j["still_possible"]
    ]
    if possible_joint:
        joint_text = "  ".join(
            f"{j['material']}/{j['orientation']}={j['posterior_joint_probability']:.1%}"
            for j in possible_joint
        )
        print(f"仍可能的材质/取向联合概率: {joint_text}")

    state = judgment["state"]
    if state == "signed_out":
        print(
            f"当前判断: 已签出 {judgment['material']}（{judgment['material_name']}），"
            f"签出时概率 {judgment['probability_at_signoff']:.1%}"
        )
    elif state == "ready_to_checkout":
        print(
            f"当前判断: 已达签出阈值，可签出 {judgment['material']}"
            f"（{judgment['material_name']}），概率 {judgment['probability']:.1%}"
        )
    elif state == "inconsistent_evidence":
        print("当前判断: 读数与所有已知取向模板矛盾；请核对明暗录入。")
    else:
        best = judgment.get("best_material")
        p = judgment.get("best_probability", 0.0)
        extra = f"；当前最高为 {best}={p:.1%}，未到 {judgment['threshold']:.0%}" if best else ""
        print(f"当前判断: 留未定{extra}")

    actions = st["actions"]
    print(
        f"剩余未做试验动作: {actions['remaining_unused_test_count']} "
        f"（当前面 {actions['currently_available_count']} 项；"
        f"可翻次数 {actions['flips_remaining']}）"
    )
    if actions["available_now"]:
        print("现在可做:")
        for a in actions["available_now"]:
            afford = "" if a["affordable_now"] else "（预算不足）"
            print(
                f"  - {a['test_id']}: {a['name']}；本次费用 {a['total_charge_now']:.2f}"
                f"（试验 {a['test_cost']:.2f}+调机 {a['setup_fee_now']:.2f}）{afford}"
            )
    if actions["on_other_face_after_flip"]:
        print("翻面后可做:")
        for a in actions["on_other_face_after_flip"]:
            print(f"  - {a['test_id']}: {a['name']}（{a['side']} 面）")
    if actions["front_specific_actions_lost_by_next_flip"]:
        print(
            "下次翻面会使这些当前面专属动作失效: "
            + ", ".join(actions["front_specific_actions_lost_by_next_flip"])
        )
    if actions.get("sealed_or_unavailable"):
        sealed = ", ".join(a["test_id"] for a in actions["sealed_or_unavailable"])
        print(f"已封/不可用动作: {sealed}")

    rec = st["recommended_next"]
    print(f"讲解员建议: [{rec['type']}] {rec['instruction']}")
    if st["history"]:
        last = st["history"][-1]
        print(f"最近记录: {json.dumps(last, ensure_ascii=False)}")


def cmd_generate(args: argparse.Namespace) -> int:
    paths = generate_all(args.out)
    for p in paths:
        print(f"wrote {p}")
    print(f"generated {len(paths)} reproducible sample inputs")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    result = run_pipeline(args.input, args.out)
    ev = result["evaluation"]["exact_bayesian"]
    ho = result["validation"]
    emp = result["evaluation"]["synthetic_new_specimens"]
    print(f"输入: {args.input}")
    print(f"交付目录: {args.out}")
    print(
        "学习取向分布迭代: "
        f"{result['learned_model']['learning']['iterations']} "
        f"(converged={result['learned_model']['learning']['converged']})"
    )
    print(
        "留出读数核对: "
        f"n={ho['checked_readings']}, Brier={ho['mean_brier']}, "
        f"log_loss={ho['mean_log_loss']}, accuracy={ho['reading_accuracy']}"
    )
    print(
        "最优工作台: "
        f"recognition={ev['best_recognition_probability']:.4f}, "
        f"expected_cost={ev['minimum_expected_cost_at_best_rate']:.4f}, "
        f"expected_correct={ev['expected_correct_recognition_probability']:.4f}"
    )
    if emp["n"]:
        print(
            "合成新矿片: "
            f"n={emp['n']}, signed_rate={emp['signed_rate']}, "
            f"correct_over_all={emp['correct_rate_over_all']}, "
            f"mean_spent={emp['mean_spent']}"
        )
    print(f"resolved input for registration: {result['resolved_input']}")
    return 0


def _load_registration_input(args: argparse.Namespace) -> Dict[str, Any]:
    data = load_json(args.input)
    weights = None
    if args.model:
        model_doc = load_json(args.model)
        weights = model_doc["orientation_weights"]
    elif "learned_orientation_weights" in data:
        weights = data["learned_orientation_weights"]
    return data if weights is None else {**data, "learned_orientation_weights": weights}


def cmd_register(args: argparse.Namespace) -> int:
    data = _load_registration_input(args)
    weights = data.get("learned_orientation_weights")
    truth = None
    if args.specimen_id:
        for sp in data.get("new_specimens", []):
            if str(sp.get("id")) == args.specimen_id:
                truth = {"material": sp["material"], "orientation": sp["orientation"]}
                break
        if truth is None:
            raise SystemExit(f"specimen id {args.specimen_id!r} not found in input")
    case = create_case(args.state_dir, args.id, data, weights, truth=truth)
    st = case_status(args.state_dir, args.id)
    print(f"registered {args.id} -> {Path(args.state_dir) / 'cases' / (args.id + '.json')}")
    _render_status(st)
    return 0


def cmd_observe(args: argparse.Namespace) -> int:
    case_observe(args.state_dir, args.id, args.action, args.result, auto_checkout=args.checkout_if_ready)
    st = case_status(args.state_dir, args.id)
    if args.json:
        _print_json(st)
    else:
        _render_status(st)
    return 0


def cmd_flip(args: argparse.Namespace) -> int:
    case_flip(args.state_dir, args.id)
    st = case_status(args.state_dir, args.id)
    if args.json:
        _print_json(st)
    else:
        _render_status(st)
    return 0


def cmd_checkout(args: argparse.Namespace) -> int:
    case_checkout(args.state_dir, args.id, args.material)
    st = case_status(args.state_dir, args.id)
    if args.json:
        _print_json(st)
    else:
        _render_status(st)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    st = case_status(args.state_dir, args.id)
    if args.json:
        _print_json(st)
    else:
        _render_status(st)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.state_dir) / "cases"
    if not root.exists():
        print("(no cases)")
        return 0
    for p in sorted(root.glob("*.json")):
        case = load_json(p)
        print(f"{case['case_id']}\t{case['status']}\tspent={case.get('spent', 0)}")
    return 0


def cmd_auto_specimen(args: argparse.Namespace) -> int:
    """Register a generated answer-key specimen and enter its actual readings."""
    from .core import validate_and_build
    from .planner import WorkbenchPlanner

    data = load_json(args.input)
    weights = None
    if args.model:
        weights = load_json(args.model)["orientation_weights"]
    elif "learned_orientation_weights" in data:
        weights = data["learned_orientation_weights"]
    sp = next((x for x in data.get("new_specimens", []) if x.get("id") == args.specimen_id), None)
    if sp is None:
        raise SystemExit(f"specimen {args.specimen_id!r} not found")
    create_case(
        args.state_dir,
        args.case_id,
        data if weights is None else {**data, "learned_orientation_weights": weights},
        weights,
        truth={"material": sp["material"], "orientation": sp["orientation"]},
    )

    # Drive only through the same persistent command API used by a guide.
    for _ in range(20):
        st = case_status(args.state_dir, args.case_id)
        rec_type = st["recommended_next"]["type"]
        if st["judgment"]["state"] == "signed_out":
            break
        if rec_type == "checkout":
            case_checkout(args.state_dir, args.case_id)
            continue
        if rec_type == "flip":
            case_flip(args.state_dir, args.case_id)
            continue
        if rec_type == "run_test":
            tid = st["recommended_next"]["test_id"]
            result = sp.get("actual_readings", {}).get(tid)
            if result is None:
                model = validate_and_build(data, weights)
                h = next(
                    h
                    for h in model.hypotheses
                    if h.material == sp["material"] and h.orientation == sp["orientation"]
                )
                result = h.outcomes[model.active_test_index[tid]]
            case_observe(args.state_dir, args.case_id, tid, result)
            continue
        break

    st = case_status(args.state_dir, args.case_id)
    if args.json:
        _print_json(st)
    else:
        _render_status(st)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import run_selftest

    ok = run_selftest(verbose=not args.quiet)
    return 0 if ok else 1


def cmd_scenario_report(args: argparse.Namespace) -> int:
    from .selftest import scenario_report

    report = scenario_report()
    if args.out:
        dump_json(args.out, report)
        print(f"wrote {args.out}")
    if not args.quiet:
        _print_json(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mineral-bench",
        description="可复跑的博物馆矿片光学识别工作台（标准库 Python）",
    )
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="生成全部小型示例输入")
    g.add_argument("--out", default="examples")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("all", help="学习、留出核对、求最优策略并评估")
    a.add_argument("--input", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_all)

    r = sub.add_parser("register", help="按片号登记一片新矿片")
    r.add_argument("--id", required=True)
    r.add_argument("--input", required=True)
    r.add_argument("--model")
    r.add_argument("--specimen-id", help="可选：从生成输入载入模拟真值")
    r.add_argument("--state-dir", default="runs/booth")
    r.set_defaults(func=cmd_register)

    o = sub.add_parser("observe", help="录入一项试验的明/暗结果")
    o.add_argument("--id", required=True)
    o.add_argument("--action", required=True)
    o.add_argument("--result", required=True, help="bright/1/明 或 dark/0/暗")
    o.add_argument("--state-dir", default="runs/booth")
    o.add_argument("--checkout-if-ready", action="store_true")
    o.add_argument("--json", action="store_true")
    o.set_defaults(func=cmd_observe)

    f = sub.add_parser("flip", help="翻一次面并更新可用动作")
    f.add_argument("--id", required=True)
    f.add_argument("--state-dir", default="runs/booth")
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_flip)

    c = sub.add_parser("checkout", help="达到 0.90 阈值后签出材质")
    c.add_argument("--id", required=True)
    c.add_argument("--material")
    c.add_argument("--state-dir", default="runs/booth")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_checkout)

    s = sub.add_parser("status", help="按片号查花费、剩余动作和当前判断")
    s.add_argument("--id", required=True)
    s.add_argument("--state-dir", default="runs/booth")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    l = sub.add_parser("list", help="列出所有在办片号")
    l.add_argument("--state-dir", default="runs/booth")
    l.set_defaults(func=cmd_list)

    au = sub.add_parser("auto-specimen", help="用生成答案键自动走一遍真实持久化流程")
    au.add_argument("--case-id", required=True)
    au.add_argument("--specimen-id", required=True)
    au.add_argument("--input", required=True)
    au.add_argument("--model")
    au.add_argument("--state-dir", default="runs/booth")
    au.add_argument("--json", action="store_true")
    au.set_defaults(func=cmd_auto_specimen)

    t = sub.add_parser("selftest", help="运行内置断言和端到端检查")
    t.add_argument("--quiet", action="store_true")
    t.set_defaults(func=cmd_selftest)

    sr = sub.add_parser("scenario-report", help="输出题目指定情形的精确概率与预算曲线")
    sr.add_argument("--out", default="deliverables/scenario_checks.json")
    sr.add_argument("--quiet", action="store_true")
    sr.set_defaults(func=cmd_scenario_report)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
