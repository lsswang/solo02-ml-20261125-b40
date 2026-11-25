"""Command line interface for training, planning, checking, and replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import run_calibration
from .generate import write_sample
from .model import train_both
from .board import CostConfig, parse_board
from .pipeline import load_input, normalize_tasks, planning_extra_words, run_pipeline, write_json
from .planner import optimize_fixed_tasks, optimize_swappable_tasks, preset_decision
from .replay import PresetRule, replay_sentence
from .session import SessionStore


def _print_calibration() -> None:
    report = run_calibration()
    print("微型枚举校准全部通过：")
    for statement in report["statements_checked"]:
        print(f"  ✓ {statement}")
    print("\n各场景最优候选：")
    for scenario in report["scenarios"]:
        best = scenario["best"]
        positions = ", ".join(f"{w}=p{p['page']}s{p['slot']}" for w, p in best["positions"].items())
        print(f"  - {scenario['id']}: E={best['expected_cost']:.4f}; {positions}")
    comps = report["component_checks"]
    print("\n分项花费：")
    for name, row in comps.items():
        print(
            f"  - {name}: 翻页{row['page_turn_cost']:g} + 等待{row['wait_cost']:g} "
            f"+ 调整{row['adjustment_cost']:g} + 选择{row['select_cost']:g} = {row['total_cost']:g}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="双开关词板训练与操作软件")
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("all", help="从一份 JSON 输入生成全部交付物")
    p_all.add_argument("--input", required=True)
    p_all.add_argument("--outdir", required=True)
    p_all.add_argument("--keep-state", action="store_true", help="复用已有会话状态而不是重置")

    p_gen = sub.add_parser("generate", help="生成另一份同格式合成输入")
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("--seed", type=int, default=2026)

    p_query = sub.add_parser("query", help="查询场景化后继概率")
    p_query.add_argument("--input", required=True)
    p_query.add_argument("--scene", choices=["home", "school"], default="home")
    p_query.add_argument("--current", default=None, help="当前词；省略表示句首分布")
    p_query.add_argument("--top-k", type=int, default=None)

    p_replay = sub.add_parser("replay", help="回放一条命令行给出的新句")
    p_replay.add_argument("--input", required=True)
    p_replay.add_argument("--outdir", required=True)
    p_replay.add_argument("--session", required=True)
    p_replay.add_argument("--scene", choices=["home", "school"], default="home")
    p_replay.add_argument("--sentence", nargs="+", required=True)
    p_replay.add_argument("--start-page", type=int, default=1)
    p_replay.add_argument("--route", choices=["fixed_preset", "swappable_cards"], default="fixed_preset")
    p_replay.add_argument("--close-after", type=int, default=None)

    p_check = sub.add_parser("check", help="运行题设微型枚举校准")
    p_check.add_argument("--json-out", default=None)

    p_resume = sub.add_parser("resume", help="按会话号读取关闭后持久化的页位")
    p_resume.add_argument("--state", required=True, help="sessions.json 路径")
    p_resume.add_argument("--session", required=True)

    args = parser.parse_args(argv)

    if args.command == "generate":
        path = write_sample(args.out, args.seed)
        print(f"已生成合成输入: {path}")
        return 0

    if args.command == "check":
        report = run_calibration()
        _print_calibration()
        if args.json_out:
            write_json(Path(args.json_out), report)
            print(f"\n完整校准 JSON: {args.json_out}")
        return 0

    if args.command == "replay":
        data = load_input(args.input)
        costs = CostConfig.from_data(data)
        models, _ = train_both(data)
        runs, schedule = normalize_tasks(data)
        task_sequence = []
        for item in schedule:
            run = next((r for r in runs if r["scene"] == item["scene"]), None)
            if run:
                task_sequence.append({"scene": item["scene"], "sentence": run["sentence"]})
        if not task_sequence:
            task_sequence = [{"scene": args.scene, "sentence": args.sentence}]
        spec = parse_board(data, costs, planning_extra_words(data) + [str(w) for w in args.sentence])
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        initial_page = int(data.get("start_page", (data.get("config", {}) or {}).get("initial_page", 1))) - 1
        if args.route == "fixed_preset":
            plan = optimize_fixed_tasks(spec, models, task_sequence, costs, initial_page=initial_page)
            layout = plan["best_layout"]
            table = {}
            for scene, model in models.items():
                for word in spec.words:
                    decision = preset_decision(layout, model, scene, word, costs)
                    table[(scene, word)] = decision.chosen_slot
            rule = PresetRule(table, costs.default_slot)
            layouts = {scene: layout for scene in ("home", "school")}
            state_path = outdir / "sessions-cli-fixed.json"
            swap_pair = None
        else:
            plan = optimize_swappable_tasks(spec, models, task_sequence, costs, initial_page=initial_page)
            layouts = {
                "home": plan["home_layout"],
                "school": plan["school_option"]["school_layout"],
            }
            rule = PresetRule({}, costs.default_slot)
            state_path = outdir / "sessions-cli-swappable.json"
            swap_pair = plan["school_option"]["swap_pair"]

        request = {
            "session_id": args.session,
            "scene": args.scene,
            "start_page": args.start_page,
            "sentence": args.sentence,
        }
        if args.close_after is not None:
            request["close_after"] = args.close_after
        if args.route == "swappable_cards" and args.scene == "school" and swap_pair:
            request["switch_scene"] = True
            request["scene_swap"] = {"pair": swap_pair, "cost": costs.card_swap}

        store = SessionStore(state_path)
        trace, _store = replay_sentence(request, store, layouts, rule, costs, route=args.route)
        print(json.dumps(trace, ensure_ascii=False, indent=2))
        return 0

    if args.command == "query":
        data = load_input(args.input)
        models, _ = train_both(data)
        rows = models[args.scene].query(args.current, top_k=args.top_k)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    if args.command == "resume":
        store = SessionStore(args.state)
        session = store.get(args.session)
        print(json.dumps({
            "session_id": session["session_id"],
            "scene": session["scene"],
            "restored_page": session["page"],
            "last_word": session["last_word"],
            "closed": session["closed"],
            "event_count": len(session["events"]),
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "all":
        result = run_pipeline(args.input, args.outdir, reset_state=not args.keep_state)
        summary_path = Path(args.outdir) / "summary.json"
        print(json.dumps({
            "outdir": args.outdir,
            "report": str(Path(args.outdir) / "report.md"),
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        }, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"未知命令: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
