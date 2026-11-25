"""End-to-end JSON input -> models, plans, replay traces, and Markdown report."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .board import BoardSpec, CostConfig, Layout, Position, parse_board, slot_preset_break_even_probability
from .calibration import enumerate_distribution_scenario, run_calibration
from .model import train_both
from .planner import (
    expected_schedule_cost,
    optimize_fixed_schedule,
    optimize_fixed_tasks,
    optimize_swappable_schedule,
    optimize_swappable_tasks,
    preset_decision,
)
from .replay import PresetRule, replay_sentence
from .session import SessionStore


def load_input(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def as_sentence_list(value: Any) -> list[list[str]]:
    if not value:
        return []
    if isinstance(value, dict):
        value = value.get("words", value.get("sentence", []))
    if isinstance(value, str):
        return [[value]]
    if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
        return [[str(x) for x in value]]
    out = []
    for seq in value:
        if isinstance(seq, dict):
            seq = seq.get("words", seq.get("sentence", []))
        out.append([str(x) for x in seq])
    return out


def normalize_tasks(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return independent runs and a two-scene retained-page schedule."""
    config = data.get("config", {}) or {}
    initial_page_1 = int(data.get("start_page", config.get("initial_page", 1)))
    start_pages = data.get("start_pages", {}) or {}
    tasks = data.get("tasks", {}) or {}
    runs: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    order = data.get("scene_order", ["home", "school"])

    for scene in order:
        sentences = as_sentence_list(tasks.get(scene, []))
        if not sentences:
            continue
        page_1 = int(start_pages.get(scene, start_pages.get("default", initial_page_1)))
        for i, sentence in enumerate(sentences):
            if not 1 <= len(sentence) <= 3:
                raise ValueError(f"{scene} 任务 {i + 1} 长度为 {len(sentence)}，题设要求 1 至 3 词")
            runs.append(
                {
                    "scene": scene,
                    "sentence": sentence,
                    "length": len(sentence),
                    "start_page": page_1 - 1,
                    "weight": 1.0,
                }
            )
        # The retained-page schedule uses the first task segment for each scene.
        schedule.append({"scene": scene, "length": len(sentences[0]), "entry_cost": 0.0})
    return runs, schedule


def planning_extra_words(data: dict[str, Any]) -> list[str]:
    words: list[str] = []
    for run in normalize_tasks(data)[0]:
        words.extend(run["sentence"])
    for replay in data.get("replays", []) or []:
        words.extend(str(w) for w in replay.get("sentence", []))
    # Preserve order/deduplicate.
    return list(dict.fromkeys(words))


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Layout):
        return obj.as_dict()
    if isinstance(obj, Position):
        return {"page": obj.page + 1, "slot": obj.slot + 1}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(jsonable(value), fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def write_enumeration_csv(path: Path, plan: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "expected_cost", "positions_page_slot"])
        for i, row in enumerate(plan.get("all_scores", []), start=1):
            pos = ";".join(f"{p['word']}@p{p['page']}s{p['slot']}" for p in row["positions"])
            writer.writerow([i, f"{row['expected_cost']:.12f}", pos])


def _layout_grid_text(layout: Layout) -> str:
    lines = []
    for page_index, row in enumerate(layout.grid(), start=1):
        first = row[0] if row[0] is not None else "（空）"
        second = row[1] if len(row) > 1 and row[1] is not None else "（空）"
        lines.append(f"第{page_index}页：[首格] {first} | [另一格] {second}")
    return "\n".join(lines)


def build_policy_rows_and_rule(
    layout: Layout, models: dict[str, Any], spec: BoardSpec, costs: CostConfig
) -> tuple[list[dict[str, Any]], PresetRule]:
    table: dict[tuple[str, str], int] = {}
    rows: list[dict[str, Any]] = []
    threshold = slot_preset_break_even_probability(costs)
    for scene, model in models.items():
        for word in spec.words:
            decision = preset_decision(layout, model, scene, word, costs, route="fixed_preset")
            table[(scene, word)] = decision.chosen_slot
            row = decision.as_dict()
            row["simple_two_slot_break_even_probability_for_nondefault"] = threshold
            rows.append(row)
    return rows, PresetRule(table, default_slot=costs.default_slot)


def run_replays_for_route(
    data: dict[str, Any],
    outdir: Path,
    route: str,
    layouts: dict[str, Layout],
    rule: PresetRule,
    costs: CostConfig,
    reset: bool = True,
    swap_pair: list[str] | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    state_path = outdir / ("sessions.json" if route == "fixed_preset" else "sessions-swappable.json")
    if reset and state_path.exists():
        state_path.unlink()
    store = SessionStore(state_path)
    traces: list[dict[str, Any]] = []
    for raw_request in data.get("replays", []) or []:
        request = dict(raw_request)
        if route == "swappable_cards" and request.get("switch_scene") and swap_pair:
            request["scene_swap"] = {
                "pair": swap_pair,
                "cost": costs.card_swap,
                "note": "进课堂时重新熟悉一对互换生词",
            }
        trace, store = replay_sentence(request, store, layouts, rule, costs, route=route)
        traces.append(trace)
    return traces, state_path


def render_report(result: dict[str, Any]) -> str:
    data = result["input"]
    costs = result["cost_config"]
    evaln = result["evaluation"]
    fixed = result["fixed_plan"]
    swap = result["swap_plan"]
    traces_fixed = result["replays_fixed"]
    traces_swap = result["replays_swappable"]
    lines: list[str] = []
    add = lines.append

    add("# 双开关词板训练与操作报告\n")
    add("## 1. 方案选择\n")
    add("主方案选择**路线乙：固定词卡 + 预置起扫格规则**。词位全程不动；规则只读取“当前场景 + 刚选完的词”，")
    add("不读取即将说出的目标词。路线甲（可换词卡，进课堂花 4 拍熟悉一对互换生词）也已完整枚举，用于比较差别。\n")
    add("动作口径：目标在起扫格时选择 1 拍；在另一格时多等 1 拍；向前循环翻页每次付设备给出的翻页费；")
    add(f"把起扫格从首格拨到另一格付调整费 {costs['toggle']:g} 拍。翻页后默认落在首格。\n")

    add("## 2. 训练数据与相容词序\n")
    for scene in ("home", "school"):
        model = result["models"][scene]
        add(f"### {scene}\n")
        add(f"- 词表：{ '、'.join(model['vocabulary']) }")
        add(f"- 平滑：add-α，α={model['alpha']:g}；模糊记录权重：{model['ambiguity_weighting']}")
        add(f"- 加权记录数：{model['weighted_records']:.3f}；加权词次：{model['weighted_tokens']:.3f}")
        groups = [e for e in model["compatible_order_enumerations"] if e["kind"] == "group"]
        add(f"- 分组记录枚举：{len(groups)} 组")
        for enum in groups:
            orders = "；".join("→".join(o) for o in enum["compatible_orders"])
            add(f"  - `{'、'.join(enum['record'])}` 相容词序：{orders}")
    add("")

    add("## 3. 完整留出句：转移模型 vs 直接词频猜测\n")
    add("| 场景 | 句数 | 词数 | 一阶转移 Top-1 | 词频 Top-1 | 转移困惑度 | 词频困惑度 |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    def fmt3(value):
        return "—" if value is None else f"{value:.3f}"
    for scene in ("home", "school"):
        e = evaln[scene]
        add(
            f"| {scene} | {e['sentence_count']} | {e['token_count']} | "
            f"{fmt3(e['transition_top1_accuracy'])} | {fmt3(e['frequency_top1_accuracy'])} | "
            f"{fmt3(e['transition_perplexity'])} | {fmt3(e['frequency_perplexity'])} |"
        )
    add("\n逐词预测见 `evaluation.json`。\n")

    q = result["query_example"]
    add("## 4. 后继概率查询示例\n")
    add(f"查询：scene=`{q['scene']}`，current=`{q['current'] or '⟦开头⟧'}`\n")
    add("| 后继词 | 平滑概率 |")
    add("|---|---:|")
    for row in q["rows"]:
        add(f"| {row['word']} | {row['probability']:.6f} |")
    add("\nCLI 可查询任意当前词：`python -m wordboard query --input inputs/sample.json --scene home --current 甲`。\n")

    add("## 5. 排卡枚举结果\n")
    add(f"共枚举 {fixed['candidate_count']} 种合法排卡。按两段输入任务的已选词前缀、"
        f"并对下一词使用模型期望，固定词卡最优期望成本：**{fixed['best_score']:.4f}**。")
    add(f"同一布局在纯模型句长滚动目标下的期望成本为 "
        f"**{result['rollout_schedule']['expected_total_cost']:.4f}**（见 `plan.json`）。\n")
    add("```text")
    add(_layout_grid_text(fixed["best_layout"]))
    add("```\n")
    add("预置规则（仅列每个场景/刚选词的决定；概率明细见 `plan.json`）：\n")
    add("| 场景 | 刚选完的词 | 当前页 | 起扫格 | 该上下文期望成本 |")
    add("|---|---|---:|---:|---:|")
    for row in result["policy_rows"]:
        add(
            f"| {row['scene']} | {row['context']} | {row['current_page']} | "
            f"第{row['chosen_start_slot']}格 | {row['expected_cost']:.4f} |"
        )
    add("")
    thr = slot_preset_break_even_probability(result["costs_obj"])
    add(f"在无翻页、两格分别为首格/另一格的简化情形中，直接把起扫格拨到另一格的盈亏平衡概率为 "
        f"`0.5 + toggle/(2*wait) = {thr:.3f}`；一般有翻页情形由逐目标枚举自动计算。\n")

    add("## 6. 路线甲对照：何时值得换卡\n")
    so = swap["school_option"]
    add(f"- 枚举方案数：{swap['candidate_count']}")
    add(f"- 家庭排卡：\n```text\n{_layout_grid_text(swap['home_layout'])}\n```")
    pair = so["swap_pair"]
    add(f"- 课堂是否换一对生词：**{pair if pair else '不换'}**；换卡熟悉成本：{so['swap_cost']:g} 拍")
    add(f"- 不换卡完成两段输入任务的期望成本：{swap['no_swap_full_task_cost']:.4f}")
    add(f"- 最优完整任务期望成本：{swap['best_total_cost']:.4f}")
    add(f"- 本次净收益：{swap['net_swap_benefit']:.4f}；现在值得换：**{swap['swap_worthwhile_now']}**")
    available = swap.get("best_available_pair")
    if available:
        pair_text = "、".join(available["swap_pair"])
        add(
            f"- 枚举中最有利的可换词对是 {pair_text}：课堂任务毛节省 "
            f"{available['gross_school_task_saving']:.4f} 拍，扣除一次 {costs['card_swap']:g} 拍后净 "
            f"{available['net_after_standard_swap_cost']:.4f} 拍。"
        )
    be = swap.get("break_even_school_entries")
    gross_for_break_even = swap.get("largest_available_gross_saving", 0.0)
    add(f"- 若每次进课堂都付 {costs['card_swap']:g} 拍，而最大毛节省为 "
        f"{gross_for_break_even:.4f} 拍，盈亏平衡进入次数："
        f"{'永不（毛节省不大于0）' if be is None else f'{be:.3f} 次（超过该值值得）'}。\n")
    add("```text")
    add(_layout_grid_text(so["school_layout"]))
    add("```\n")

    add("## 7. 题设微型校准\n")
    cal = result["calibration"]
    for scen in cal["scenarios"]:
        best_positions = scen["best"]["positions"]
        add(f"- {scen['id']}：最优期望成本 {scen['best']['expected_cost']:.4f}，位置 `{best_positions}`")
    add("- 分项核对：乙在第一页另一格：选择 1 + 等待 1 = 2 拍；丙在第二页首格：翻页 3 + 选择 1 = 4 拍。")
    add("- 预置到另一格时另计调整费；所有候选见 `calibration.json`。\n")
    if result.get("probability_scenarios"):
        add("### 输入中的直接概率场景\n")
        for scen in result["probability_scenarios"]:
            positions = scen["best"]["positions"]
            check = scen.get("expectation_met")
            check_text = "" if check is None else f"；期望位置核对={check}"
            add(f"- {scen.get('id', 'scenario')}：最优期望成本 "
                f"{scen['best']['expected_cost']:.4f}，位置 `{positions}`{check_text}")
        add("\n换入新的 `probability_scenarios` 后，结果写入 `probability_scenarios.json`。\n")

    add("## 8. 实际新句回放\n")
    for trace in traces_fixed:
        add(f"### 固定卡/预置头：{trace['session_id']} — {' '.join(trace['sentence'])}\n")
        add(f"实际总成本 **{trace['total_actual_cost']:.3f}**；翻页 **{trace['total_page_turns']}** 次；"
            f"等待 {trace['total_wait_beats']} 拍；结束页：第 {trace['final_page']} 页。\n")
        add("| 词序 | 词 | 原页 | 上文词 | 规则起扫格 | 翻页次数/费 | 等待拍/费 | 调整费 | 选择费 | 合计 | 后页 |")
        add("|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for s in trace["steps"]:
            ctx = s["previous_selected_word"] or "—"
            add(
                f"| {s['word_index']} | {s['word']} | {s['page_before']} | {ctx} | "
                f"{s['preset_rule_call']['chosen_slot']} | "
                f"{s['page_steps_forward']}/{s['page_turn_cost']:g} | "
                f"{s['wait_beats']}/{s['wait_cost']:g} | {s['adjustment_cost']:g} | "
                f"{s['select_cost']:g} | {s['total_cost']:g} | {s['target_page']} |"
            )
        add("")
    if traces_swap:
        add("### 路线甲实际回放总成本对照\n")
        add("| 会话 | 句子 | 固定卡总成本 | 可换卡总成本（含换卡） | 可换卡结束页 |")
        add("|---|---|---:|---:|---:|")
        for f, a in zip(traces_fixed, traces_swap):
            add(
                f"| {f['session_id']} | {' '.join(f['sentence'])} | {f['total_actual_cost']:.3f} | "
                f"{a['total_actual_cost']:.3f} | {a['final_page']} |"
            )
        add("\n完整动作轨迹见 `replay_fixed.json` 与 `replay_swappable.json`。\n")

    add("## 9. 复跑与换输入\n")
    add("```bash")
    add("python -m wordboard check")
    add("python -m wordboard all --input inputs/sample.json --outdir outputs/sample")
    add("python -m wordboard generate --seed 2026 --out inputs/generated.json")
    add("python -m wordboard all --input inputs/generated.json --outdir outputs/generated")
    add("python -m unittest discover -s tests -v")
    add("```")
    add("换入同 schema 的 JSON 后，重新执行 `all` 即得到新的模型、排卡、策略、回放和会话状态。")
    return "\n".join(lines) + "\n"


def run_pipeline(input_path: str | Path, outdir: str | Path, reset_state: bool = True) -> dict[str, Any]:
    input_path = Path(input_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_input(input_path)
    costs = CostConfig.from_data(data)
    runs, schedule = normalize_tasks(data)
    models, evaluation = train_both(data)

    extra_words = planning_extra_words(data)
    spec = parse_board(data, costs, extra_words)

    initial_page_1 = int(data.get("start_page", (data.get("config", {}) or {}).get("initial_page", 1)))
    initial_page = initial_page_1 - 1

    task_sequence = []
    for item in schedule:
        scene = item["scene"]
        run = next(r for r in runs if r["scene"] == scene)
        task_sequence.append({"scene": scene, "sentence": run["sentence"], "entry_cost": 0.0})
    if not task_sequence:
        raise ValueError("请在 tasks.home / tasks.school 中提供各为 1 至 3 词的任务")

    # Primary objective: the two supplied 1-3 word tasks.  The first word is
    # costed exactly; each later start-slot decision is scored from P(next|prev).
    fixed_plan = optimize_fixed_tasks(spec, models, task_sequence, costs, initial_page=initial_page)
    swap_plan = optimize_swappable_tasks(spec, models, task_sequence, costs, initial_page=initial_page)

    # Secondary fully-distributional rollout (lengths only, BOS + transitions).
    rollout_schedule = expected_schedule_cost(
        {scene: fixed_plan["best_layout"] for scene in ("home", "school")},
        models,
        schedule,
        costs,
        initial_page=initial_page,
        route="fixed_preset",
    )
    # Retain exhaustive model-rollout optimizers as machine-readable comparison.
    fixed_rollout_plan = optimize_fixed_schedule(spec, models, schedule, costs, initial_page=initial_page)
    swap_rollout_plan = optimize_swappable_schedule(spec, models, schedule, costs, initial_page=initial_page)

    policy_rows, rule = build_policy_rows_and_rule(fixed_plan["best_layout"], models, spec, costs)
    fixed_layouts = {scene: fixed_plan["best_layout"] for scene in ("home", "school")}
    swap_layouts = {
        "home": swap_plan["home_layout"],
        "school": swap_plan["school_option"]["school_layout"],
    }

    traces_fixed, fixed_state_path = run_replays_for_route(
        data, outdir, "fixed_preset", fixed_layouts, rule, costs, reset=reset_state
    )
    traces_swap, swap_state_path = run_replays_for_route(
        data,
        outdir,
        "swappable_cards",
        swap_layouts,
        PresetRule({}, costs.default_slot),
        costs,
        reset=reset_state,
        swap_pair=swap_plan["school_option"]["swap_pair"],
    )

    query_cfg = data.get("query", {}) or {}
    query_scene = query_cfg.get("scene", "home")
    query_current = query_cfg.get("current", query_cfg.get("current_word"))
    query_rows = models[query_scene].query(query_current, top_k=query_cfg.get("top_k"))
    query_example = {"scene": query_scene, "current": query_current, "rows": query_rows}

    calibration = run_calibration(costs)
    probability_scenarios = [
        enumerate_distribution_scenario(scenario, costs)
        for scenario in data.get("probability_scenarios", []) or []
    ]

    result = {
        "input_path": str(input_path),
        "input": data,
        "cost_config": costs.__dict__,
        "costs_obj": costs,
        "models": {scene: model.as_dict() for scene, model in models.items()},
        "evaluation": evaluation,
        "task_runs": runs,
        "schedule": schedule,
        "fixed_plan": fixed_plan,
        "swap_plan": swap_plan,
        "rollout_schedule": rollout_schedule,
        "fixed_rollout_plan": fixed_rollout_plan,
        "swap_rollout_plan": swap_rollout_plan,
        "policy_rows": policy_rows,
        "query_example": query_example,
        "calibration": calibration,
        "probability_scenarios": probability_scenarios,
        "replays_fixed": traces_fixed,
        "replays_swappable": traces_swap,
        "session_state_fixed_path": str(fixed_state_path),
        "session_state_swap_path": str(swap_state_path),
    }

    # Write machine-readable deliverables.
    write_json(outdir / "input.resolved.json", data)
    write_json(outdir / "models.json", {s: m.as_dict() for s, m in models.items()})
    write_json(outdir / "evaluation.json", evaluation)
    write_json(outdir / "plan.json", {
        "primary_objective": "input_task_prefixes_with_model_successor_expectation",
        "fixed_route": {k: v for k, v in fixed_plan.items() if k != "all_scores"},
        "swappable_route": swap_plan,
        "fully_distributional_rollout": {
            "chosen_fixed_layout_schedule": rollout_schedule,
            "fixed_rollout_best_score": fixed_rollout_plan["best_score"],
            "fixed_rollout_best_layout": fixed_rollout_plan["best_layout"],
            "swappable_rollout_best_score": swap_rollout_plan["best_total_cost"],
            "swappable_rollout_swap_pair": swap_rollout_plan["school_option"]["swap_pair"],
        },
        "policy_table": policy_rows,
        "task_runs": runs,
        "task_sequence": task_sequence,
        "schedule": schedule,
    })
    write_enumeration_csv(outdir / "layout_enumeration.csv", fixed_plan)
    write_json(outdir / "calibration.json", calibration)
    if probability_scenarios:
        write_json(outdir / "probability_scenarios.json", probability_scenarios)
    write_json(outdir / "query_example.json", query_example)
    write_json(outdir / "replay_fixed.json", traces_fixed)
    write_json(outdir / "replay_swappable.json", traces_swap)

    report = render_report(result)
    (outdir / "report.md").write_text(report, encoding="utf-8")

    # A concise machine summary is convenient for automated verifiers.
    summary = {
        "primary_objective": "input_task_prefixes_with_model_successor_expectation",
        "fixed_expected_task_cost": fixed_plan["best_score"],
        "swappable_expected_task_cost": swap_plan["best_total_cost"],
        "fixed_expected_schedule_cost": fixed_plan["best_score"],
        "swappable_expected_schedule_cost": swap_plan["best_total_cost"],
        "fixed_model_rollout_cost": rollout_schedule["expected_total_cost"],
        "fixed_layout": fixed_plan["best_layout"].as_dict(),
        "swap_pair": swap_plan["school_option"]["swap_pair"],
        "swap_worthwhile": swap_plan["swap_worthwhile_now"],
        "replay_fixed_total": sum(t["total_actual_cost"] for t in traces_fixed),
        "replay_swappable_total": sum(t["total_actual_cost"] for t in traces_swap),
        "fixed_replay_final_pages": {t["session_id"]: t["final_page"] for t in traces_fixed},
        "evaluation_accuracy": {
            s: {
                "transition": evaluation[s]["transition_top1_accuracy"],
                "frequency": evaluation[s]["frequency_top1_accuracy"],
            }
            for s in evaluation
        },
    }
    write_json(outdir / "summary.json", summary)
    return result
