"""Expected-cost planning and exhaustive small-board layout optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .board import (
    BoardSpec,
    CostConfig,
    Layout,
    action_cost,
    enumerate_layouts,
    slot_preset_break_even_probability,
)
from .model import SceneModel


@dataclass
class PresetDecision:
    scene: str
    context: str | None
    current_page: int
    chosen_slot: int
    expected_cost: float
    candidate_costs: dict[int, float]
    distribution: dict[str, float]
    on_board_mass: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "context": self.context,
            "current_page": self.current_page + 1,
            "chosen_start_slot": self.chosen_slot + 1,
            "expected_cost": self.expected_cost,
            "candidate_expected_costs": {
                f"slot_{slot + 1}": cost for slot, cost in self.candidate_costs.items()
            },
            "on_board_probability_mass": self.on_board_mass,
            "successor_distribution_used": self.distribution,
        }


def preset_decision(
    layout: Layout,
    model: SceneModel,
    scene: str,
    previous_word: str | None,
    costs: CostConfig,
    route: str = "fixed_preset",
) -> PresetDecision:
    """Choose a start slot using only scene + the already-selected word.

    The prospective target word is deliberately not an argument.
    """
    board_words = layout.spec.words
    pos = layout.position(previous_word)
    current_page = pos.page
    dist, mass = model.normalized_board_distribution(previous_word, board_words)

    if route in ("swappable_cards", "swap_cards", "card_swap"):
        allowed_slots = [costs.default_slot]
    else:
        allowed_slots = list(range(layout.slots_per_page))

    candidate_costs: dict[int, float] = {}
    for slot in allowed_slots:
        candidate_costs[slot] = sum(
            prob
            * action_cost(
                layout, word, current_page, costs, chosen_slot=slot
            )["total_cost"]
            for word, prob in dist.items()
        )

    # Exact ties go to the default slot; rounding suppresses float noise.
    chosen = min(allowed_slots, key=lambda s: (round(candidate_costs[s], 12), s))
    return PresetDecision(
        scene=scene,
        context=previous_word,
        current_page=current_page,
        chosen_slot=chosen,
        expected_cost=candidate_costs[chosen],
        candidate_costs=candidate_costs,
        distribution=dist,
        on_board_mass=mass,
    )


def first_word_expected_cost(
    layout: Layout,
    model: SceneModel,
    start_page: int,
    costs: CostConfig,
    start_page_distribution: dict[int, float] | None = None,
) -> tuple[float, dict[str, float], float]:
    dist, mass = model.normalized_board_distribution(None, layout.spec.words)
    page_dist = start_page_distribution or {int(start_page): 1.0}
    expected = 0.0
    for page, page_prob in page_dist.items():
        for word, word_prob in dist.items():
            expected += page_prob * word_prob * action_cost(
                layout, word, int(page), costs, chosen_slot=costs.default_slot
            )["total_cost"]
    return expected, dist, mass


def expected_task_cost(
    layout: Layout,
    model: SceneModel,
    scene: str,
    length: int,
    costs: CostConfig,
    route: str = "fixed_preset",
    start_page: int = 0,
    start_page_distribution: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Expected cost of ``length`` selections under the learned distribution."""
    if not 1 <= length <= 3:
        raise ValueError("题设任务长度为 1 至 3 词")
    first_cost, first_dist, first_mass = first_word_expected_cost(
        layout, model, start_page, costs, start_page_distribution
    )
    position_costs = [first_cost]
    masses = [first_mass]
    prev_dist = dict(first_dist)
    decisions: dict[str, dict[str, Any]] = {}

    for _step in range(1, length):
        step_cost = 0.0
        step_mass = 0.0
        next_dist: dict[str, float] = {}
        for previous, prev_prob in prev_dist.items():
            decision = preset_decision(layout, model, scene, previous, costs, route=route)
            step_cost += prev_prob * decision.expected_cost
            step_mass += prev_prob * decision.on_board_mass
            decisions[previous] = {
                "chosen_start_slot": decision.chosen_slot + 1,
                "expected_cost": decision.expected_cost,
                "candidate_expected_costs": {
                    f"slot_{slot + 1}": value for slot, value in decision.candidate_costs.items()
                },
            }
            for nxt, prob in decision.distribution.items():
                next_dist[nxt] = next_dist.get(nxt, 0.0) + prev_prob * prob
        position_costs.append(step_cost)
        masses.append(step_mass)
        total_mass = sum(next_dist.values())
        if total_mass:
            next_dist = {w: p / total_mass for w, p in next_dist.items()}
        prev_dist = next_dist

    return {
        "scene": scene,
        "length": length,
        "expected_cost": sum(position_costs),
        "position_expected_costs": position_costs,
        "mean_on_board_mass": sum(masses) / len(masses),
        "last_word_distribution": prev_dist,
        "decisions": decisions,
    }


def page_distribution_from_words(layout: Layout, word_dist: dict[str, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for word, prob in word_dist.items():
        page = layout.position(word).page
        out[page] = out.get(page, 0.0) + prob
    return out


def expected_schedule_cost(
    layouts: dict[str, Layout],
    models: dict[str, SceneModel],
    schedule: list[dict[str, Any]],
    costs: CostConfig,
    initial_page: int = 0,
    route: str = "fixed_preset",
) -> dict[str, Any]:
    """Run scene tasks in order while carrying the page number across scenes."""
    page_dist = {int(initial_page): 1.0}
    total = 0.0
    segments: list[dict[str, Any]] = []
    for item in schedule:
        scene = item["scene"]
        length = int(item["length"])
        entry_cost = float(item.get("entry_cost", 0.0))
        layout = layouts[scene]
        report = expected_task_cost(
            layout,
            models[scene],
            scene,
            length,
            costs,
            route=item.get("route", route),
            start_page_distribution=page_dist,
        )
        segment_total = entry_cost + report["expected_cost"]
        total += segment_total
        new_page_dist = page_distribution_from_words(layout, report["last_word_distribution"])
        segments.append(
            {
                "scene": scene,
                "length": length,
                "entry_cost": entry_cost,
                "expected_task_cost": report["expected_cost"],
                "segment_total": segment_total,
                "incoming_page_distribution": {p + 1: v for p, v in page_dist.items()},
                "outgoing_page_distribution": {p + 1: v for p, v in new_page_dist.items()},
                "position_expected_costs": report["position_expected_costs"],
            }
        )
        page_dist = new_page_dist
    return {"expected_total_cost": total, "segments": segments}


def _score_runs(
    layout: Layout,
    models: dict[str, SceneModel],
    runs: list[dict[str, Any]],
    costs: CostConfig,
    route: str,
) -> tuple[float, list[dict[str, Any]]]:
    total = 0.0
    rows = []
    for run in runs:
        scene = run["scene"]
        weight = float(run.get("weight", 1.0))
        report = expected_task_cost(
            layout,
            models[scene],
            scene,
            int(run["length"]),
            costs,
            route=route,
            start_page=int(run.get("start_page", 0)),
        )
        weighted = weight * report["expected_cost"]
        total += weighted
        rows.append(
            {
                "scene": scene,
                "length": int(run["length"]),
                "start_page": int(run.get("start_page", 0)) + 1,
                "weight": weight,
                "expected_cost": report["expected_cost"],
                "weighted_expected_cost": weighted,
            }
        )
    return total, rows


def optimize_fixed_layout(
    spec: BoardSpec,
    models: dict[str, SceneModel],
    runs: list[dict[str, Any]],
    costs: CostConfig,
) -> dict[str, Any]:
    """Route B: one fixed layout for both scenes plus contextual start slots."""
    results = []
    for layout in enumerate_layouts(spec):
        score, run_rows = _score_runs(layout, models, runs, costs, route="fixed_preset")
        results.append((score, layout, run_rows))
    results.sort(key=lambda item: (round(item[0], 12), item[1].signature()))
    best_score, best_layout, best_runs = results[0]
    return {
        "route": "fixed_preset",
        "candidate_count": len(results),
        "best_layout": best_layout,
        "best_score": best_score,
        "best_run_costs": best_runs,
        "rankings": [
            {
                "rank": i + 1,
                "expected_cost": score,
                "layout": layout.as_dict(),
            }
            for i, (score, layout, _rows) in enumerate(results[: min(20, len(results))])
        ],
        "all_scores": [
            {
                "expected_cost": score,
                "positions": [
                    {"word": w, "page": p.page + 1, "slot": p.slot + 1}
                    for w, p in sorted(layout.positions.items(), key=lambda kv: (kv[1].page, kv[1].slot))
                ],
            }
            for score, layout, _rows in results
        ],
    }


def optimize_swappable_layout(
    spec: BoardSpec,
    models: dict[str, SceneModel],
    home_runs: list[dict[str, Any]],
    school_runs: list[dict[str, Any]],
    costs: CostConfig,
    school_entries: int = 1,
) -> dict[str, Any]:
    """Route A using independent per-scene starting pages."""
    swap_words = sorted(spec.swappable - set(spec.fixed))
    pairs = [(a, b) for i, a in enumerate(swap_words) for b in swap_words[i + 1 :]]
    candidates = []

    for home_layout in enumerate_layouts(spec):
        home_score, home_rows = _score_runs(
            home_layout, models, home_runs, costs, route="swappable_cards"
        )

        school_options = []
        same_score, same_rows = _score_runs(
            home_layout, models, school_runs, costs, route="swappable_cards"
        )
        school_options.append(
            {
                "swap_pair": None,
                "school_layout": home_layout,
                "school_task_cost": same_score,
                "swap_cost": 0.0,
                "school_total": same_score,
                "school_rows": same_rows,
            }
        )
        for a, b in pairs:
            school_layout = home_layout.swap(a, b)
            score, rows = _score_runs(
                school_layout, models, school_runs, costs, route="swappable_cards"
            )
            swap_total = score + school_entries * costs.card_swap
            school_options.append(
                {
                    "swap_pair": [a, b],
                    "school_layout": school_layout,
                    "school_task_cost": score,
                    "swap_cost": school_entries * costs.card_swap,
                    "school_total": swap_total,
                    "school_rows": rows,
                }
            )
        best_school = min(school_options, key=lambda o: (round(o["school_total"], 12), str(o["swap_pair"])))
        total = home_score + best_school["school_total"]
        candidates.append((total, home_layout, home_rows, best_school, school_options))

    candidates.sort(key=lambda x: (round(x[0], 12), x[1].signature()))
    best_total, best_home, home_rows, best_school, _ = candidates[0]
    # Recompute the no-swap option at the winning home layout for a clear benefit report.
    no_swap_score, no_swap_rows = _score_runs(
        best_home, models, school_runs, costs, route="swappable_cards"
    )
    gross_saving = no_swap_score - best_school["school_task_cost"]
    net_benefit = gross_saving - best_school["swap_cost"]
    return {
        "route": "swappable_cards",
        "candidate_count": len(candidates) * (1 + len(pairs)),
        "eligible_swap_words": swap_words,
        "eligible_pairs": [list(p) for p in pairs],
        "best_total_cost": best_total,
        "home_layout": best_home,
        "home_run_costs": home_rows,
        "school_option": {
            "swap_pair": best_school["swap_pair"],
            "swap_cost": best_school["swap_cost"],
            "school_task_cost": best_school["school_task_cost"],
            "school_total": best_school["school_total"],
            "school_run_costs": best_school["school_rows"],
            "school_layout": best_school["school_layout"],
        },
        "no_swap_school_task_cost": no_swap_score,
        "gross_swap_saving_per_entry": gross_saving / max(1, school_entries),
        "net_swap_benefit_for_planned_entries": net_benefit,
        "swap_worthwhile_now": net_benefit > 1e-12,
        "swap_indifferent": abs(net_benefit) <= 1e-12,
        "break_even_school_entries": (
            costs.card_swap / gross_saving if gross_saving > 1e-12 else None
        ),
    }


def task_segment_expected_cost(
    layout: Layout,
    model: SceneModel,
    scene: str,
    sentence: list[str],
    costs: CostConfig,
    route: str = "fixed_preset",
    start_page: int = 0,
) -> dict[str, Any]:
    """Expected cost for one supplied task prefix.

    The first supplied word is costed exactly because the task asks for it.
    For every later slot, the rule sees only the already-selected prefix word;
    its slot choice is scored by the learned successor distribution rather than
    the actual next word in the replay sentence.
    """
    first = action_cost(layout, sentence[0], start_page, costs, chosen_slot=costs.default_slot)
    total = float(first["total_cost"])
    context_reports = []
    for i in range(1, len(sentence)):
        previous = sentence[i - 1]
        decision = preset_decision(layout, model, scene, previous, costs, route=route)
        total += decision.expected_cost
        context_reports.append(
            {
                "task_position": i + 1,
                "rule_context_previous_word": previous,
                "expected_cost": decision.expected_cost,
                "chosen_start_slot": decision.chosen_slot + 1,
                "candidate_expected_costs": {
                    f"slot_{slot + 1}": value for slot, value in decision.candidate_costs.items()
                },
                "model_distribution": decision.distribution,
            }
        )
    return {
        "scene": scene,
        "sentence_prefix_used_for_contexts": sentence,
        "length": len(sentence),
        "incoming_page": start_page + 1,
        "first_word_cost": first,
        "context_expected_costs": context_reports,
        "expected_cost": total,
        "final_page": layout.position(sentence[-1]).page,
    }


def evaluate_task_sequence(
    layouts: dict[str, Layout],
    models: dict[str, SceneModel],
    task_sequence: list[dict[str, Any]],
    costs: CostConfig,
    initial_page: int = 0,
    route: str = "fixed_preset",
) -> dict[str, Any]:
    """Evaluate ordered scene tasks while retaining the actual final page."""
    current_page = int(initial_page)
    total = 0.0
    segments = []
    for item in task_sequence:
        scene = item["scene"]
        sentence = [str(w) for w in item["sentence"]]
        entry_cost = float(item.get("entry_cost", 0.0))
        segment_route = item.get("route", route)
        report = task_segment_expected_cost(
            layouts[scene],
            models[scene],
            scene,
            sentence,
            costs,
            route=segment_route,
            start_page=current_page,
        )
        segment_total = entry_cost + report["expected_cost"]
        total += segment_total
        segments.append(
            {
                "scene": scene,
                "entry_cost": entry_cost,
                "incoming_page": current_page + 1,
                "expected_task_cost": report["expected_cost"],
                "segment_total": segment_total,
                "actual_prefix_final_page": report["final_page"] + 1,
                "report": report,
            }
        )
        current_page = report["final_page"]
    return {"expected_total_cost": total, "segments": segments}


def optimize_fixed_tasks(
    spec: BoardSpec,
    models: dict[str, SceneModel],
    task_sequence: list[dict[str, Any]],
    costs: CostConfig,
    initial_page: int = 0,
) -> dict[str, Any]:
    results = []
    scenes = [item["scene"] for item in task_sequence]
    for layout in enumerate_layouts(spec):
        layouts = {scene: layout for scene in scenes}
        report = evaluate_task_sequence(
            layouts, models, task_sequence, costs, initial_page=initial_page, route="fixed_preset"
        )
        results.append((report["expected_total_cost"], layout, report))
    results.sort(key=lambda x: (round(x[0], 12), x[1].signature()))
    best_score, best_layout, best_report = results[0]
    return {
        "route": "fixed_preset",
        "objective": "input_task_prefixes_with_model_successor_expectation",
        "candidate_count": len(results),
        "best_score": best_score,
        "best_layout": best_layout,
        "task_report": best_report,
        "all_scores": [
            {
                "expected_cost": score,
                "positions": [
                    {"word": w, "page": p.page + 1, "slot": p.slot + 1}
                    for w, p in sorted(layout.positions.items(), key=lambda kv: (kv[1].page, kv[1].slot))
                ],
            }
            for score, layout, _report in results
        ],
    }


def optimize_swappable_tasks(
    spec: BoardSpec,
    models: dict[str, SceneModel],
    task_sequence: list[dict[str, Any]],
    costs: CostConfig,
    initial_page: int = 0,
) -> dict[str, Any]:
    swap_words = sorted(spec.swappable - set(spec.fixed))
    pairs = [(a, b) for i, a in enumerate(swap_words) for b in swap_words[i + 1 :]]
    school_scene = next((item["scene"] for item in task_sequence[1:]), task_sequence[-1]["scene"])
    candidates = []

    for home_layout in enumerate_layouts(spec):
        options = []
        no_swap_layouts = {item["scene"]: home_layout for item in task_sequence}
        no_swap_seq = [{**item, "entry_cost": 0.0} for item in task_sequence]
        no_swap_report = evaluate_task_sequence(
            no_swap_layouts, models, no_swap_seq, costs, initial_page, route="swappable_cards"
        )
        options.append(
            {"swap_pair": None, "school_layout": home_layout, "swap_cost": 0.0,
             "report": no_swap_report, "total": no_swap_report["expected_total_cost"]}
        )
        for a, b in pairs:
            school_layout = home_layout.swap(a, b)
            layouts = {task_sequence[0]["scene"]: home_layout, school_scene: school_layout}
            for item in task_sequence:
                layouts.setdefault(item["scene"], school_layout)
            swapped_seq = []
            for item in task_sequence:
                swapped_seq.append({
                    **item,
                    "entry_cost": costs.card_swap if item["scene"] == school_scene else 0.0,
                })
            report = evaluate_task_sequence(
                layouts, models, swapped_seq, costs, initial_page, route="swappable_cards"
            )
            options.append(
                {"swap_pair": [a, b], "school_layout": school_layout,
                 "swap_cost": costs.card_swap, "report": report,
                 "total": report["expected_total_cost"]}
            )
        best_option = min(options, key=lambda o: (round(o["total"], 12), str(o["swap_pair"])))
        candidates.append((best_option["total"], home_layout, best_option, options))

    candidates.sort(key=lambda x: (round(x[0], 12), x[1].signature()))
    best_total, best_home, best_option, options = candidates[0]
    no_swap_option = options[0]
    pair_options = options[1:]
    best_pair = min(pair_options, key=lambda o: (round(o["total"] - o["swap_cost"], 12), str(o["swap_pair"]))) if pair_options else None
    school_index = next(i for i, s in enumerate(best_option["report"]["segments"]) if s["scene"] == school_scene)
    no_swap_school_cost = no_swap_option["report"]["segments"][school_index]["expected_task_cost"]
    best_pair_school_cost = best_pair["report"]["segments"][school_index]["expected_task_cost"] if best_pair else no_swap_school_cost
    gross = no_swap_school_cost - best_pair_school_cost
    full_benefit = no_swap_option["total"] - best_total
    return {
        "route": "swappable_cards",
        "objective": "input_task_prefixes_with_retained_page",
        "candidate_count": len(candidates) * (1 + len(pairs)),
        "eligible_pairs": [list(p) for p in pairs],
        "best_total_cost": best_total,
        "home_layout": best_home,
        "school_option": {
            "swap_pair": best_option["swap_pair"],
            "swap_cost": best_option["swap_cost"],
            "school_layout": best_option["school_layout"],
            "task_report": best_option["report"],
        },
        "best_available_pair": (
            {"swap_pair": best_pair["swap_pair"], "gross_school_task_saving": gross,
             "net_after_standard_swap_cost": gross - costs.card_swap,
             "school_layout": best_pair["school_layout"]}
            if best_pair else None
        ),
        "no_swap_full_task_cost": no_swap_option["total"],
        "net_swap_benefit": full_benefit,
        "swap_worthwhile_now": full_benefit > 1e-12 and best_option["swap_pair"] is not None,
        "swap_indifferent": abs(full_benefit) <= 1e-12,
        "largest_available_gross_saving": gross,
        "break_even_school_entries": costs.card_swap / gross if gross > 1e-12 else None,
        "task_report": best_option["report"],
    }


def optimize_fixed_schedule(
    spec: BoardSpec,
    models: dict[str, SceneModel],
    schedule: list[dict[str, Any]],
    costs: CostConfig,
    initial_page: int = 0,
) -> dict[str, Any]:
    """Optimize Route B when the page is carried through the scene schedule."""
    results = []
    scenes = [item["scene"] for item in schedule]
    for layout in enumerate_layouts(spec):
        layouts = {scene: layout for scene in scenes}
        report = expected_schedule_cost(
            layouts, models, schedule, costs, initial_page=initial_page, route="fixed_preset"
        )
        results.append((report["expected_total_cost"], layout, report))
    results.sort(key=lambda item: (round(item[0], 12), item[1].signature()))
    best_score, best_layout, best_report = results[0]
    return {
        "route": "fixed_preset",
        "objective": "retained_page_schedule",
        "candidate_count": len(results),
        "best_score": best_score,
        "best_layout": best_layout,
        "schedule_report": best_report,
        "rankings": [
            {"rank": i + 1, "expected_cost": score, "layout": layout.as_dict()}
            for i, (score, layout, _report) in enumerate(results[: min(20, len(results))])
        ],
        "all_scores": [
            {
                "expected_cost": score,
                "positions": [
                    {"word": w, "page": p.page + 1, "slot": p.slot + 1}
                    for w, p in sorted(layout.positions.items(), key=lambda kv: (kv[1].page, kv[1].slot))
                ],
            }
            for score, layout, _report in results
        ],
    }


def optimize_swappable_schedule(
    spec: BoardSpec,
    models: dict[str, SceneModel],
    schedule: list[dict[str, Any]],
    costs: CostConfig,
    initial_page: int = 0,
) -> dict[str, Any]:
    """Optimize Route A with at most one familiarization-priced pair swap."""
    swap_words = sorted(spec.swappable - set(spec.fixed))
    pairs = [(a, b) for i, a in enumerate(swap_words) for b in swap_words[i + 1 :]]
    school_scene = next((item["scene"] for item in schedule if item["scene"] != schedule[0]["scene"]), "school")
    all_candidates = []

    for home_layout in enumerate_layouts(spec):
        options = []

        no_swap_schedule = [{**item, "entry_cost": 0.0} for item in schedule]
        no_swap_report = expected_schedule_cost(
            {schedule[0]["scene"]: home_layout, school_scene: home_layout},
            models,
            no_swap_schedule,
            costs,
            initial_page=initial_page,
            route="swappable_cards",
        )
        options.append(
            {
                "swap_pair": None,
                "school_layout": home_layout,
                "swap_cost": 0.0,
                "report": no_swap_report,
                "total": no_swap_report["expected_total_cost"],
            }
        )

        for a, b in pairs:
            school_layout = home_layout.swap(a, b)
            swapped_schedule = []
            for item in schedule:
                entry = costs.card_swap if item["scene"] == school_scene else 0.0
                swapped_schedule.append({**item, "entry_cost": entry})
            report = expected_schedule_cost(
                {schedule[0]["scene"]: home_layout, school_scene: school_layout},
                models,
                swapped_schedule,
                costs,
                initial_page=initial_page,
                route="swappable_cards",
            )
            options.append(
                {
                    "swap_pair": [a, b],
                    "school_layout": school_layout,
                    "swap_cost": costs.card_swap,
                    "report": report,
                    "total": report["expected_total_cost"],
                }
            )

        best_option = min(options, key=lambda o: (round(o["total"], 12), str(o["swap_pair"])))
        all_candidates.append((best_option["total"], home_layout, best_option, options))

    all_candidates.sort(key=lambda x: (round(x[0], 12), x[1].signature()))
    best_total, best_home, best_option, winning_options = all_candidates[0]
    no_swap_option = winning_options[0]
    no_swap_total = no_swap_option["total"]
    school_index = next(
        i for i, seg in enumerate(no_swap_option["report"]["segments"]) if seg["scene"] == school_scene
    )
    pair_options = winning_options[1:]
    best_pair_by_gross = None
    best_pair_gross = 0.0
    if pair_options:
        def school_task_cost(option):
            return option["report"]["segments"][school_index]["expected_task_cost"]

        best_pair_by_gross = min(pair_options, key=lambda o: (round(school_task_cost(o), 12), str(o["swap_pair"])))
        best_pair_gross = school_task_cost(no_swap_option) - school_task_cost(best_pair_by_gross)

    selected_gross = (
        no_swap_option["report"]["segments"][school_index]["expected_task_cost"]
        - best_option["report"]["segments"][school_index]["expected_task_cost"]
    )
    # Compare full no-swap schedule total with chosen full total.
    full_saving_vs_no_swap = no_swap_total - best_total
    return {
        "route": "swappable_cards",
        "objective": "retained_page_schedule",
        "candidate_count": len(all_candidates) * (1 + len(pairs)),
        "eligible_swap_words": swap_words,
        "eligible_pairs": [list(p) for p in pairs],
        "best_total_cost": best_total,
        "home_layout": best_home,
        "school_option": {
            "swap_pair": best_option["swap_pair"],
            "swap_cost": best_option["swap_cost"],
            "school_layout": best_option["school_layout"],
            "schedule_report": best_option["report"],
        },
        "no_swap_full_schedule_cost": no_swap_total,
        "best_available_pair": (
            {
                "swap_pair": best_pair_by_gross["swap_pair"],
                "gross_school_task_saving": best_pair_gross,
                "net_after_standard_swap_cost": best_pair_gross - costs.card_swap,
                "school_layout": best_pair_by_gross["school_layout"],
            }
            if best_pair_by_gross is not None
            else None
        ),
        "gross_school_task_saving_per_entry": selected_gross,
        "largest_available_gross_saving": best_pair_gross,
        "net_swap_benefit": full_saving_vs_no_swap,
        "swap_worthwhile_now": full_saving_vs_no_swap > 1e-12 and best_option["swap_pair"] is not None,
        "swap_indifferent": abs(full_saving_vs_no_swap) <= 1e-12,
        "break_even_school_entries": (
            costs.card_swap / best_pair_gross if best_pair_gross > 1e-12 else None
        ),
        "schedule_report": best_option["report"],
    }
