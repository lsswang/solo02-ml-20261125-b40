"""The tiny three-word enumeration specified in the problem.

These checks are intentionally independent of the statistical corpus: they
verify the cost machine itself before it is used with learned probabilities.
"""

from __future__ import annotations

from typing import Any

from .board import BoardSpec, CostConfig, Position, action_cost, enumerate_layouts


def _expected_for_distribution(layout, distribution, start_page, costs):
    rows = []
    expected = 0.0
    for word, probability in distribution.items():
        detail = action_cost(layout, word, start_page, costs, chosen_slot=0)
        weighted = probability * detail["total_cost"]
        expected += weighted
        rows.append({"word": word, "probability": probability, "cost": detail, "weighted_cost": weighted})
    return expected, rows


def enumerate_distribution_scenario(scenario: dict[str, Any], costs: CostConfig) -> dict[str, Any]:
    """Enumerate placements for an input-supplied one-word distribution."""
    distribution = {str(w): float(p) for w, p in scenario["distribution"].items()}
    locked_word = scenario.get("locked_word", "甲")
    words: list[str] = []
    if locked_word:
        words.append(str(locked_word))
    for word in distribution:
        if word not in words:
            words.append(word)

    fixed: dict[str, Position] = {}
    familiar: set[str] = set()
    if locked_word:
        locked_page = int(scenario.get("locked_page", 1)) - 1
        locked_slot = int(scenario.get("locked_slot", 1)) - 1
        fixed[str(locked_word)] = Position(locked_page, locked_slot)
        familiar.add(str(locked_word))

    spec = BoardSpec(
        words=words,
        num_pages=int(scenario.get("num_pages", 2)),
        fixed=fixed,
        familiar=familiar,
    )
    layouts = enumerate_layouts(spec)
    start_page = int(scenario.get("start_page", 1)) - 1
    candidates = []
    for layout in layouts:
        expected, rows = _expected_for_distribution(layout, distribution, start_page, costs)
        candidates.append(
            {
                "expected_cost": expected,
                "positions": {
                    w: {"page": p.page + 1, "slot": p.slot + 1}
                    for w, p in layout.positions.items()
                },
                "weighted_actions": rows,
            }
        )
    candidates.sort(key=lambda c: (round(c["expected_cost"], 12), str(c["positions"])))
    result = {**scenario, "start_page_zero_based": start_page, "best": candidates[0], "all_candidates": candidates}
    expected_position = scenario.get("expected_position")
    if expected_position:
        word = expected_position["word"]
        actual = candidates[0]["positions"][word]
        result["expectation_met"] = actual == {
            "page": int(expected_position["page"]),
            "slot": int(expected_position["slot"]),
        }
    return result


def run_calibration(costs: CostConfig | None = None) -> dict[str, Any]:
    costs = costs or CostConfig(page_turn=3.0, select=1.0, wait=1.0, toggle=0.5)
    spec = BoardSpec(
        words=["甲", "乙", "丙"],
        num_pages=2,
        fixed={"甲": Position(0, 0)},
        familiar={"甲"},
    )
    layouts = enumerate_layouts(spec)

    scenarios = [
        {
            "id": "page1_yi_0.8_bing_0.2_no_preset",
            "start_page": 0,
            "distribution": {"乙": 0.8, "丙": 0.2},
            "expected_position": {"word": "乙", "page": 1, "slot": 2},
        },
        {
            "id": "page1_swapped_probabilities",
            "start_page": 0,
            "distribution": {"乙": 0.2, "丙": 0.8},
            "expected_position": {"word": "丙", "page": 1, "slot": 2},
        },
        {
            "id": "page2_start_original_probabilities",
            "start_page": 1,
            "distribution": {"乙": 0.8, "丙": 0.2},
            "expected_position": {"word": "乙", "page": 2, "slot": 1},
        },
    ]

    report_scenarios = []
    for scenario in scenarios:
        candidates = []
        for layout in layouts:
            expected, rows = _expected_for_distribution(
                layout, scenario["distribution"], scenario["start_page"], costs
            )
            candidates.append(
                {
                    "expected_cost": expected,
                    "layout": layout.as_dict(),
                    "positions": {
                        w: {"page": p.page + 1, "slot": p.slot + 1}
                        for w, p in layout.positions.items()
                    },
                    "weighted_actions": rows,
                }
            )
        candidates.sort(key=lambda c: (round(c["expected_cost"], 12), str(c["positions"])))
        best = candidates[0]
        expected_pos = scenario["expected_position"]
        actual = best["positions"][expected_pos["word"]]
        if actual != {"page": expected_pos["page"], "slot": expected_pos["slot"]}:
            raise AssertionError(
                f"校准失败 {scenario['id']}: 最优位置 {actual}, 期望 {expected_pos}"
            )
        report_scenarios.append({**scenario, "best": best, "all_candidates": candidates})

    # Explicit component checks.
    best_first = report_scenarios[0]["best"]["layout"]
    # Reconstruct winning layout object from positions via a fresh enumeration match.
    winning_layout = next(
        lay for lay in layouts
        if lay.position("乙") == Position(0, 1) and lay.position("丙") == Position(1, 0)
    )
    yi = action_cost(winning_layout, "乙", 0, costs, chosen_slot=0)
    bing_page2_first = action_cost(winning_layout, "丙", 0, costs, chosen_slot=0)
    assert yi["wait_cost"] == 1 and yi["select_cost"] == 1 and yi["page_turn_cost"] == 0
    assert yi["total_cost"] == 2
    assert bing_page2_first["page_steps_forward"] == 1
    assert bing_page2_first["page_turn_cost"] == 3
    assert bing_page2_first["total_cost"] == 4

    preset_yi = action_cost(winning_layout, "乙", 0, costs, chosen_slot=1)
    assert preset_yi["adjustment_cost"] == costs.toggle
    assert preset_yi["wait_cost"] == 0
    assert preset_yi["total_cost"] == costs.toggle + costs.select

    return {
        "cost_config": costs.__dict__,
        "statements_checked": [
            "甲锁定第一页首格",
            "切页一次三拍",
            "第一页开始且不预调：P(乙)=0.8 时乙在第一页另一格",
            "交换概率后：该格留给丙",
            "第二页开始且原概率：乙在第二页首格",
            "选择/等待/翻页/调整分项花费已逐词枚举",
        ],
        "scenarios": report_scenarios,
        "component_checks": {
            "yi_on_page1_second_from_page1": yi,
            "bing_on_page2_first_from_page1": bing_page2_first,
            "yi_with_preset_second_slot": preset_yi,
        },
    }
