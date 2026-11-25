from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wordboard.board import (
    BoardSpec,
    CostConfig,
    Position,
    action_cost,
    enumerate_layouts,
    page_steps,
)
from wordboard.calibration import run_calibration
from wordboard.generate import build_sample_input, write_sample
from wordboard.model import compatible_orders, train_both
from wordboard.pipeline import run_pipeline
from wordboard.replay import PresetRule
from wordboard.session import SessionStore


class ModelTests(unittest.TestCase):
    def test_group_record_has_all_distinct_orders(self):
        orders = compatible_orders(["甲", "乙", "丙"])
        self.assertEqual(len(orders), 6)
        self.assertIn(("丙", "乙", "甲"), orders)

    def test_uniform_ambiguity_split_and_smoothing(self):
        data = {
            "corpus": {"home": {"groups": [{"id": "g1", "words": ["甲", "乙"]}]}, "school": {}},
            "config": {"alpha": 1.0},
        }
        models, _ = train_both(data)
        home = models["home"]
        enum = home.as_dict()["compatible_order_enumerations"][0]
        self.assertEqual(len(enum["compatible_orders"]), 2)
        self.assertAlmostEqual(enum["weight_per_order"], 0.5)
        # 0.5 fractional transition count + alpha, context 0.5 + alpha*V(2).
        self.assertAlmostEqual(home.probability("甲", "乙"), 1.5 / 2.5)
        self.assertGreater(home.probability("甲", "乙"), home.probability("甲", "甲"))

    def test_scene_models_are_separate(self):
        data = {
            "corpus": {
                "home": {"sentences": [["甲", "乙"]] * 5},
                "school": {"sentences": [["甲", "丙"]] * 5},
            },
            "board": {"words": ["甲", "乙", "丙"]},
        }
        models, _ = train_both(data)
        self.assertGreater(models["home"].probability("甲", "乙"), models["home"].probability("甲", "丙"))
        self.assertGreater(models["school"].probability("甲", "丙"), models["school"].probability("甲", "乙"))


class BoardCostTests(unittest.TestCase):
    def setUp(self):
        self.costs = CostConfig(page_turn=3, select=1, wait=1, toggle=0.5)
        self.spec = BoardSpec(
            words=["甲", "乙", "丙"],
            num_pages=2,
            fixed={"甲": Position(0, 0)},
        )
        self.layouts = enumerate_layouts(self.spec)
        self.layout = next(
            lay for lay in self.layouts
            if lay.position("乙") == Position(0, 1) and lay.position("丙") == Position(1, 0)
        )

    def test_forward_only_cyclic_pages(self):
        self.assertEqual(page_steps(0, 1, 2), 1)
        self.assertEqual(page_steps(1, 0, 2), 1)
        self.assertEqual(page_steps(2, 0, 3), 1)
        self.assertEqual(page_steps(0, 2, 3), 2)

    def test_cost_components(self):
        second = action_cost(self.layout, "乙", 0, self.costs, 0)
        self.assertEqual(second["page_steps_forward"], 0)
        self.assertEqual(second["wait_cost"], 1)
        self.assertEqual(second["select_cost"], 1)
        self.assertEqual(second["total_cost"], 2)

        other_page_first = action_cost(self.layout, "丙", 0, self.costs, 0)
        self.assertEqual(other_page_first["page_turn_cost"], 3)
        self.assertEqual(other_page_first["wait_cost"], 0)
        self.assertEqual(other_page_first["total_cost"], 4)

        preset = action_cost(self.layout, "乙", 0, self.costs, 1)
        self.assertEqual(preset["adjustment_cost"], 0.5)
        self.assertEqual(preset["wait_cost"], 0)
        self.assertEqual(preset["total_cost"], 1.5)

    def test_max_six_and_enumeration_size(self):
        # One fixed word + two free words + three available slots = P(3,2)=6.
        self.assertEqual(len(self.layouts), 6)
        seven = BoardSpec(words=list("一二三四五六七"), num_pages=4)
        with self.assertRaises(ValueError):
            seven.validate()


class CalibrationTests(unittest.TestCase):
    def test_problem_statements(self):
        report = run_calibration()
        first, swapped, second_page = report["scenarios"]
        self.assertEqual(first["best"]["positions"]["乙"], {"page": 1, "slot": 2})
        self.assertEqual(swapped["best"]["positions"]["丙"], {"page": 1, "slot": 2})
        self.assertEqual(second_page["best"]["positions"]["乙"], {"page": 2, "slot": 1})
        self.assertLess(first["best"]["expected_cost"], 2.4000001)


class SessionAndRuleTests(unittest.TestCase):
    def test_close_reopen_and_scene_retains_page(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sessions.json"
            store = SessionStore(path)
            store.create("s1", scene="home", start_page=1)
            store.record_word("s1", {"target_word": "丙", "target_page": 3})
            store.close("s1")

            reopened = SessionStore(path)
            self.assertEqual(reopened.get("s1")["page"], 3)
            reopened.switch_scene("s1", "school")
            self.assertEqual(reopened.get("s1")["page"], 3)
            again = SessionStore(path)
            self.assertEqual(again.get("s1")["scene"], "school")
            self.assertEqual(again.get("s1")["page"], 3)

    def test_preset_rule_sees_only_scene_and_previous_word(self):
        rule = PresetRule({("home", "甲"): 1})
        # Different hypothetical targets cannot change a rule decision because
        # the API has no target argument.
        first = rule.start_slot("home", "甲")
        second = rule.start_slot("home", "甲")
        self.assertEqual(first, 1)
        self.assertEqual(second, 1)
        self.assertEqual(rule.start_slot("home", None), 0)


class PipelineTests(unittest.TestCase):
    def test_sample_pipeline_outputs_and_replacement_seed(self):
        data = build_sample_input(seed=2026)
        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "input.json"
            outdir = Path(td) / "out"
            input_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            result = run_pipeline(input_path, outdir)
            for name in (
                "models.json",
                "evaluation.json",
                "plan.json",
                "report.md",
                "summary.json",
                "calibration.json",
                "replay_fixed.json",
                "replay_swappable.json",
                "sessions.json",
                "layout_enumeration.csv",
            ):
                self.assertTrue((outdir / name).exists(), name)
            summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("fixed_expected_schedule_cost", summary)
            self.assertGreater(summary["evaluation_accuracy"]["home"]["transition"],
                               summary["evaluation_accuracy"]["home"]["frequency"])
            # One-word calibration replay costs selection+wait exactly.
            one = next(t for t in result["replays_fixed"] if t["session_id"] == "one-word-from-page-1")
            self.assertAlmostEqual(one["total_actual_cost"], 2.0)
            self.assertTrue(all(s.get("expectation_met", True) for s in result["probability_scenarios"]))

            # A different seed is another valid same-format input.
            other_path = write_sample(Path(td) / "other.json", seed=7)
            other_result = run_pipeline(other_path, Path(td) / "other-out")
            self.assertTrue(other_result["fixed_plan"]["candidate_count"] > 0)

    def test_resume_cli_state_is_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sessions.json"
            store = SessionStore(path)
            store.create("abc", start_page=2)
            store.record_word("abc", {"target_word": "乙", "target_page": 2})
            store.close("abc")
            loaded = SessionStore(path).get("abc")
            self.assertEqual(loaded["page"], 2)
            self.assertTrue(loaded["closed"])


if __name__ == "__main__":
    unittest.main()
