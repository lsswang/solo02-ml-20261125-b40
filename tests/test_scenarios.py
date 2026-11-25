from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from relief_workshop.ioformats import build_problem, load_json
from relief_workshop.solver import brute_force_best, solve_map
from relief_workshop.workshop import WorkshopStore

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


def load_case(folder: str, name: str):
    edges = load_json(EX / folder / "edge_probabilities.json")
    order = load_json(EX / folder / name)
    return edges, order


class Canonical2x2Tests(unittest.TestCase):
    def test_original_forces_right_column_zero(self):
        edges, order = load_case("canonical_2x2", "order_original.json")
        problem, _ = build_problem(edges, order["gauge"], order["holes"])
        dp = solve_map(problem)
        bf = brute_force_best(problem)
        self.assertTrue(dp.feasible)
        self.assertTrue(bf.feasible)
        self.assertEqual(list(dp.heights), [1, 0, 1, 0])
        self.assertAlmostEqual(dp.score, bf.score, places=10)

    def test_lower_plus_gauge_forces_right_column_two(self):
        edges, order = load_case("canonical_2x2", "order_lower_plus.json")
        problem, _ = build_problem(edges, order["gauge"], order["holes"])
        dp = solve_map(problem)
        self.assertTrue(dp.feasible)
        self.assertEqual(list(dp.heights), [1, 2, 1, 2])

    def test_locked_right_height_two_makes_original_infeasible(self):
        edges, order = load_case("canonical_2x2", "order_lock_right2.json")
        report = WorkshopStore(tempfile.mkdtemp()).start(edges, order)
        self.assertFalse(report["feasible"])
        self.assertIn("no height assignment", report["solver_reason"])


class Canonical2x3Tests(unittest.TestCase):
    def setUp(self):
        self.edges, self.order = load_case("canonical_2x3", "order.json")

    def test_start_risk_and_selected_right_seam(self):
        with tempfile.TemporaryDirectory() as td:
            report = WorkshopStore(td).start(self.edges, self.order)
        self.assertTrue(report["feasible"])
        self.assertEqual(report["optimal_height_map"], [[1, 1, 1], [1, 1, 1]])
        risk = {
            (c["method"], c["seam_column"]): c["risk"]["expected_rejection_loss"]
            for c in report["candidates"]
        }
        self.assertAlmostEqual(risk[("whole", None)], 2.0)
        self.assertAlmostEqual(risk[("segment", 2)], 1.25)
        self.assertAlmostEqual(risk[("segment", 3)], 0.75)
        selected = report["selected"]
        self.assertEqual(selected["method"], "segment")
        self.assertEqual(selected["seam_column"], 3)
        self.assertEqual([(c["row"], c["col"], c["height"]) for c in selected["cuts"]],
                         [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1)])
        self.assertTrue(selected["extra_station_required"])
        self.assertTrue(selected["extra_station_required"])
        self.assertEqual(selected["legal_margin"]["optimal_tuples"], [[1, 1]])
        legal_tuples = [x["heights_top_to_bottom"]
                        for x in selected["legal_margin"]["tuples"]]
        self.assertIn([1, 1], legal_tuples)
        self.assertIn([2, 2], legal_tuples)

    def test_conflicting_mid_ticket_retains_cuts_then_right_ticket_cuts_seam(self):
        with tempfile.TemporaryDirectory() as td:
            store = WorkshopStore(td)
            store.start(self.edges, self.order)
            mid = load_json(EX / "canonical_2x3" / "supplement_mid2.json")
            rejected = store.supplement("WO-2X3-SEAM", mid)
            self.assertFalse(rejected["accepted"])
            self.assertEqual(rejected["reason"], "cut_conflict")
            report = store.cuts_report("WO-2X3-SEAM")
            self.assertEqual(report["status"], "seam_open")
            self.assertEqual(report["cut_count"], 4)

            right = load_json(EX / "canonical_2x3" / "supplement_right2.json")
            accepted = store.supplement("WO-2X3-SEAM", right)
            self.assertTrue(accepted["accepted"])
            self.assertEqual(accepted["height_map"], [[1, 1, 2], [1, 1, 2]])
            ticket_cuts = store.cuts_report("WO-2X3-RIGHT2")
            self.assertEqual([(c["row"], c["col"], c["height"])
                              for c in ticket_cuts["cuts_for_queried_ticket"]],
                             [(1, 3, 2), (2, 3, 2)])
            parent = store.cuts_report("WO-2X3-SEAM")
            self.assertEqual(parent["status"], "completed")
            self.assertEqual(parent["cut_count"], 6)


class SolverPropertyTests(unittest.TestCase):
    def test_dp_matches_enumeration_on_random_constrained_grids(self):
        import random
        from relief_workshop.grid import ALL_HEIGHT_MASK, Edge, GridProblem
        rng = random.Random(7)
        for case in range(30):
            rows, cols = 2, rng.choice([2, 3])
            edges = []
            for r in range(rows):
                for c in range(cols - 1):
                    u = r * cols + c
                    v = u + 1
                    logw = [-abs(rng.gauss(0, 1.5)) for _ in range(5)]
                    edges.append(Edge(f"H{r}{c}", u, v, logw))
            for r in range(rows - 1):
                for c in range(cols):
                    u = r * cols + c
                    v = u + cols
                    logw = [-abs(rng.gauss(0, 1.5)) for _ in range(5)]
                    edges.append(Edge(f"V{r}{c}", u, v, logw))
            masks = [ALL_HEIGHT_MASK] * (rows * cols)
            if case % 3 == 0:
                masks[0] = 1 << rng.randrange(4)
            problem = GridProblem(rows, cols, masks, edges)
            dp = solve_map(problem)
            bf = brute_force_best(problem)
            self.assertEqual(dp.feasible, bf.feasible)
            if dp.feasible:
                self.assertAlmostEqual(dp.score, bf.score, places=10)


if __name__ == "__main__":
    unittest.main()
