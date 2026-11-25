"""End-to-end tests for the proofing program (pure stdlib, no pytest needed).

Run with:  python3 -m unittest discover -s tests -v
"""

import json
import os
import shutil
import tempfile
import unittest

from proofing import synthetic
from proofing.jobs import enumerate_offsets, load_job, job_from_dict, example_job_specs, RECIPES
from proofing.model import CHANNELS, dual_D, single_D, fit_all
from proofing.planner import plan_direct
from proofing.render import evaluate, render_sheet


def _write_dataset(d):
    truth = synthetic.build_truth()
    return synthetic.write_dataset(d, truth), truth


def _read(path):
    import csv
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for k in ("coverage", "cu", "cv"):
                if k in row:
                    row[k] = float(row[k])
            for k in ("d0", "d1", "d2"):
                row[k] = float(row[k])
            rows.append(row)
    return rows


class TestPhysics(unittest.TestCase):
    def test_full_coverage_from_targets(self):
        truth = synthetic.build_truth()
        p1 = truth["Paper1"]
        # forward (BA) = A pressing B = 0.90, reverse (AB) = 0.70, spot = 0.90
        for ch in CHANNELS:
            self.assertAlmostEqual(p1.dual_density("BA", 1, 1, ch), 0.90, places=4)
            self.assertAlmostEqual(p1.dual_density("AB", 1, 1, ch), 0.70, places=4)
            self.assertAlmostEqual(p1.single_density("S", 1, ch), 0.90, places=4)
        p2 = truth["Paper2"]
        for ch in CHANNELS:
            self.assertAlmostEqual(p2.dual_density("BA", 1, 1, ch), 0.75, places=3)
            self.assertAlmostEqual(p2.single_density("S", 1, ch), 0.90, places=3)

    def test_monotone_density(self):
        truth = synthetic.build_truth()["Paper1"]
        prev = -1
        for k in range(11):
            d = truth.single_density("A", k / 10, 1)
            self.assertGreaterEqual(d + 1e-9, prev)
            prev = d


class TestFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        paths, cls.truth = _write_dataset(cls.tmp)
        single = _read(paths["single"])
        dual = _read(paths["dual"])
        cls.responses, cls.reports = fit_all(single, dual)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_holdout_within_tolerance(self):
        # intermediate-coverage holdout must be learned to well inside 0.02
        for p, rep in self.reports.items():
            self.assertLess(rep["single"]["max_abs"], 0.01, p)
            self.assertLess(rep["dual"]["max_abs"], 0.01, p)

    def test_full_dual_recovered(self):
        for p, spec in synthetic.PAPER_SPEC.items():
            r = self.responses[p]
            for ch in CHANNELS:
                self.assertAlmostEqual(r.dual_density("BA", 1, 1, ch), spec["D_forward"], places=2)
                self.assertAlmostEqual(r.dual_density("AB", 1, 1, ch), spec["D_reverse"], places=2)

    def test_achromatic_at_full(self):
        r = self.responses["Paper1"]
        d = r.full_dual("BA")
        self.assertLess(max(d) - min(d), 0.01)

    def test_fit_is_deterministic(self):
        paths = [os.path.join(self.tmp, "ramps_single.csv"),
                 os.path.join(self.tmp, "ramps_dual.csv")]
        r2, rep2 = fit_all(_read(paths[0]), _read(paths[1]))
        for p in self.responses:
            for ch in CHANNELS:
                self.assertAlmostEqual(self.responses[p].s["A"][ch], r2[p].s["A"][ch], places=9)


class TestRoute1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        paths, _ = _write_dataset(cls.tmp)
        cls.resp, _ = fit_all(_read(paths["single"]), _read(paths["dual"]))
        cls.tmpjobs = tempfile.mkdtemp()
        from proofing.jobs import write_example_jobs
        write_example_jobs(cls.tmpjobs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        shutil.rmtree(cls.tmpjobs, ignore_errors=True)

    def _plan(self, name, paper="Paper1"):
        job = load_job(os.path.join(self.tmpjobs, name + ".json"))
        return job, plan_direct(job, self.resp[job.paper])

    def test_cheap_forward_when_clean(self):
        job, plan = self._plan("wide_block")
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.recipe, "forward")
        self.assertEqual(plan.cost, 200.0)

    def test_locked_reverse_forces_spot(self):
        job, plan = self._plan("wide_block_locked_reverse")
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.recipe, "spot")

    def test_new_paper_forces_spot(self):
        job, plan = self._plan("wide_block_paper2")
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.recipe, "spot")

    def test_edge_image_infeasible_direct(self):
        job, plan = self._plan("edge_image")
        self.assertFalse(plan.feasible)
        self.assertIn("offset", plan.evidence["reason"])

    def test_edge_image_zero_offset_feasible(self):
        job, plan = self._plan("edge_image_locked_zero")
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.recipe, "forward")

    def test_scrap_rule(self):
        job, plan = self._plan("scrap_demo")
        # triple is the only recipe and stacks three layers -> infeasible
        self.assertFalse(plan.feasible)
        self.assertIn("scrap", plan.evidence["reason"])
        fails = plan.evidence["failures"]["triple"]
        self.assertTrue(any(f["kind"] == "scrap_3_layers" for f in fails))

    def test_cheapest_tiebreak_prefers_forward(self):
        # when both forward and reverse would pass, forward is cheaper-or-equal
        # and must be chosen (canonical order: forward before reverse at same cost)
        job, plan = self._plan("wide_block")
        self.assertEqual(plan.recipe, "forward")

    def test_every_offset_checked_in_edge(self):
        job, plan = self._plan("edge_image")
        ev = plan.checks["forward"]
        self.assertEqual(len(ev.offsets), 9)  # 3x3 offsets for two plates

    def test_render_target_density_matches_model(self):
        job, plan = self._plan("wide_block")
        rec = job.recipe("forward")
        res = render_sheet(job, self.resp["Paper1"], rec, {"B": 0, "A": 0})
        for cell, (dens, tol) in job.targets.items():
            cr = res.target_cells[cell]
            for ch in CHANNELS:
                self.assertLess(abs(cr.density[ch] - dens[ch]), tol)

    def test_intermediate_ramp_uses_learned_response(self):
        # targets sit on the forward intermediate curve; only forward passes
        job, plan = self._plan("tone_ramp")
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.recipe, "forward")
        self.assertFalse(plan.checks["reverse"].passes)
        self.assertFalse(plan.checks["spot"].passes)
        # reverse/spot are genuinely out of tolerance (not a scrap artifact)
        self.assertGreater(plan.checks["reverse"].worst_relief, 0.02)
        self.assertGreater(plan.checks["spot"].worst_relief, 0.02)


class TestOffsets(unittest.TestCase):
    def test_zero_mode(self):
        spec = example_job_specs()["wide_block"]
        job = job_from_dict(spec)
        offs = enumerate_offsets(job, job.recipe("forward"))
        self.assertEqual(offs, [{"B": 0, "A": 0}])

    def test_all_mode_count(self):
        spec = example_job_specs()["edge_image"]
        job = job_from_dict(spec)
        offs = enumerate_offsets(job, job.recipe("forward"))
        self.assertEqual(len(offs), 9)

    def test_spot_single_plate_offsets(self):
        spec = example_job_specs()["edge_image"]
        job = job_from_dict(spec)
        offs = enumerate_offsets(job, job.recipe("spot"))
        self.assertEqual(len(offs), 3)


if __name__ == "__main__":
    unittest.main()
