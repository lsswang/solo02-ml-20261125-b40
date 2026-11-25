import json
import tempfile
import unittest
from pathlib import Path

from guide_studio.bundle import load_bundle
from guide_studio.model import fit_referent_model
from guide_studio.pipeline import run_pipeline
from guide_studio.replay import replay_plates
from guide_studio.sample import generate_sample


ROOT = Path(__file__).resolve().parents[1]


class GuideStudioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.input = generate_sample(cls.base / "input", "baseline")
        cls.alt = generate_sample(cls.base / "input_west_lin", "west_lin")
        cls.out = cls.base / "out"
        cls.closed_out = cls.base / "closed"
        cls.alt_out = cls.base / "alt"
        cls.summary = run_pipeline(cls.input, cls.out)
        run_pipeline(cls.input, cls.closed_out, closed_stations=["west_gate"])
        cls.alt_summary = run_pipeline(cls.alt, cls.alt_out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_files_and_two_mishearing_records(self):
        for name in [
            "referent_model.json",
            "holdout_log_loss.json",
            "two_mishearing_records.json",
            "proofreading_report.csv",
            "optimization.json",
            "recording_manifest.csv",
            "run_summary.md",
        ]:
            self.assertTrue((self.out / name).exists(), name)
        two = json.loads((self.out / "two_mishearing_records.json").read_text(encoding="utf-8"))
        self.assertEqual(len(two), 2)
        self.assertTrue(all(r["chosen_person_id"] != r["intended_person_id"] for r in two))
        holdout = json.loads((self.out / "holdout_log_loss.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(holdout["mean_log_loss_nats"])
        self.assertEqual(set(holdout["per_language"]), {"zh", "en"})

    def test_complete_distributions_sum_to_one(self):
        bundle = load_bundle(self.input)
        model = fit_referent_model(bundle)
        for lang in ("zh", "en"):
            candidates = model["persons"][lang]
            for level in model["distributions"].values():
                for dist in level.values():
                    if set(dist) == set(candidates):
                        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_baseline_wall_shared_full_pen_east_short(self):
        for lang in ("zh", "en"):
            wall = json.loads((self.out / "plans" / f"wall_{lang}.json").read_text())
            pen = json.loads((self.out / "plans" / f"pen_{lang}.json").read_text())
            self.assertTrue(wall["feasible"])
            self.assertTrue(pen["feasible"])

            wall_portrait = [
                s for t in wall["traces"] for s in t["steps"] if s["station_id"] == "portrait"
            ]
            self.assertEqual(len({s["wording_id"] for s in wall_portrait}), 1)
            self.assertTrue(all("full" in s["wording_id"] for s in wall_portrait))

            pen_by_route = {
                t["route_id"]: [s for s in t["steps"] if s["station_id"] == "portrait"][0]
                for t in pen["traces"]
            }
            self.assertIn("she", pen_by_route["east"]["wording_id"])
            self.assertIn("full", pen_by_route["west"]["wording_id"])
            self.assertLessEqual(pen["recording_count"], pen["pen_capacity"])

    def test_closed_west_uses_short_on_wall_and_pen(self):
        for device in ("wall", "pen"):
            for lang in ("zh", "en"):
                plan = json.loads((self.closed_out / "plans" / f"{device}_{lang}.json").read_text())
                portrait = [s for t in plan["traces"] for s in t["steps"] if s["station_id"] == "portrait"]
                self.assertEqual(len(portrait), 1)
                self.assertIn("she", portrait[0]["wording_id"])
                self.assertTrue(portrait[0]["reference_approved"])

    def test_replacement_material_changes_shared_wall_script(self):
        for lang in ("zh", "en"):
            wall = json.loads((self.alt_out / "plans" / f"wall_{lang}.json").read_text())
            portrait = [s for t in wall["traces"] for s in t["steps"] if s["station_id"] == "portrait"]
            self.assertEqual(len({s["wording_id"] for s in portrait}), 1)
            self.assertIn("she", portrait[0]["wording_id"])
            self.assertTrue(all(s["reference_approved"] for s in portrait))

    def test_replay_uses_plate_memory(self):
        plan = json.loads((self.out / "plans" / "pen_zh.json").read_text(encoding="utf-8"))
        rep = replay_plates(plan, ["east_gate", "lin_hall", "silent_porch", "portrait"])
        self.assertEqual(len(rep["steps"]), 4)
        self.assertIn("她", rep["steps"][-1]["text"])
        self.assertEqual(rep["steps"][-1]["device_memory"]["second_previous_station"], "lin_hall")
        with self.assertRaises(KeyError):
            replay_plates(plan, ["portrait"])

    def test_seed_reproducibility(self):
        again = self.base / "again"
        s2 = run_pipeline(self.input, again, render_audio=False)
        self.assertEqual(self.summary["holdout"]["selected_two_record_ids"],
                         s2["holdout"]["selected_two_record_ids"])
        self.assertEqual(
            self.summary["plans"]["pen"]["zh"]["objective"],
            s2["plans"]["pen"]["zh"]["objective"],
        )


if __name__ == "__main__":
    unittest.main()
