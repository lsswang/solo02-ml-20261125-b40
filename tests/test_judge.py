from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from boulder_judge.archive import create_case, file_calibration_appeal
from boulder_judge.evaluate import evaluate_corpus, labelled_route_class
from boulder_judge.handcheck import run_hand_examples
from boulder_judge.inference import adjudicate, aggregate_route_probabilities
from boulder_judge.model import JointModel, train_model
from boulder_judge.synthetic import (
    default_route_card,
    generate_corpus,
    generate_false_top_revision,
    make_appeal_attempt,
    make_sealed_testimonies,
)
from boulder_judge.sealed import build_sealed_bundle, compare_testimonies, rejudge_sealed

ROOT = Path(__file__).resolve().parents[1]


class HandExampleTests(unittest.TestCase):
    def test_required_hand_probabilities(self) -> None:
        report = run_hand_examples()
        self.assertTrue(report["all_expected_values_reproduced"])
        by_name = {x["name"]: x for x in report["examples"]}
        self.assertAlmostEqual(by_name["half_HH_half_EE_one_frame_K1"]["p_top"], 0.5, places=12)
        self.assertAlmostEqual(by_name["always_exactly_one_hand_K1"]["p_top"], 0.0, places=12)
        self.assertAlmostEqual(by_name["empty_then_final_HH_K1"]["p_top"], 1.0, places=12)
        self.assertAlmostEqual(by_name["empty_then_final_HH_K2"]["p_top"], 0.0, places=12)

        # Both wrists still have marginal probability 1/2 in the first two probes.
        for name in ["half_HH_half_EE_one_frame_K1", "always_exactly_one_hand_K1"]:
            for frame in by_name[name]["hand_marginals"]:
                self.assertAlmostEqual(frame["p_left_hold"], 0.5, places=12)
                self.assertAlmostEqual(frame["p_right_hold"], 0.5, places=12)


class ModelAndDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = generate_corpus(seed=123, n_train=10, n_heldout=3)
        self.model = train_model(
            [c for c in self.corpus["clips"] if c["split"] == "train"],
            self.corpus["vertices"],
            self.corpus["reading_values"],
            alpha=0.35,
            model_version="test",
        )
        self.route = default_route_card()

    def test_probabilities_sum_to_one(self) -> None:
        clip = self.corpus["clips"][-1]
        agg = aggregate_route_probabilities(self.model, clip, self.route)
        self.assertAlmostEqual(sum(agg["probabilities"].values()), 1.0, places=10)
        decision = adjudicate(self.model, clip, self.route)
        self.assertAlmostEqual(
            sum(x["probability"] for x in decision["score_probabilities"]), 1.0, places=10
        )
        for fr in decision["frame_evidence"]:
            self.assertAlmostEqual(sum(fr["joint_state_posterior"].values()), 1.0, places=9)

    def test_deadline_blocks_late_stable_top(self) -> None:
        # Exact readings: NN, TT, TT.  With deadline frame 0, the two-frame
        # TOP run occurs entirely after the cutoff and must not count.
        states = [["NONE", "NONE"], ["TOP", "TOP"]]
        pairs = [[a, b] for a in ["NONE", "TOP"] for b in ["NONE", "TOP"]]
        obs_pairs = [[a, b] for a in ["NONE", "TOP", "OCCLUDED"] for b in ["NONE", "TOP", "OCCLUDED"]]
        emission = [[0.0] * len(obs_pairs) for _ in pairs]
        obs_index = {f"{a}|{b}": i for i, (a, b) in enumerate(obs_pairs)}
        for i, s in enumerate(pairs):
            emission[i][obs_index[f"{s[0]}|{s[1]}"]] = 1.0
        model = JointModel(
            vertices=["NONE", "TOP"],
            reading_values=["NONE", "TOP", "OCCLUDED"],
            states=pairs,
            reading_pairs=obs_pairs,
            initial=[1.0, 0.0, 0.0, 0.0],
            transition=[[0.1, 0.1, 0.1, 0.7]] * 4,
            emission=emission,
        )
        clip = {
            "clip_id": "late",
            "frames": [
                {"left_reading": "NONE", "right_reading": "NONE"},
                {"left_reading": "TOP", "right_reading": "TOP"},
                {"left_reading": "TOP", "right_reading": "TOP"},
            ],
        }
        route = dict(self.route)
        route.update(stable_frames=2, deadline_frame=0, zone_vertex="ZONE")
        agg = aggregate_route_probabilities(model, clip, route)
        self.assertLess(agg["probabilities"]["TOP"], 1e-9)

        route["deadline_frame"] = 2
        agg2 = aggregate_route_probabilities(model, clip, route)
        self.assertGreater(agg2["probabilities"]["TOP"], 0.999)

    def test_heldout_report_has_joint_table(self) -> None:
        report = evaluate_corpus(self.model, self.corpus, self.route)
        self.assertEqual(report["joint_distribution_check"]["table"] and True, True)
        self.assertLess(report["joint_distribution_check"]["total_variation"], 0.25)
        self.assertEqual(report["frame_count"], sum(len(c["frames"]) for c in self.corpus["clips"] if c["split"] == "heldout"))

    def test_oracle_route_labels_are_mutually_exclusive(self) -> None:
        for clip in self.corpus["clips"]:
            code = labelled_route_class(clip, self.route)
            self.assertIn(code, {"TOP", "ZONE", "NO_ZONE"})


class AppealArchiveTests(unittest.TestCase):
    def test_same_number_records_initial_and_revision(self) -> None:
        corpus = generate_corpus(seed=321, n_train=16, n_heldout=2)
        base_model = train_model(
            corpus["clips"], corpus["vertices"], corpus["reading_values"], 0.35, "v1"
        )
        revised_data = generate_false_top_revision(n_clips=10)
        revised_model = train_model(
            corpus["clips"] + revised_data["clips"],
            corpus["vertices"],
            corpus["reading_values"],
            0.35,
            "v2",
        )
        clip, _truth = make_appeal_attempt()
        route = default_route_card()
        initial = adjudicate(base_model, clip, route)

        with tempfile.TemporaryDirectory() as td:
            case_dir = create_case(td, "APP-TEST-1", clip, route, base_model, initial)
            dossier, record = file_calibration_appeal(
                case_dir, revised_model, "test paired calibration change"
            )
            self.assertEqual(dossier["appeal_number"], "APP-TEST-1")
            self.assertTrue(record["old_judgment_reproduction"]["matches_initial_judgment"])
            self.assertEqual(record["disposition"], "revised")
            self.assertEqual(record["prior_signed_class"], "TOP")
            self.assertEqual(record["new_signed_class"], "ZONE")

            # A second appeal on the same number must reproduce the ZONE decision too.
            dossier2, record2 = file_calibration_appeal(
                case_dir, revised_model, "repeat review with same calibration"
            )
            self.assertEqual(len(dossier2["appeals"]), 2)
            self.assertEqual(record2["disposition"], "upheld")
            self.assertTrue(record2["old_judgment_reproduction"]["matches_recorded_judgment"])
            self.assertTrue(
                record2["initial_judgment_reproduction"]["matches_initial_judgment"]
            )
            on_disk = json.loads((case_dir / "dossier.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["appeal_number"], "APP-TEST-1")
            self.assertEqual(len(on_disk["appeals"]), 2)
            self.assertTrue((case_dir / "DOSSIER.md").exists())


class SealedFactorTests(unittest.TestCase):
    def test_sealed_factors_reproduce_decision_and_score_testimonies(self) -> None:
        corpus = generate_corpus(seed=77, n_train=10, n_heldout=1)
        model = train_model(corpus["clips"], corpus["vertices"], corpus["reading_values"], 0.35)
        clip, _ = make_appeal_attempt()
        route = default_route_card()
        direct = aggregate_route_probabilities(model, clip, route)["probabilities"]
        bundle = build_sealed_bundle(model, clip, route)
        sealed = rejudge_sealed(bundle)["probabilities"]
        for code in direct:
            self.assertAlmostEqual(direct[code], sealed[code], places=10)
        self.assertNotIn("left_reading", json.dumps(bundle))

        a, b = make_sealed_testimonies()
        comp = compare_testimonies(bundle, a, b)
        self.assertEqual(len(comp["conflict_frames"]), 4)
        self.assertGreater(comp["log_bayes_factor_a_over_b"], 0)
        self.assertEqual(comp["testimony_a"]["route_class_if_testimony_exact"], "TOP")
        self.assertEqual(comp["testimony_b"]["route_class_if_testimony_exact"], "ZONE")


if __name__ == "__main__":
    unittest.main()
