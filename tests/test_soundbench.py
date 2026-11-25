import json
import os
import tempfile
import unittest

from soundbench.calibrate import calibrate_file
from soundbench.diagnose import run_diagnostics
from soundbench.ir import convolve_room, load_room
from soundbench.render import evaluate_rendered_constraints, render_program, render_robust_program
from soundbench.synth import SUM_DIFF_ROOM, generate_sample_tree
from soundbench.wavio import read_wav, write_wav


class LinearAndWavTests(unittest.TestCase):
    def test_wav_roundtrip_multichannel(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "a.wav")
            write_wav(path, 1000, [[0.0, 0.5, -0.5], [0.25, -1.0, 1.0]])
            sr, channels = read_wav(path)
            self.assertEqual(sr, 1000)
            self.assertEqual(len(channels), 2)
            self.assertAlmostEqual(channels[0][1], 0.5, places=4)
            self.assertAlmostEqual(channels[1][2], 1.0 - 1 / 32767, places=3)

    def test_sum_diff_identity(self):
        x1 = [0.2, -0.4, 0.8]
        x2 = [0.2, -0.4, 0.8]
        mics = convolve_room([x1, x2], SUM_DIFF_ROOM)
        for a, n, f in zip(x1, mics[0], mics[1]):
            self.assertAlmostEqual(n, 2 * a)
            self.assertAlmostEqual(f, 0.0)


class EndToEndSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.samples = os.path.join(cls.tmp.name, "samples")
        generate_sample_tree(cls.samples)
        cls.learned = os.path.join(cls.tmp.name, "learned.json")
        calibrate_file(os.path.join(cls.samples, "recordings", "cal_original.json"), cls.learned)
        cls.room, _ = load_room(cls.learned)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_learned_h0_is_sum_difference(self):
        h0 = [[sum(c for d, c in k if d == 0) for k in row] for row in self.room]
        self.assertAlmostEqual(h0[0][0], 1.0, places=2)
        self.assertAlmostEqual(h0[0][1], 1.0, places=2)
        self.assertAlmostEqual(h0[1][0], 1.0, places=2)
        self.assertAlmostEqual(h0[1][1], -1.0, places=2)

    def test_scene_program_passes_constraints(self):
        program = os.path.join(self.samples, "programs", "scene_change_program.json")
        rendered = render_program(program, self.room)
        report = evaluate_rendered_constraints(rendered)
        self.assertTrue(report["passes"], msg=json.dumps(report, indent=2))
        self.assertLessEqual(report["speaker_peak"], 0.9 + 1e-9)

    def test_curator_program_passes_constraints(self):
        program = os.path.join(self.samples, "programs", "curator_program.json")
        rendered = render_program(program, self.room)
        report = evaluate_rendered_constraints(rendered)
        self.assertTrue(report["passes"], msg=json.dumps(report, indent=2))
        self.assertEqual(len(report["cues"]), 4)

    def test_robust_track_passes_for_all_three_states(self):
        rooms = []
        for state in ("partition_open", "partition_half", "partition_closed"):
            learned = os.path.join(self.samples, "learned_tmp.json")
            recording = os.path.join(self.samples, "recordings", f"cal_{state}.json")
            calibrate_file(recording, learned)
            room, _ = load_room(learned)
            rooms.append(room)
        program = os.path.join(self.samples, "programs", "scene_change_program.json")
        _rendered, robust = render_robust_program(program, rooms)
        self.assertTrue(robust["passes"], msg=json.dumps(robust, indent=2))

    def test_diagnostics_expected_feasibility(self):
        out = os.path.join(self.tmp.name, "diagnostics")
        report = run_diagnostics(self.samples, out)
        self.assertTrue(report["sum_diff_original"]["meets_near_one_far_zero"])
        self.assertTrue(report["sum_diff_original"]["emits_two_half_level_channels"])
        self.assertFalse(report["far_also_sum_recalibration"]["isolable"])
        self.assertFalse(report["restored_sum_diff_then_speaker_2_unplugged"]["isolable"])


if __name__ == "__main__":
    unittest.main()
