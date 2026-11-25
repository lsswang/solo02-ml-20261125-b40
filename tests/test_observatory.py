from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from observatory.calibration import CalibrationInput, calibrate
from observatory.querying import evaluate_observation
from observatory.reservations import book_reservation, record_first
from observatory.scheduling import (
    Application,
    Ephemeris,
    filter_ephemerides,
    load_shifts,
    plan_fixed,
    plan_voucher,
)
from observatory.synthetic import generate_sample_tree


def eph(eid: str, star: str, pattern: str) -> Ephemeris:
    return Ephemeris(eid, star, tuple(int(c) for c in pattern))


class QueryMathTests(unittest.TestCase):
    def test_textbook_6_3_supports_only_yi(self) -> None:
        cal = {
            "responses": {
                "甲": {"ap1": 1.0, "ap2": 1.0},
                "乙": {"ap1": 1.0, "ap2": 0.5},
            },
            "response_intervals_95": {
                "甲": {"ap1": [1, 1], "ap2": [1, 1]},
                "乙": {"ap1": [1, 1], "ap2": [0.5, 0.5]},
            },
        }
        result = evaluate_observation(cal, [6, 3], 0.3, uncertainty="none")
        self.assertEqual(result["supported_stars"], ["乙"])
        lo, hi = result["stars"]["乙"]["depth_interval"]
        self.assertAlmostEqual(lo, 5.7)
        self.assertAlmostEqual(hi, 6.3)
        self.assertFalse(result["stars"]["甲"]["possible"])

    def test_coincident_columns_return_two_stars(self) -> None:
        cal = {
            "responses": {
                "甲": {"ap1": 1.0, "ap2": 0.52},
                "乙": {"ap1": 1.0, "ap2": 0.50},
            },
            "response_intervals_95": {
                "甲": {"ap1": [0.94, 1.06], "ap2": [0.44, 0.60]},
                "乙": {"ap1": [0.94, 1.06], "ap2": [0.43, 0.57]},
            },
        }
        result = evaluate_observation(cal, [6, 3], 0.3, uncertainty="se95")
        self.assertEqual(set(result["supported_stars"]), {"甲", "乙"})


class SchedulingTests(unittest.TestCase):
    def test_three_nights_four_strings(self) -> None:
        candidates = [
            eph("E000", "乙", "000"),
            eph("E010", "乙", "010"),
            eph("E100", "乙", "100"),
            eph("E101", "乙", "101"),
        ]
        shifts = load_shifts(Path("sample_inputs/shifts/shifts_three.csv"), 3)
        fixed = plan_fixed(candidates, shifts)
        self.assertEqual(fixed["worst_remaining"], 2)
        self.assertTrue(all(p["worst_remaining"] >= 2 for p in fixed["ranked_plans"]))

        voucher = plan_voucher(candidates, shifts, voucher_fee=0.0)
        self.assertEqual(voucher["first_night"], 1)
        self.assertEqual(voucher["voucher_listed_nights"], [2, 3])
        self.assertEqual(voucher["worst_remaining"], 1)
        self.assertEqual(voucher["policy_by_first_bit"]["0"]["redeem_night"], 2)
        self.assertEqual(voucher["policy_by_first_bit"]["1"]["redeem_night"], 3)

    def test_ab_calibration_and_open_night_effects(self) -> None:
        app = Application.load("sample_inputs/applications/order_ab.json")
        shifts4 = load_shifts("sample_inputs/shifts/shifts_ab_open4.csv", 8)
        shifts5 = load_shifts("sample_inputs/shifts/shifts_ab_open5.csv", 8)

        original, excluded = filter_ephemerides(app.ephemerides, ["乙"], app.known_clear_nights)
        self.assertEqual([e.id for e in original], ["B4", "B6"])
        self.assertEqual([x["ephemeris"]["id"] for x in excluded], ["A4"])
        p4 = plan_voucher(original, shifts4, app.voucher_fee)
        p5 = plan_voucher(original, shifts5, app.voucher_fee)
        self.assertEqual(p4["worst_remaining"], 1)
        self.assertEqual(p4["voucher_listed_nights"], [4])
        self.assertEqual(p5["worst_remaining"], 2)
        self.assertEqual(p5["voucher_listed_nights"], [5])

        p5_skip = plan_voucher(original, shifts5, app.voucher_fee, allow_skip_redemption=True)
        self.assertEqual(p5_skip["worst_remaining"], 2)
        self.assertIsNone(p5_skip["policy_by_first_bit"]["0"]["redeem_night"])
        self.assertEqual(p5_skip["worst_cost"], 1.0 + app.voucher_fee)

        recal, excluded = filter_ephemerides(app.ephemerides, ["甲", "乙"], app.known_clear_nights)
        self.assertEqual([e.id for e in recal], ["B4", "B6", "A4"])
        pr4 = plan_voucher(recal, shifts4, app.voucher_fee)
        self.assertEqual(pr4["worst_remaining"], 2)
        dim_group = pr4["policy_by_first_bit"]["0"]["groups_after_redeem"]
        dim_ids = next(g["ephemeris_ids"] for g in dim_group if g["observed_bits"] == [1])
        self.assertEqual(set(dim_ids), {"B4", "A4"})
        pr5 = plan_voucher(recal, shifts5, app.voucher_fee)
        self.assertEqual(pr5["worst_remaining"], 3)

    def test_max_eight_nights_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text(json.dumps({"order_id": "X", "horizon_nights": 9, "ephemerides": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                Application.load(p)

    def test_voucher_cannot_be_promised_twice(self) -> None:
        candidates = [eph("B4", "乙", "00010000"), eph("B6", "乙", "00000100")]
        shifts = load_shifts("sample_inputs/shifts/shifts_ab_open4.csv", 8)
        plan = plan_voucher(candidates, shifts)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            reg = td_path / "registry.json"
            book_reservation(
                order_id="O1",
                reservation_id="R1",
                voucher_id="V1",
                plan=plan,
                compatible_ephemerides=candidates,
                excluded=[],
                shifts_snapshot=[],
                query_summary=None,
                registry_path=reg,
                state_dir=td_path,
            )
            with self.assertRaises(PermissionError):
                book_reservation(
                    order_id="O2",
                    reservation_id="R2",
                    voucher_id="V1",
                    plan=plan,
                    compatible_ephemerides=candidates,
                    excluded=[],
                    shifts_snapshot=[],
                    query_summary=None,
                    registry_path=reg,
                    state_dir=td_path,
                )


class GeneratedCalibrationTests(unittest.TestCase):
    def test_generator_and_holdout_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = generate_sample_tree(Path(td) / "sample")
            cal = CalibrationInput.load(root / "calibration_original")
            result = calibrate(cal)
            self.assertEqual(len(result["holdout_validation"]), 3)
            query = evaluate_observation(result, [6, 3], [0.3, 0.3], "se95")
            self.assertEqual(query["supported_stars"], ["乙"])


if __name__ == "__main__":
    unittest.main()
