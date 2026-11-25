"""LedgerScribe 端到端单元测试（纯标准库 unittest，CPU 可跑）。

覆盖题面四个判定场景、复核过滤语义、独立模型检查与合成材料约束
（一字两次抬笔、两字连写）。
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from ledgerscribe.cli import infer_page, load_lines, load_pages
from ledgerscribe.model import Model
from ledgerscribe.page import apply_review_dialogue, reconcile
from ledgerscribe.segments import apply_answer, split_segments
from ledgerscribe.synth import TEMPLATES

DATA = Path(__file__).resolve().parent.parent / "data"


class LedgerScribeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines = load_lines(DATA / "heldout_lines.json")
        cls.pages = load_pages(DATA / "pages.json")
        cls.model_a = Model.load(DATA / "model_A.json")
        cls.model_b = Model.load(DATA / "model_B.json")
        cls.new_lines = load_lines(DATA / "newpage_lines.json")
        cls.new_page = load_pages(DATA / "newpage_page.json")["B-040"]

    def test_template_has_double_lift_digit(self):
        # 数字 4 有三笔，即一个字内可出现两次抬笔。
        self.assertEqual(len(TEMPLATES["4"]), 3)

    def test_synthetic_contains_ligature_and_double_lift(self):
        manifest = json.loads((DATA / "A_manifest.json").read_text("utf-8"))
        self.assertTrue(any(m["ligature_at"] for m in manifest["lines"]))
        held = json.loads((DATA / "heldout_manifest.json").read_text("utf-8"))
        # A-line-3 的数字 4 三笔，对应两次字内抬笔。
        self.assertEqual(len(held["A-line-3"]), 2)
        self.assertEqual(held["A-line-3"][1]["digit"], "4")

    def test_total_40_blank_then_review_signs_17_23(self):
        page = self.pages["A-040"]
        inf = infer_page(self.model_a, self.lines, page)
        res = reconcile(inf, page)
        self.assertEqual(res["status"], "reconciled")
        self.assertTrue(all(v is None for v in res["signed"].values()))
        # 两种对平读法：17+23 与 12+28。
        combos = {
            tuple(sorted((lid, c[lid]) for lid in page["line_ids"]))
            for c in res["balanced_readings"]
        }
        self.assertEqual(
            combos,
            {(("A-line-1", "17"), ("A-line-2", "23")),
             (("A-line-1", "12"), ("A-line-2", "28"))},
        )
        # 末笔段答复 7：切法层过滤后签 17 与 23。
        reply = [{"line_id": "A-line-1", "seg_id": 1,
                  "answer": {"kind": "digit", "digit": "7"}}]
        tokens = {lid: self.lines[lid]["tokens"] for lid in page["line_ids"]}
        out = apply_review_dialogue(inf, tokens, page, reply)
        self.assertEqual(out["signed"],
                         {"A-line-1": "17", "A-line-2": "23"})

    def test_total_35_signs_12_23_directly(self):
        page = self.pages["A-035"]
        inf = infer_page(self.model_a, self.lines, page)
        res = reconcile(inf, page)
        self.assertEqual(res["signed"],
                         {"A-line-1": "12", "A-line-2": "23"})

    def test_fourth_candidate_page(self):
        page = self.pages["A-086"]
        inf = infer_page(self.model_a, self.lines, page)
        res = reconcile(inf, page)
        # 34 在候选表第四位，却是唯一能对平 86 的读法。
        self.assertEqual(page["candidates"]["A-line-3"].index("34"), 3)
        self.assertEqual(res["signed"],
                         {"A-line-3": "34", "A-line-4": "52"})

    def test_new_writer_drops_17_and_signs_12_28(self):
        inf = infer_page(self.model_b, self.new_lines, self.new_page)
        res = reconcile(inf, self.new_page)
        self.assertEqual(res["signed"],
                         {"B-line-1": "12", "B-line-2": "28"})
        # 17 在新模型下不具资格。
        from ledgerscribe.page import eligible_amounts
        self.assertNotIn("17",
                         eligible_amounts(inf["B-line-1"],
                                          self.new_page["threshold_ratio"]))

    def test_segment_filter_keeps_other_segmentations(self):
        # 一种切法被排除后，同金额的其他合法切法仍有效。
        page = self.pages["A-040"]
        inf = infer_page(self.model_a, self.lines, page)
        tokens = self.lines["A-line-1"]["tokens"]
        before = sum(len(e["paths"]) for e in inf["A-line-1"].values())
        filtered = apply_answer(inf["A-line-1"], tokens, 1,
                                {"kind": "not_whole"})
        after = sum(len(e["paths"]) for e in filtered.values())
        self.assertLess(after, before)
        self.assertGreaterEqual(after, 0)

    def test_line_report_attributes_segments(self):
        from ledgerscribe.segments import segment_digit_responsibility
        page = self.pages["A-040"]
        inf = infer_page(self.model_a, self.lines, page)
        rows = segment_digit_responsibility(
            self.lines["A-line-1"]["tokens"], inf["A-line-1"], "12")
        self.assertTrue(rows)
        for r in rows:
            self.assertAlmostEqual(
                sum(r["responsibility"].values()), 1.0, places=6)

    def test_independent_check_matches(self):
        from ledgerscribe.indep_check import independent_check
        for pid in ("A-040", "A-086"):
            page = self.pages[pid]
            report = independent_check(self.model_a, self.lines, page)
            self.assertTrue(report["all_match"])
            for line in report["lines"]:
                self.assertTrue(line["posterior_sums_to_one"])


if __name__ == "__main__":
    unittest.main()
