"""系统测试：概率推断、学习、两种设备策略、单据档案全链路。

便携头与桥式头的最优期望成本在测试里用独立写的暴力枚举再算一遍，
与 policy 模块的结果逐项对照，避免“同一套代码自证”。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fabric_scan.common import (all_maps, condition, marginals, normalize,
                                posterior, str_to_map)
from fabric_scan.datagen import MATERIALS, make_dataset, split_fabrics
from fabric_scan.demo import run_demo
from fabric_scan.learn import evaluate, learn_model
from fabric_scan.order import (blocked_cells, issue_cut, new_ledger,
                               settle)
from fabric_scan.policy import (best_cut, bridge_plan, bridge_windows,
                                expected_cut_loss, feasible_segments,
                                portable_plan, realized_cut_loss, segment_cells)
from fabric_scan.workflow import place_order

PRIOR_ONE = {"01": 0.5, "10": 0.5}
PRIOR_SAME = {"00": 0.5, "11": 0.5}


def brute_best_cut(post, losses, retain, blocked):
    n = len(losses)
    return min(expected_cut_loss(post, seg, losses)
               for seg in feasible_segments(n, retain, blocked))


def cond_loss(post, obs, losses, retain, blocked):
    p = post
    for c, b in obs.items():
        p = condition(p, c, b)
    return best_cut(p, losses, retain, blocked)["expected_loss"]


def brute_portable(post, losses, retain, blocked, head0, startup, move):
    """独立暴力枚举：不扫 / 扫一格 / 自适应两格。"""
    n = len(losses)
    allowed = [j for j in range(n)
               if j not in set(retain) | set(blocked)]
    marg = marginals(post)
    best = brute_best_cut(post, losses, retain, blocked)
    for c1 in allowed:
        fee = startup + move * abs(c1 - head0)
        val = sum((marg[c1] if b else 1 - marg[c1])
                  * cond_loss(post, {c1: b}, losses, retain, blocked)
                  for b in (0, 1))
        best = min(best, fee + val)
    for c1 in allowed:
        for c2 in allowed:
            if c2 == c1:
                continue
            first_fee = startup + move * abs(c1 - head0)
            total = first_fee
            for b1 in (0, 1):
                p1 = marg[c1] if b1 else 1 - marg[c1]
                stop = cond_loss(post, {c1: b1}, losses, retain, blocked)
                extra = move * abs(c2 - c1)
                p_cond = condition(post, c1, b1)
                m2 = marginals(p_cond)
                go = extra + sum(
                    (m2[c2] if b2 else 1 - m2[c2])
                    * cond_loss(post, {c1: b1, c2: b2}, losses, retain, blocked)
                    for b2 in (0, 1))
                total += p1 * min(stop, go)
            best = min(best, total)
    return best


def joint_prob(post, cells, assignment):
    p = post
    prob = 1.0
    for c in cells:
        m = marginals(p)[c]
        prob *= m if assignment[c] == 1 else 1 - m
        p = condition(p, c, assignment[c])
    return prob


def brute_bridge(post, losses, retain, blocked, fee):
    best = brute_best_cut(post, losses, retain, blocked)
    for win in bridge_windows(len(losses), blocked):
        cells = list(range(win[0], win[1] + 1))
        val = 0.0
        for bits in range(1 << len(cells)):
            assignment = {c: (bits >> (len(cells) - 1 - k)) & 1
                          for k, c in enumerate(cells)}
            val += joint_prob(post, cells, assignment) * cond_loss(
                post, assignment, losses, retain, blocked)
        best = min(best, fee + val)
    return best


class TestInference(unittest.TestCase):
    def test_marginals_half_before_precision(self):
        self.assertEqual([round(x, 9) for x in marginals(PRIOR_ONE)], [0.5, 0.5])
        self.assertEqual([round(x, 9) for x in marginals(PRIOR_SAME)], [0.5, 0.5])

    def test_two_scenarios_opposite_decisions(self):
        post_one = condition(PRIOR_ONE, 0, 0)
        self.assertEqual(best_cut(post_one, [1, 1], [], [])["segment"], (1, 1))
        post_same = condition(PRIOR_SAME, 0, 0)
        self.assertEqual(best_cut(post_same, [1, 1], [], [])["segment"], None)

    def test_retain_cell_forces_skip_and_counts_miss(self):
        post = condition(PRIOR_ONE, 0, 0)
        cut = best_cut(post, [1, 1], [1], [])
        self.assertIsNone(cut["segment"])
        self.assertEqual(realized_cut_loss("01", None, [1, 1]), 1.0)

    def test_posterior_normalized_and_missing_reading_ignored(self):
        model = learn_model(
            split_fabrics(make_dataset(MATERIALS["material_a"]))["train"], 6)
        readings = [[0, 0], None, [1, 1], [0, 1], [0, 0]]
        p_full = posterior(model["prior"], model["emission"], readings)
        # 缺读在原位置保留 None：略去该次观测，其余观测的格位不错位
        p_skip = posterior(model["prior"], model["emission"],
                           [readings[0], None, readings[2], readings[3],
                            readings[4]])
        self.assertAlmostEqual(sum(p_full.values()), 1.0)
        self.assertEqual(set(p_full), set(p_skip))
        for k in p_full:
            self.assertAlmostEqual(p_full[k], p_skip[k])

    def test_emission_recovers_flip_rate(self):
        model = learn_model(
            split_fabrics(make_dataset(MATERIALS["material_a"]))["train"], 6)
        # flip=0.1：用原始计数核对“真值对被正确读到”的经验频率约为 0.81，
        # 不受学习平滑项影响
        correct, total = 0, 0
        for fab in split_fabrics(
                make_dataset(MATERIALS["material_a"]))["train"]:
            for i, read in enumerate(fab["readings"]):
                if read is None:
                    continue
                true_pair = fab["truth"][i:i + 2]
                total += 1
                if "".join(map(str, read)) == true_pair:
                    correct += 1
        self.assertAlmostEqual(correct / total, 0.81, delta=0.05)
        # 学来的分布每行都归一
        for row in model["emission"].values():
            self.assertAlmostEqual(sum(row.values()), 1.0)

    def test_holdout_evaluation(self):
        ds = make_dataset(MATERIALS["material_b"])
        parts = split_fabrics(ds)
        model = learn_model(parts["train"], 5)
        report = evaluate(model, parts["holdout"])
        self.assertEqual(report["n_holdout"], 6)
        for row in report["fabrics"]:
            self.assertGreater(row["probability_of_truth"], 0.0)
            self.assertAlmostEqual(sum(row["posterior"].values()), 1.0)


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.losses = [1, 1, 2, 1, 1]
        self.post = normalize({"00000": 1, "10101": 2, "01010": 1, "11111": 1})

    def test_feasible_segments_avoid_forbidden(self):
        segs = feasible_segments(5, [2], [4])
        for seg in segs:
            if seg is None:
                continue
            self.assertNotIn(2, segment_cells(seg))
            self.assertNotIn(4, segment_cells(seg))

    def test_portable_matches_brute_force(self):
        for startup, move in ((0.2, 0.1), (5.0, 2.0), (0.0, 0.0)):
            plan = portable_plan(self.post, self.losses, [2], [], 1,
                                 startup, move)
            expect = brute_portable(self.post, self.losses, [2], [], 1,
                                    startup, move)
            self.assertAlmostEqual(plan["expected_cost"], expect, places=9)

    def test_bridge_matches_brute_force(self):
        for fee in (0.3, 5.0, 0.0):
            plan = bridge_plan(self.post, self.losses, [2], [], fee)
            expect = brute_bridge(self.post, self.losses, [2], [], fee)
            self.assertAlmostEqual(plan["expected_cost"], expect, places=9)

    def test_huge_fees_mean_no_scan(self):
        plan = portable_plan(self.post, self.losses, [], [], 0, 99, 99)
        self.assertEqual(plan["kind"], "no_scan")
        bplan = bridge_plan(self.post, self.losses, [], [], 99.0)
        self.assertEqual(bplan["kind"], "no_scan")

    def test_adaptive_second_scan_branch_structure(self):
        # 零费用时两格扫严格不劣，分支必须给出 stop/scan_second 决策
        plan = portable_plan(self.post, self.losses, [], [], 0, 0.0, 0.0)
        if plan["kind"] == "two_scan":
            for key in ("first_0", "first_1"):
                self.assertIn(plan["branches"][key]["decision"],
                              ("stop", "scan_second"))


class TestLedgerAndWorkflow(unittest.TestCase):
    def test_end_to_end_and_blocked_guard(self):
        ledger = new_ledger()
        order = {"order_id": "O1", "fabric_id": "F", "device": "portable",
                 "losses": [1, 1], "retain": [], "head0": 0,
                 "startup_fee": 0.2, "move_fee": 0.1, "bridge_fee": 0.3,
                 "readings": []}
        result = place_order({"prior": PRIOR_ONE, "n": 2}, order,
                             precision_available={0: 0}, ledger=ledger)
        self.assertEqual(result["cut_segment"], (1, 1))
        types_ = [e["type"] for e in ledger["fabrics"]["F"]["events"]]
        self.assertEqual(types_, ["prepared", "reading", "cut"])
        self.assertEqual(blocked_cells(ledger, "F"), [1])
        with self.assertRaises(ValueError):
            issue_cut(ledger, "F", "BAD", (1, 1), PRIOR_ONE, order)
        settle(ledger, "F", "01", order)
        self.assertEqual(ledger["fabrics"]["F"]["events"][-1]
                         ["actual_cut_loss"], 0.0)

    def test_bridge_workflow_window_scan(self):
        ledger = new_ledger()
        order = {"order_id": "B1", "fabric_id": "G", "device": "bridge",
                 "losses": [1, 1], "retain": [], "head0": 0,
                 "startup_fee": 0.2, "move_fee": 0.1, "bridge_fee": 0.05,
                 "readings": []}
        result = place_order({"prior": PRIOR_ONE, "n": 2}, order,
                             precision_available={0: 0, 1: 1}, ledger=ledger)
        if result["plan_at_order_time"]["kind"] == "one_pass":
            self.assertTrue(all(s["leg"] == "one_pass"
                                for s in result["scan_steps"]))
            self.assertEqual(result["scan_fee"], 0.05)

    def test_demo_runs_and_all_checks_pass(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            report = run_demo(tmp)
            self.assertTrue(all(c["pass"] for c in report["checks"]))
            self.assertTrue(os.path.exists(os.path.join(tmp, "ledger.json")))


if __name__ == "__main__":
    unittest.main()
