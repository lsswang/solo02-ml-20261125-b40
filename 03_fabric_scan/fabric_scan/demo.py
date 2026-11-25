"""题目指定的两格玩具场景与端到端核对（纯标准库，可重跑）。

运行：python -m fabric_scan.cli demo --out-dir out
所有关键结论都带 assert 断言，同时把明细写成 JSON 供查验。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from .common import condition, dump_json, marginals, posterior, str_to_map
from .datagen import MATERIALS, make_dataset, split_fabrics
from .learn import learn_model
from .order import (blocked_cells, issue_cut, new_ledger, save_ledger, settle)
from .policy import (best_cut, bridge_plan, portable_plan, realized_cut_loss,
                     segment_cells)
from .workflow import place_order


def _toy_order(device: str = "portable", retain: List[int] = None) -> Dict[str, Any]:
    return {
        "order_id": "TOY-1", "fabric_id": "TOY", "device": device,
        "losses": [1, 1], "retain": retain or [],
        "head0": 0, "startup_fee": 0.2, "move_fee": 0.1,
        "bridge_fee": 0.3, "readings": [],  # 玩具模型的先验已含“各半概率坏”
    }


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def run_demo(out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    report: Dict[str, Any] = {"checks": [], "fee_walkthrough": []}

    def check(name: str, got: Any, expect: Any) -> None:
        ok = got == expect if not isinstance(expect, float) else _approx(got, expect)
        report["checks"].append({"name": name, "got": got, "expected": expect,
                                 "pass": bool(ok)})
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: got={got} expected={expect}")
        assert ok, f"核对失败: {name}"

    # 两种两格先验：精检前两格坏布概率都各是 0.5
    prior_exactly_one = {"01": 0.5, "10": 0.5}   # 整图总是恰坏一格
    prior_same = {"00": 0.5, "11": 0.5}          # 整图总是同好同坏
    for name, pr in (("exactly_one", prior_exactly_one), ("same_status", prior_same)):
        m = marginals(pr)
        check(f"{name}: 精检前两格坏布概率均为0.5", [round(x, 6) for x in m],
              [0.5, 0.5])

    order = _toy_order()

    # 场景一：恰坏一格，精检第一格为好 -> 应剪第二格，损失 0
    post1 = condition(prior_exactly_one, 0, 0)
    cut1 = best_cut(post1, order["losses"], order["retain"], [])
    check("恰坏一格+第一格好: 剪口=(1,1)", cut1["segment"], (1, 1))
    check("恰坏一格+第一格好: 期望损失=0", round(cut1["expected_loss"], 9), 0.0)
    check("恰坏一格+第一格好: 对真值01实际损失=0",
          realized_cut_loss("01", cut1["segment"], order["losses"]), 0.0)

    # 场景二：同好同坏，精检第一格为好 -> 两格都保留，损失 0
    post2 = condition(prior_same, 0, 0)
    cut2 = best_cut(post2, order["losses"], order["retain"], [])
    check("同好同坏+第一格好: 不剪(None)", cut2["segment"], None)
    check("同好同坏+第一格好: 期望损失=0", round(cut2["expected_loss"], 9), 0.0)

    # 场景三：固定第一种模型与精检结果，只把第二格列为客户留样
    order_retain = _toy_order(retain=[1])
    cut3 = best_cut(post1, order_retain["losses"], order_retain["retain"], [])
    check("留样第二格: 剪口必须为空", cut3["segment"], None)
    check("留样第二格: 计漏剪损失=1",
          round(realized_cut_loss("01", None, order_retain["losses"]), 9), 1.0)

    # 决策层：低费率下复扫值得；另一条桥式头路线的结果一并给出用于理解差别
    plan_p = portable_plan(prior_exactly_one, order["losses"], order["retain"],
                           [], order["head0"], order["startup_fee"], order["move_fee"])
    plan_b = bridge_plan(prior_exactly_one, order["losses"], order["retain"],
                         [], order["bridge_fee"])
    report["toy_plans"] = {"portable": plan_p, "bridge": plan_b}
    check("便携头选择扫第一格", plan_p["kind"], "one_scan")
    check("便携头扫描费=开机0.2+移动0.1*0=0.2", round(plan_p["scan_fee"], 9), 0.2)
    check("桥式头选择单格窗口一趟看清", plan_b["kind"], "one_pass")

    # ---------------- 费用逐项核对（小单 + 手算） ----------------
    # 便携头从 head0=2 出发：先扫格 4，再依结果扫格 0
    fee_order = {**_toy_order(), "head0": 2, "startup_fee": 3.0, "move_fee": 2.0}
    first_fee = fee_order["startup_fee"] + fee_order["move_fee"] * abs(4 - 2)   # 3+4=7
    second_fee = fee_order["move_fee"] * abs(0 - 4)                              # 8
    check("第一扫: 开机费3 + 移动费2*2 = 7", first_fee, 7)
    check("第二扫: 移动费2*4 = 8（开机只计一次）", second_fee, 8)
    check("两扫合计扫描费 = 15", first_fee + second_fee, 15)
    # 剪口损失手算：真值 01，剪 (1,1) -> 正确处理 -> 0；不剪 -> 漏剪 -> 1
    check("剪口损失(剪对坏格)=0", realized_cut_loss("01", (1, 1), [1, 1]), 0)
    check("剪口损失(漏剪坏格)=1", realized_cut_loss("01", None, [1, 1]), 1)
    check("剪口损失(误剪好格)=1", realized_cut_loss("00", (0, 0), [1, 1]), 1)
    report["fee_walkthrough"].append({
        "head0": 2, "startup_fee": 3.0, "move_fee": 2.0,
        "first_scan_cell": 4, "first_fee": first_fee,
        "second_scan_cell": 0, "second_fee": second_fee,
        "total_scan_fee": first_fee + second_fee,
    })

    # ---------------- 端到端：正常下单入口走两格玩具单 ----------------
    ledger = new_ledger()
    toy_result = place_order(
        {**dict(prior=prior_exactly_one, n=2)}, order,
        precision_available={0: 0}, ledger=ledger)
    check("入口单: 实扫格0且读数为好", toy_result["scanned_cells"], {0: 0})
    check("入口单: 开出剪口=(1,1)", toy_result["cut_segment"], (1, 1))
    settle(ledger, "TOY", "01", order)
    check("入口单: 结清实际损失=0",
          ledger["fabrics"]["TOY"]["events"][-1]["actual_cut_loss"], 0)

    # 留样单走同一入口
    retain_result = place_order(
        {"prior": prior_exactly_one, "n": 2}, order_retain,
        precision_available={0: 0}, ledger=ledger)
    check("留样入口单: 剪口为空", retain_result["cut_segment"], None)
    settle(ledger, "TOY", "01", order_retain)
    check("留样入口单: 漏剪损失=1 入账",
          ledger["fabrics"]["TOY"]["events"][-1]["actual_cut_loss"], 1)

    # 重新开单：档案显示已裁断格，冲突剪口被硬拦截
    blocked = blocked_cells(ledger, "TOY")
    check("重新开单: 格1已不可再剪", blocked, [1])
    guard_msg = ""
    try:
        issue_cut(ledger, "TOY", "BAD-1", (1, 1), post1, order)
    except ValueError as exc:
        guard_msg = str(exc)
    check("冲突剪口被拦截", guard_msg.startswith("剪口 (1, 1) 触及"), True)
    report["ledger_guard_message"] = guard_msg

    # ---------- 正常入口 + 未在演示中出现的读数：6 格合成材料 ----------
    dataset = make_dataset(MATERIALS["material_a"])
    parts = split_fabrics(dataset)
    model6 = learn_model(parts["train"], 6)
    synth_order = {
        "order_id": "SYN-1", "fabric_id": "SYN-FAB", "device": "portable",
        "losses": [1, 1, 2, 1, 1, 1], "retain": [5],
        "head0": 1, "startup_fee": 0.5, "move_fee": 0.2, "bridge_fee": 1.0,
        # 含一次缺读（null）：推断只略去该次观测
        "readings": [[0, 0], None, [1, 1], [0, 1], [0, 0]],
    }
    # 该台实际返回的单格读数：演示未用过的一份数据（格5 留样，设备不返该格）
    precision = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}
    synth_result = place_order(model6, synth_order, precision, ledger)
    check("合成单: 缺读下仍完成推断与开单", synth_result["cut_cells"] is not None, True)
    check("合成单: 留样格5不在剪口内", 5 not in synth_result["cut_cells"], True)
    settle(ledger, "SYN-FAB",
           "".join(str(b) for b in (0, 0, 1, 0, 0, 0)), synth_order)

    # 同一布号再开一单：blocked 生效，剪口自动避开已裁格
    synth_order2 = {**synth_order, "order_id": "SYN-2"}
    synth_result2 = place_order(model6, synth_order2, precision, ledger)
    already = set(blocked_cells(ledger, "SYN-FAB")) - {5}
    check("二次开单: 新剪口避开上一段",
          all(c not in already for c in synth_result2["cut_cells"]), True)

    report["toy_end_to_end"] = toy_result
    report["retain_end_to_end"] = retain_result
    report["synthetic_order"] = synth_result
    report["synthetic_order_reopen"] = synth_result2
    report["blocked_cells"] = {
        fid: ledger["fabrics"][fid]["blocked_cells"] for fid in ledger["fabrics"]}

    ledger_path = os.path.join(out_dir, "ledger.json")
    save_ledger(ledger, ledger_path)
    dump_json(report, os.path.join(out_dir, "demo_report.json"))
    print(f"\n档案 -> {ledger_path}\n明细 -> {os.path.join(out_dir, 'demo_report.json')}")
    return report
