from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .experiment import train_and_evaluate
from .model import BlendModel
from .recommend import recommend_recipes
from .shop import ShopService
from .utils import ensure_dir, read_json, write_json


def find_records(input_dir: str) -> str:
    for name in ("sensory_votes.csv", "sensory_records.csv", "blind_votes.csv"):
        path = os.path.join(input_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"no sensory_votes.csv / sensory_records.csv / blind_votes.csv in {input_dir}"
    )


def _action_summary(act: Dict[str, Any], result: Any) -> Dict[str, Any]:
    return {"cmd": act.get("cmd"), "label": act.get("label"), "ok": True, "result": result}


def execute_actions(shop: ShopService, actions: List[Dict[str, Any]], run_dir: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for act in actions:
        cmd = act["cmd"]
        try:
            if cmd == "order":
                res = shop.receive_order(
                    act["order_id"],
                    cups=int(act.get("cups", 1)),
                    composition=act["composition"],
                    roast_id=act["roast_id"],
                    formula_id=act.get("formula_id"),
                    mode=act.get("mode", "bucket"),
                    shift_id=act.get("shift_id"),
                    require_trial=bool(act.get("require_trial", False)),
                )
            elif cmd == "serve":
                res = shop.serve_order(act["order_id"])
            elif cmd == "trial":
                res = shop.record_trial(act["order_id"], act["result"])
            elif cmd == "withdraw":
                res = shop.withdraw_stock(
                    act["batch_id"], int(act["parts"]), act.get("reason", "stock withdrawal")
                )
            elif cmd == "scan":
                res = shop.scan_order(act["order_id"])
                if act.get("save"):
                    write_json(os.path.join(run_dir, str(act["save"])), res)
            elif cmd == "stock":
                res = shop.stock_report()
                if act.get("save"):
                    write_json(os.path.join(run_dir, str(act["save"])), res)
            elif cmd == "recommend":
                res = recommend_recipes(
                    shop.model,  # type: ignore[arg-type]
                    shop.state["loose_stock_parts"],
                    threshold=float(shop.state["config"]["gate_threshold"]),
                )
            else:
                raise ValueError(f"unknown action cmd {cmd!r}")
            item = _action_summary(act, res)
        except Exception as exc:  # demo scripts may intentionally probe an error
            if not act.get("allow_error", False):
                raise
            item = {"cmd": cmd, "label": act.get("label"), "ok": False, "error": str(exc)}
        out.append(item)
    return out


def run_pipeline(input_dir: str, out_dir: str) -> Dict[str, Any]:
    ensure_dir(out_dir)
    records_path = find_records(input_dir)
    inventory_path = os.path.join(input_dir, "inventory.json")
    if not os.path.exists(inventory_path):
        raise FileNotFoundError(f"missing {inventory_path}")
    inventory = read_json(inventory_path)
    plan_path = os.path.join(input_dir, "plan.json")
    plan = read_json(plan_path) if os.path.exists(plan_path) else {}

    threshold = float((inventory.get("config") or {}).get("gate_threshold", plan.get("threshold", 0.8)))
    requested_holdout = plan.get("holdout_roast")
    try:
        training = train_and_evaluate(
            records_path, out_dir, holdout_roast=requested_holdout, threshold=threshold
        )
        holdout_used = requested_holdout
    except ValueError as exc:
        if not requested_holdout:
            raise
        # A swapped file may be too small for the requested holdout.  Fall back
        # to deterministic grouping rather than refusing delivery.
        training = train_and_evaluate(records_path, out_dir, holdout_roast=None, threshold=threshold)
        holdout_used = None
        write_json(os.path.join(out_dir, "holdout_fallback.json"), {"requested": requested_holdout, "error": str(exc)})

    split_audit = training["report"]["split"]
    if split_audit.get("strategy") == "whole_roast_fallback":
        holdout_used = split_audit.get("selected_roast")
    elif split_audit.get("strategy") == "formula_group_fallback_no_whole_roast_available":
        holdout_used = None

    model = BlendModel.from_dict(read_json(os.path.join(out_dir, "model.json")))
    recommendations = recommend_recipes(
        model,
        inventory.get("loose_stock_parts", {}),
        threshold=threshold,
        parts_per_cup=int((inventory.get("config") or {}).get("parts_per_cup", 4)),
    )
    rec_payload = {
        "threshold": threshold,
        "holdout_roast_requested": requested_holdout,
        "holdout_roast_used_after_fallback": holdout_used,
        "feasible_passing_recipes": [
            r for r in recommendations if r["passes_gate"] and (r["max_cups_from_loose_stock"] or 0) >= 1
        ],
        "all_recipes": recommendations,
    }
    write_json(os.path.join(out_dir, "production_recommendations.json"), rec_payload)

    run_summaries: Dict[str, Any] = {}
    shops_root = os.path.join(out_dir, "shops")
    ensure_dir(shops_root)
    shop_runs = plan.get("shop_runs", [])
    for run in shop_runs:
        name = run["name"]
        run_dir = os.path.join(shops_root, name)
        try:
            shop = ShopService.initialize(
                run_dir,
                inventory,
                model=model,
                config_overrides=run.get("config_overrides"),
                reset=True,
            )
            actions = execute_actions(shop, run.get("actions", []), run_dir)
            stock = shop.stock_report()
            write_json(os.path.join(run_dir, "actions.json"), actions)
            write_json(os.path.join(run_dir, "stock_final.json"), stock)
            run_summaries[name] = {
                "status": "ok",
                "run_dir": os.path.relpath(run_dir, out_dir),
                "actions": actions,
                "stock_final": stock,
            }
        except Exception as exc:
            # A swapped sensory file may intentionally make the shipped scripted
            # orders incompatible.  Model deliverables must still be produced.
            ensure_dir(run_dir)
            write_json(os.path.join(run_dir, "run_error.json"), {"error": str(exc)})
            run_summaries[name] = {
                "status": "error",
                "run_dir": os.path.relpath(run_dir, out_dir),
                "error": str(exc),
            }

    # Preserve one concrete successful weighing at the top level.
    example = None
    preferred_run = plan.get("successful_weighing_run")
    if preferred_run and preferred_run in run_summaries:
        ledger_shop = ShopService(os.path.join(shops_root, preferred_run), model=model)
        successes = [w for w in ledger_shop.state["weighings"] if w.get("status") == "success"]
        if successes:
            example = successes[0]
    if example is None:
        for name in run_summaries:
            ledger_shop = ShopService(os.path.join(shops_root, name), model=model)
            bucket_success = [w for w in ledger_shop.state["weighings"] if w.get("kind") == "whole_bucket"]
            if bucket_success:
                example = bucket_success[0]
                break
    if example is not None:
        write_json(os.path.join(out_dir, "actual_successful_weighing_example.json"), example)

    summary = {
        "input_dir": os.path.abspath(input_dir),
        "out_dir": os.path.abspath(out_dir),
        "records": os.path.abspath(records_path),
        "blind_metrics": training["report"]["metrics"],
        "blind_roasts": training["report"]["split"]["blind_roasts"],
        "split_invariants_ok": training["report"]["split"]["invariants_ok"],
        "passing_recipes": rec_payload["feasible_passing_recipes"],
        "shop_runs": run_summaries,
        "successful_weighing_example": None if example is None else example["weighing_id"],
    }
    write_json(os.path.join(out_dir, "pipeline_summary.json"), summary)
    with open(os.path.join(out_dir, "pipeline_summary.md"), "w", encoding="utf-8") as f:
        f.write(render_summary_markdown(summary, rec_payload))
    return summary


def render_summary_markdown(summary: Dict[str, Any], rec: Dict[str, Any]) -> str:
    mt = summary["blind_metrics"]
    lines = [
        "# 咖啡拼配交付摘要",
        "",
        f"- 盲评烘次：`{', '.join(summary['blind_roasts'])}`",
        f"- 划分不变量通过：`{summary['split_invariants_ok']}`",
        f"- 盲评 log loss：**{mt['blind_vote_log_loss']:.6f}**；常数基线：{mt['intercept_baseline_log_loss']:.6f}",
        f"- 成功称料样例：`{summary.get('successful_weighing_example')}`",
        "",
        "## 过门槛且库存可做的配方",
        "",
    ]
    if not rec["feasible_passing_recipes"]:
        lines.append("无：第一单应更换配方或拒售并留料。")
    else:
        lines.append("| 配方 | 预测相似率 | 库存可做杯数 |")
        lines.append("|---|---:|---:|")
        for r in rec["feasible_passing_recipes"]:
            lines.append(
                f"| `{r['formula_id']}` | {r['predicted_probability']:.3f} | {r['max_cups_from_loose_stock']} |"
            )
    lines.extend(["", "## 营业演练", ""])
    for name, run in summary["shop_runs"].items():
        lines.append(f"### {name}")
        if run.get("status") == "error":
            lines.append(f"- 脚本营业动作与换入输入不兼容，已保留模型交付物：`{run.get('error')}`")
            lines.append("")
            continue
        for a in run.get("actions", []):
            if not a.get("ok", True):
                lines.append(f"- `{a['cmd']}` 预期/非预期错误：{a.get('error')}")
                continue
            res = a["result"]
            if a["cmd"] == "order":
                p = res.get("probability")
                p_text = "n/a" if p is None else f"{p:.3f}"
                lines.append(f"- `{res['order_id']}` → `{res.get('decision')}` (p={p_text})")
            elif a["cmd"] == "trial":
                lines.append(
                    f"- 试杯 `{res['reading']}`：后验 {res['posterior']:.3f} → `{res['decision']}`"
                )
            elif a["cmd"] == "serve":
                lines.append(f"- 出杯 `{res['order_id']}` ×{res['served_cups']} 成功")
            elif a["cmd"] == "withdraw":
                lines.append(f"- 提料/盘减事件：`{a['result'].get('type')}`")
        lines.append("")
    return "\n".join(lines)
