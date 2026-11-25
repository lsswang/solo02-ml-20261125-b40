"""Causal replay of actual new sentences on the simulated board."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .board import CostConfig, Layout, action_cost
from .session import SessionStore


class PresetRule:
    """Fixed lookup rule.  It cannot see the word about to be spoken."""

    def __init__(self, table: dict[tuple[str, str], int], default_slot: int = 0):
        self.table = table
        self.default_slot = default_slot
        self.calls: list[dict[str, Any]] = []

    def start_slot(self, scene: str, previous_word: str | None) -> int:
        if previous_word is None:
            chosen = self.default_slot
            reason = "no_previous_word_use_default_first_slot"
        else:
            chosen = self.table.get((scene, previous_word), self.default_slot)
            reason = "fixed_scene_previous_word_lookup"
        self.calls.append(
            {"scene": scene, "previous_word": previous_word, "chosen_slot": chosen + 1, "reason": reason}
        )
        return chosen


def replay_sentence(
    request: dict[str, Any],
    store: SessionStore,
    layouts: dict[str, Layout],
    rule: PresetRule,
    costs: CostConfig,
    route: str = "fixed_preset",
) -> tuple[dict[str, Any], SessionStore]:
    session_id = str(request["session_id"])
    scene = str(request.get("scene", "home"))
    sentence = [str(w) for w in request.get("sentence", [])]
    if not 1 <= len(sentence) <= 3 and not request.get("allow_long"):
        # The planned task contract is 1-3 words.  Replay can be longer only if explicit.
        if len(sentence) > 3:
            raise ValueError("示例任务长度应为 1 至 3 词；如确需长句请设置 allow_long=true")

    start_page = int(request.get("start_page", 1))
    session = store.ensure(session_id, scene=scene, start_page=start_page, route=route)
    entry_cost = 0.0
    if session.get("closed"):
        store.reopen(session_id)
    if request.get("switch_scene"):
        swap_info = request.get("scene_swap") or {}
        entry_cost = float(swap_info.get("cost", 0.0))
        store.switch_scene(
            session_id,
            scene,
            entry_event={"card_swap": swap_info or None, "entry_cost": entry_cost},
        )

    close_after = request.get("close_after", [])
    if isinstance(close_after, int):
        close_after = [close_after]
    close_after = set(int(x) for x in close_after)

    steps: list[dict[str, Any]] = []
    active_store = store
    for index, word in enumerate(sentence, start=1):
        session = active_store.get(session_id)
        before_page = int(session["page"])  # 1-based
        previous_word = session.get("last_word")
        if route == "swappable_cards":
            chosen_slot = costs.default_slot
            rule_call = {
                "scene": scene,
                "previous_word": previous_word,
                "chosen_slot": chosen_slot + 1,
                "reason": "swappable_card_route_always_scan_from_first_slot",
            }
        else:
            chosen_slot = rule.start_slot(scene, previous_word)
            rule_call = rule.calls[-1]
        breakdown = action_cost(
            layouts[scene],
            word,
            current_page=before_page - 1,
            costs=costs,
            chosen_slot=chosen_slot,
        )
        step = {
            "word_index": index,
            "word": word,
            "session_id": session_id,
            "scene": scene,
            "page_before": before_page,
            "previous_selected_word": previous_word,
            "preset_rule_call": rule_call,
            **breakdown,
        }
        active_store.record_word(session_id, step)
        steps.append(step)

        if index in close_after:
            active_store.close(session_id)
            # Re-read from disk in a brand-new object to prove persistence works.
            reloaded = SessionStore(active_store.path)
            restored_page = reloaded.get(session_id)["page"]
            if restored_page != breakdown["target_page"]:
                raise AssertionError("关闭后恢复的页位与实际页位不一致")
            reloaded.append_event(
                session_id,
                {"type": "verified_restore", "restored_page": restored_page, "after_word": word},
            )
            if index < len(sentence):
                reloaded.reopen(session_id)
            active_store = reloaded

    final_session = active_store.get(session_id)
    trace = {
        "session_id": session_id,
        "scene": scene,
        "sentence": sentence,
        "route": route,
        "step_count": len(steps),
        "scene_entry_cost": entry_cost,
        "total_page_turns": sum(int(s["page_steps_forward"]) for s in steps),
        "total_wait_beats": sum(int(s["wait_beats"]) for s in steps),
        "total_adjustment_cost": sum(float(s["adjustment_cost"]) for s in steps),
        "total_actual_cost": entry_cost + sum(s["total_cost"] for s in steps),
        "word_selection_cost": sum(s["total_cost"] for s in steps),
        "final_page": final_session["page"],
        "closed": final_session["closed"],
        "steps": steps,
    }
    return trace, active_store
