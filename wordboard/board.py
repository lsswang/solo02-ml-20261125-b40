"""Physical two-slot pages, layout enumeration, and exact action costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Iterable

SLOT_NAMES = ("首格", "另一格")


@dataclass(frozen=True)
class Position:
    page: int  # zero-based
    slot: int  # zero-based


@dataclass
class CostConfig:
    select: float = 1.0
    wait: float = 1.0
    page_turn: float = 3.0
    toggle: float = 0.5
    card_swap: float = 4.0
    default_slot: int = 0
    landing_slot_after_page_turn: int = 0
    page_turn_resets_slot: bool = True
    slots_per_page: int = 2
    max_words: int = 6

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "CostConfig":
        raw = dict(data.get("config", {}) or {})
        # Accept a few synonyms to make replacement input convenient.
        aliases = {
            "select": ("select_cost", "selection_cost", "选词拍"),
            "wait": ("wait_cost", "scan_wait_cost", "等待拍"),
            "page_turn": ("page_turn_cost", "turn_page_cost", "切页费", "翻页费"),
            "toggle": ("toggle_cost", "adjust_cost", "preset_cost", "拨头费", "调整费"),
            "card_swap": ("card_swap_cost", "swap_cost", "swap_card_cost", "refamiliarize_cost", "换卡费", "熟悉费"),
        }
        values: dict[str, Any] = {}
        for field_name, keys in aliases.items():
            for key in (field_name, *keys):
                if key in raw:
                    values[field_name] = raw[key]
                    break
        for key in (
            "default_slot",
            "landing_slot_after_page_turn",
            "page_turn_resets_slot",
            "slots_per_page",
            "max_words",
        ):
            if key in raw:
                values[key] = raw[key]
        # Chinese 1-based slot can be supplied as default_slot_number.
        if "default_slot_number" in raw and "default_slot" not in values:
            values["default_slot"] = int(raw["default_slot_number"]) - 1
        return cls(**values)


@dataclass
class BoardSpec:
    words: list[str]
    num_pages: int
    slots_per_page: int = 2
    fixed: dict[str, Position] = field(default_factory=dict)
    familiar: set[str] = field(default_factory=set)
    swappable: set[str] = field(default_factory=set)
    labels: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.words) > 6:
            raise ValueError("一块板最多六词")
        if self.slots_per_page != 2:
            raise ValueError("本模拟板按题设每页两格")
        seen: dict[Position, str] = {}
        for word, pos in self.fixed.items():
            if not (0 <= pos.page < self.num_pages and 0 <= pos.slot < self.slots_per_page):
                raise ValueError(f"固定位置越界: {word} {pos}")
            if pos in seen:
                raise ValueError(f"位置 {pos} 同时分配给 {seen[pos]} 和 {word}")
            seen[pos] = word
        for word in self.words:
            if word not in self.fixed and word in self.familiar:
                raise ValueError(f"熟词 {word} 必须在输入中给出固定页格")

    @property
    def all_slots(self) -> list[Position]:
        return [Position(p, s) for p in range(self.num_pages) for s in range(self.slots_per_page)]

    @property
    def free_slots(self) -> list[Position]:
        occupied = set(self.fixed.values())
        return [p for p in self.all_slots if p not in occupied]

    @property
    def free_words(self) -> list[str]:
        return [w for w in self.words if w not in self.fixed]


@dataclass
class Layout:
    spec: BoardSpec
    positions: dict[str, Position]

    @property
    def num_pages(self) -> int:
        return self.spec.num_pages

    @property
    def slots_per_page(self) -> int:
        return self.spec.slots_per_page

    def position(self, word: str) -> Position:
        if word not in self.positions:
            raise KeyError(f"词板上没有词: {word}")
        return self.positions[word]

    def grid(self) -> list[list[str | None]]:
        out: list[list[str | None]] = [
            [None for _ in range(self.slots_per_page)] for _ in range(self.num_pages)
        ]
        for word, pos in self.positions.items():
            out[pos.page][pos.slot] = word
        return out

    def signature(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(sorted((w, p.page + 1, p.slot + 1) for w, p in self.positions.items()))

    def swap(self, word_a: str, word_b: str) -> "Layout":
        if word_a not in self.positions or word_b not in self.positions:
            raise KeyError("只能交换板上已有的词")
        if word_a in self.spec.fixed or word_b in self.spec.fixed:
            raise ValueError("熟词/锁定位置不能交换")
        positions = dict(self.positions)
        positions[word_a], positions[word_b] = positions[word_b], positions[word_a]
        return Layout(spec=self.spec, positions=positions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_pages": self.num_pages,
            "cards": [
                {"word": w, "page": p.page + 1, "slot": p.slot + 1,
                 "slot_name": SLOT_NAMES[p.slot],
                 "fixed": w in self.spec.fixed,
                 "familiar": w in self.spec.familiar}
                for w, p in sorted(self.positions.items(), key=lambda kv: (kv[1].page, kv[1].slot))
            ],
            "grid_1_based_pages": [
                {
                    "page": i + 1,
                    "first": row[0],
                    "second": row[1] if len(row) > 1 else None,
                }
                for i, row in enumerate(self.grid())
            ],
        }


def parse_board(data: dict[str, Any], costs: CostConfig, extra_words: Iterable[str] = ()) -> BoardSpec:
    board = data.get("board", {}) or {}
    cards = board.get("cards", board.get("words", []))
    card_by_word: dict[str, dict[str, Any]] = {}
    ordered_words: list[str] = []

    for card in cards:
        if isinstance(card, str):
            word, obj = card, {}
        else:
            word = card.get("word", card.get("text", card.get("词")))
            obj = card
        if word is None:
            continue
        word = str(word)
        if word not in card_by_word:
            ordered_words.append(word)
        card_by_word[word] = obj

    # Task/replay words can implicitly become cards, making a minimal input easy.
    for word in extra_words:
        word = str(word)
        if word not in card_by_word:
            ordered_words.append(word)
            card_by_word[word] = {}

    if len(ordered_words) > costs.max_words:
        raise ValueError(f"词板最多 {costs.max_words} 词，输入给出 {len(ordered_words)} 词")

    explicit_pages: list[int] = []
    fixed: dict[str, Position] = {}
    familiar: set[str] = set()
    swappable: set[str] = set()

    for word in ordered_words:
        obj = card_by_word.get(word, {})
        is_familiar = bool(obj.get("familiar", obj.get("熟词", False)))
        is_fixed = bool(obj.get("fixed", obj.get("locked", obj.get("锁定", False))))
        if is_familiar:
            familiar.add(word)
        if obj.get("swappable", obj.get("可换", False)):
            swappable.add(word)
        if is_familiar or is_fixed or "page" in obj or "页" in obj or "slot" in obj or "格" in obj:
            page = int(obj.get("page", obj.get("页", 1))) - 1
            slot_value = obj.get("slot", obj.get("格", 1))
            # Accept Chinese labels "首格"/"另一格".
            if slot_value in ("首格", "first", 1):
                slot = 0
            elif slot_value in ("另一格", "second", 2):
                slot = 1
            else:
                slot = int(slot_value) - 1
            fixed[word] = Position(page=page, slot=slot)
            explicit_pages.append(page + 1)

    num_pages = board.get("num_pages", board.get("pages", data.get("num_pages")))
    if num_pages is None:
        max_page = max(explicit_pages, default=0)
        num_pages = max(1, max((len(ordered_words) + 1) // 2, max_page))
    num_pages = int(num_pages)

    # Unfamiliar cards are the permitted one-pair swap set unless explicitly given.
    swap_pair = board.get("swap_pair", data.get("swap_pair"))
    if swap_pair:
        swappable = {str(swap_pair[0]), str(swap_pair[1])}
    elif not swappable:
        swappable = {w for w in ordered_words if w not in familiar and w not in fixed}
    else:
        swappable |= {w for w in ordered_words if card_by_word[w].get("swappable", False)}

    spec = BoardSpec(
        words=ordered_words,
        num_pages=num_pages,
        slots_per_page=int(board.get("slots_per_page", costs.slots_per_page)),
        fixed=fixed,
        familiar=familiar,
        swappable=swappable,
    )
    spec.validate()
    return spec


def enumerate_layouts(spec: BoardSpec) -> list[Layout]:
    free_words = spec.free_words
    free_slots = spec.free_slots
    if len(free_words) > len(free_slots):
        raise ValueError("空格不足，无法放下所有词")
    layouts: list[Layout] = []
    for chosen in permutations(free_slots, len(free_words)):
        positions = dict(spec.fixed)
        for word, pos in zip(free_words, chosen):
            positions[word] = pos
        layouts.append(Layout(spec=spec, positions=positions))
    # Deterministic order independent of dict/set iteration.
    layouts.sort(key=lambda lay: tuple((p.page, p.slot, w) for w, p in sorted(lay.positions.items())))
    return layouts


def page_steps(current_page: int, target_page: int, num_pages: int) -> int:
    """Forward-only cyclic page distance."""
    return (int(target_page) - int(current_page)) % int(num_pages)


def action_cost(
    layout: Layout,
    target_word: str,
    current_page: int,
    costs: CostConfig,
    chosen_slot: int | None = None,
) -> dict[str, Any]:
    """Micro-enumerated cost of one selection.

    ``chosen_slot`` is the scanning start slot chosen by the preset rule.
    Non-default starts add the device's adjustment/head-moving overhead.
    When a page turn occurs, the simulated device lands in the first slot by
    default; set ``page_turn_resets_slot=False`` for a head-phase-persistent
    device.
    """
    target = layout.position(target_word)
    chosen = costs.default_slot if chosen_slot is None else int(chosen_slot)
    steps = page_steps(current_page, target.page, layout.num_pages)
    page_cost = steps * costs.page_turn
    adjustment = costs.toggle if chosen != costs.default_slot else 0.0

    if steps > 0 and costs.page_turn_resets_slot:
        scan_start = costs.landing_slot_after_page_turn
    else:
        scan_start = chosen
    waits = 1 if target.slot != scan_start else 0
    wait_cost = waits * costs.wait
    select_cost = costs.select
    total = adjustment + page_cost + wait_cost + select_cost
    return {
        "target_word": target_word,
        "current_page": current_page + 1,
        "target_page": target.page + 1,
        "target_slot": target.slot + 1,
        "chosen_start_slot": chosen + 1,
        "page_steps_forward": steps,
        "page_turn_cost": page_cost,
        "adjustment_cost": adjustment,
        "wait_beats": waits,
        "wait_cost": wait_cost,
        "select_cost": select_cost,
        "total_cost": total,
    }


def slot_preset_break_even_probability(costs: CostConfig) -> float:
    """Probability of the non-default slot above which presetting it wins."""
    if costs.wait <= 0:
        return 1.0
    return 0.5 + costs.toggle / (2.0 * costs.wait)
