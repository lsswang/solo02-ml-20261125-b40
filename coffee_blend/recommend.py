from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .model import BlendModel
from .synthetic import all_quarter_compositions
from .utils import composition_to_string


def recommend_recipes(
    model: BlendModel,
    stock_parts: Dict[str, int],
    *,
    threshold: float = 0.8,
    parts_per_cup: int = 4,
    only_in_stock_batches: bool = True,
) -> List[Dict[str, Any]]:
    batches = list(stock_parts.keys()) if only_in_stock_batches else list(model.batches)
    # Keep model batch ordering where possible, then any inventory-only ids.
    ordered = [b for b in model.batches if b in batches]
    ordered += sorted(b for b in batches if b not in model.batches)
    rows: List[Dict[str, Any]] = []
    for parts in all_quarter_compositions(len(ordered), parts_per_cup):
        comp = {b: p for b, p in zip(ordered, parts) if p > 0}
        if not comp:
            continue
        try:
            p = model.predict_proba(comp)
            probability_source = "mixture_model"
        except KeyError:
            # Inventory may contain a new batch absent from an old model.
            p = None
            probability_source = "unknown_batch_refit_required"
        max_cups: Optional[int] = None
        if p is not None:
            max_cups = min(int(stock_parts.get(b, 0)) // dose for b, dose in comp.items())
        rows.append(
            {
                "formula_id": composition_to_string(comp),
                "composition": comp,
                "predicted_probability": p,
                "probability_source": probability_source,
                "passes_gate": bool(p is not None and p >= threshold),
                "max_cups_from_loose_stock": max_cups,
                "can_make_at_least_one": bool(max_cups and max_cups >= 1),
            }
        )
    rows.sort(key=lambda r: (r["predicted_probability"] is None, -(r["predicted_probability"] or -1.0)))
    return rows


def feasible_passing_recipe(rows: Sequence[Dict[str, Any]], cups: int = 1) -> Optional[Dict[str, Any]]:
    for r in rows:
        if r["passes_gate"] and (r["max_cups_from_loose_stock"] or 0) >= cups:
            return r
    return None
