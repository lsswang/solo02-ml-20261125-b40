from __future__ import annotations

import os
from typing import Any, Dict

from .shop import ShopService
from .utils import write_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def preweighed_half_bucket_demo() -> Dict[str, Any]:
    """Show the schema/state path for a half bucket weighed before orders."""
    state_dir = os.path.join(ROOT, "deliverables", "good", "shops", "06_preweighed_half_bucket_input")
    out_path = os.path.join(ROOT, "deliverables", "good", "preweighed_half_bucket_input_scan.json")
    inventory = {
        "batches": [
            {"batch_id": "JIA-2409", "bean_name": "甲豆"},
            {"batch_id": "YI-2409", "bean_name": "乙豆"},
            {"batch_id": "BING-2409", "bean_name": "丙豆"},
        ],
        # The preweighed bucket's beans are already occupied and therefore are
        # not included in these loose portions.
        "loose_stock_parts": {"JIA-2409": 2, "YI-2409": 2, "BING-2409": 4},
        "empty_buckets": [],
        "weighed_buckets": [
            {
                "bucket_id": "KEG-HALF-INPUT",
                "capacity_cups": 4,
                "cups_total": 4,
                "remaining_cups": 2,
                "roast_id": "R-PREWEIGHED",
                "formula_id": "F-J2-Y2",
                "composition": "JIA-2409:2|YI-2409:2",
                "weighing_id": "INPUT-WEIGHING-HALF-BUCKET",
            }
        ],
        "roast_states": [
            {
                "roast_id": "R-PREWEIGHED",
                "formula_id": "F-J2-Y2",
                "probability": 0.88,
                "source": "pre-calibrated input roast state",
            }
        ],
        "config": {"gate_threshold": 0.8},
    }
    shop = ShopService.initialize(state_dir, inventory, reset=True)
    shop.receive_order(
        "REFILL-1",
        cups=1,
        mode="bucket",
        roast_id="R-PREWEIGHED",
        formula_id="F-J2-Y2",
        composition="JIA-2409:2|YI-2409:2",
    )
    shop.receive_order(
        "REFILL-2",
        cups=1,
        mode="bucket",
        roast_id="R-PREWEIGHED",
        formula_id="F-J2-Y2",
        composition="JIA-2409:2|YI-2409:2",
    )
    scan = shop.scan_order("REFILL-1")
    write_json(out_path, scan)
    return scan


def main() -> int:
    scan = preweighed_half_bucket_demo()
    print(
        "preweighed half-bucket demo:",
        scan["remaining_buckets_and_refills"][0]["bucket_id"],
        "candidate",
        scan["remaining_buckets_and_refills"][0]["can_refill_orders"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
