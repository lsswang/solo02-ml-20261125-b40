from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .scheduling import Ephemeris


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_registry(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"schema": "observatory-voucher-registry-v1", "vouchers": {}, "reservations": {}}


def save_registry(registry: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ephemeris_map(reservation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {e["id"]: e for e in reservation["compatible_ephemerides"]}


def book_reservation(
    *,
    order_id: str,
    reservation_id: str,
    voucher_id: str,
    plan: dict[str, Any],
    compatible_ephemerides: list[Ephemeris],
    excluded: list[dict[str, Any]],
    shifts_snapshot: list[dict[str, Any]],
    query_summary: dict[str, Any] | None,
    registry_path: str | Path,
    state_dir: str | Path,
    force: bool = False,
) -> tuple[dict[str, Any], Path]:
    registry_path = Path(registry_path)
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(registry_path)

    existing = registry["vouchers"].get(voucher_id)
    if existing and existing.get("order_id") != order_id and not force:
        raise PermissionError(
            f"voucher {voucher_id} is already promised to order {existing.get('order_id')} "
            f"under reservation {existing.get('reservation_id')}; one voucher cannot be given to two groups"
        )

    reservation_path = state_dir / f"{reservation_id}.json"
    if reservation_path.exists() and not force:
        raise FileExistsError(f"reservation already exists: {reservation_path} (use --force to replace)")

    reservation = {
        "schema": "observatory-reservation-v1",
        "reservation_id": reservation_id,
        "order_id": order_id,
        "voucher_id": voucher_id,
        "created_at": _now(),
        "status": "awaiting_first_result",
        "method": "flexible_voucher",
        "plan": plan,
        "compatible_ephemerides": [e.to_dict() for e in compatible_ephemerides],
        "excluded_ephemerides": excluded,
        "shifts_snapshot": shifts_snapshot,
        "photometry_summary": query_summary,
        "first_result": None,
        "first_recorded_at": None,
        "actual_redeem_night": None,
        "second_result": None,
        "second_recorded_at": None,
        "final_compatible_ids": None,
    }
    reservation_path.write_text(
        json.dumps(reservation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    registry["vouchers"][voucher_id] = {
        "order_id": order_id,
        "reservation_id": reservation_id,
        "reservation_path": str(reservation_path.resolve()),
        "held_since": _now(),
    }
    registry["reservations"][reservation_id] = {
        "order_id": order_id,
        "voucher_id": voucher_id,
        "path": str(reservation_path.resolve()),
    }
    save_registry(registry, registry_path)
    return reservation, reservation_path


def resolve_reservation_path(
    reservation: str | Path | None = None,
    reservation_id: str | None = None,
    state_dir: str | Path = "out/state",
    registry_path: str | Path | None = None,
) -> Path:
    if reservation is not None:
        p = Path(reservation)
        if p.exists():
            return p
        if reservation_id is None:
            raise FileNotFoundError(f"reservation file not found: {p}")
    if reservation_id:
        direct = Path(state_dir) / f"{reservation_id}.json"
        if direct.exists():
            return direct
        if registry_path:
            reg = load_registry(registry_path)
            info = reg.get("reservations", {}).get(reservation_id)
            if info and Path(info["path"]).exists():
                return Path(info["path"])
        raise FileNotFoundError(f"reservation id {reservation_id!r} not found in {state_dir}")
    raise ValueError("provide --reservation or --reservation-id")


def load_reservation(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_reservation(reservation: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(reservation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _bit(result: str | int) -> int:
    text = str(result).strip().lower()
    if text in {"1", "dim", "dimmed", "dark", "变暗", "暗"}:
        return 1
    if text in {"0", "bright", "clear", "not-dim", "not_dim", "未变暗", "亮"}:
        return 0
    raise ValueError("result must be dim/1 or bright/0")


def record_first(reservation: dict[str, Any], result: str | int) -> dict[str, Any]:
    if reservation["status"] not in {"awaiting_first_result", "first_recorded"}:
        raise RuntimeError(f"cannot record first result in status {reservation['status']}")
    bit = _bit(result)
    policy = reservation["plan"]["policy_by_first_bit"][str(bit)]
    if not policy.get("possible"):
        # An outcome absent from all input ephemerides is still an observation.
        # It means the application/ephemeris set is wrong, not a software crash.
        after_first: list[dict[str, Any]] = []
        redeem_night = None
        instruction = (
            "Backfill accepted. This first-result branch is not predicted by any ephemeris; "
            "inspect the application instead of redeeming the voucher."
        )
    else:
        ids = policy["after_first_ids"]
        emap = _ephemeris_map(reservation)
        after_first = [emap[i] for i in ids]
        redeem_night = policy["redeem_night"]
        if redeem_night is not None:
            instruction = (
                f"Backfill accepted. Redeem the voucher for night {redeem_night} "
                "and report its binary result."
            )
        elif policy.get("action") == "skip_redemption":
            instruction = (
                "Backfill accepted. The cost-optimal policy does not redeem the voucher; "
                "the remaining ephemeris count cannot be improved by any open shift."
            )
        else:
            instruction = "Backfill accepted. No voucher night is defined; inspect ephemerides."

    reservation["first_result"] = bit
    reservation["first_recorded_at"] = _now()
    reservation["actual_redeem_night"] = redeem_night
    reservation["status"] = "first_recorded"
    reservation["compatible_after_first"] = after_first
    reservation["action_for_successor"] = {
        "watch_night": redeem_night,
        "instruction": instruction,
        "still_compatible_ephemerides": after_first,
    }
    return reservation


def record_second(reservation: dict[str, Any], result: str | int) -> dict[str, Any]:
    if reservation["status"] != "first_recorded":
        raise RuntimeError("record the first result before the redemption-night result")
    if reservation.get("actual_redeem_night") is None:
        raise RuntimeError("this reservation has no redemption night")
    bit = _bit(result)
    first_bit = reservation["first_result"]
    redeem_night = int(reservation["actual_redeem_night"])
    final: list[dict[str, Any]] = []
    for e in reservation["compatible_ephemerides"]:
        pattern = [int(ch) for ch in e["pattern"]]
        if pattern[reservation["plan"]["first_night"] - 1] != first_bit:
            continue
        if pattern[redeem_night - 1] != bit:
            continue
        final.append(e)
    reservation["second_result"] = bit
    reservation["second_recorded_at"] = _now()
    reservation["final_compatible_ids"] = [e["id"] for e in final]
    reservation["final_compatible_ephemerides"] = final
    reservation["status"] = "completed"
    return reservation
