"""Sparse finite-impulse-response room models and convolution."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

Kernel = List[Tuple[int, float]]
Room = List[List[Kernel]]  # room[mic][speaker]


def normalize_kernel(raw: Any) -> Kernel:
    kernel: Kernel = []
    for item in raw:
        if isinstance(item, dict):
            delay = int(item["delay"])
            coeff = float(item["coeff"])
        else:
            delay = int(item[0])
            coeff = float(item[1])
        if delay < 0:
            raise ValueError("impulse-response delays must be non-negative")
        if abs(coeff) > 1e-12:
            kernel.append((delay, coeff))
    kernel.sort()
    return kernel


def room_from_dict(data: Dict[str, Any]) -> Room:
    raw_ir = data["ir"]
    room: Room = []
    for mic_rows in raw_ir:
        room.append([normalize_kernel(k) for k in mic_rows])
    if len(room) != 2 or any(len(row) != 2 for row in room):
        raise ValueError("this bench expects exactly 2 microphones and 2 speakers")
    return room


def load_room(path: str) -> Tuple[Room, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return room_from_dict(data), data


def max_delay(room: Room) -> int:
    value = 0
    for mic in room:
        for kernel in mic:
            for delay, _ in kernel:
                value = max(value, delay)
    return value


def convolve_kernel(signal: Sequence[float], kernel: Kernel, length: int | None = None) -> List[float]:
    if length is None:
        dmax = max((d for d, _ in kernel), default=0)
        length = len(signal) + dmax
    out = [0.0] * length
    n = len(signal)
    for delay, coeff in kernel:
        if coeff == 0.0:
            continue
        # y[t] += coeff * x[t-delay]; valid source indices are 0..n-1.
        first_t = delay
        last_t = min(length, n + delay)
        for t in range(first_t, last_t):
            out[t] += coeff * signal[t - delay]
    return out


def convolve_room(speakers: Sequence[Sequence[float]], room: Room, length: int | None = None) -> List[List[float]]:
    if length is None:
        length = max((len(x) for x in speakers), default=0) + max_delay(room)
    mics = [[0.0] * length for _ in range(2)]
    for m in range(2):
        for s in range(2):
            part = convolve_kernel(speakers[s], room[m][s], length)
            for i, value in enumerate(part):
                mics[m][i] += value
    return mics


def h0_matrix(room: Room) -> List[List[float]]:
    h0 = [[0.0, 0.0], [0.0, 0.0]]
    for m in range(2):
        for s in range(2):
            for delay, coeff in room[m][s]:
                if delay == 0:
                    h0[m][s] += coeff
    return h0


def average_rooms(rooms: Sequence[Room]) -> Room:
    """Average coefficients on the union of sparse delay positions."""
    if not rooms:
        raise ValueError("at least one room is required")
    avg: Room = [[[], []], [[], []]]
    for m in range(2):
        for s in range(2):
            values: Dict[int, float] = {}
            for room in rooms:
                for delay, coeff in room[m][s]:
                    values[delay] = values.get(delay, 0.0) + coeff / len(rooms)
            avg[m][s] = sorted((d, c) for d, c in values.items() if abs(c) > 1e-10)
    return avg


def room_to_jsonable(room: Room) -> List[List[List[Dict[str, float]]]]:
    return [
        [
            [{"delay": delay, "coeff": coeff} for delay, coeff in kernel]
            for kernel in mic_row
        ]
        for mic_row in room
    ]
