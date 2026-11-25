"""PCM-16 WAV helpers used by the synthetic bench.

Audio is represented as a list of per-channel float lists.  All functions are
small and synchronous; no NumPy is required.
"""

from __future__ import annotations

import array
import os
import wave
from typing import List, Sequence, Tuple

Audio = List[List[float]]


def read_wav(path: str) -> Tuple[int, Audio]:
    """Read a PCM WAV file and return ``(sample_rate, channels)``."""
    with wave.open(path, "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth != 2:
        raise ValueError(f"{path}: only 16-bit PCM is supported, got width {sampwidth}")

    samples = array.array("h")
    samples.frombytes(raw)
    if samples.itemsize != 2:  # pragma: no cover - normal Linux/x86 path
        samples.byteswap()

    channels: Audio = [[] for _ in range(nchannels)]
    for i, value in enumerate(samples):
        channels[i % nchannels].append(value / 32768.0)

    return sample_rate, channels


def write_wav(path: str, sample_rate: int, channels: Sequence[Sequence[float]]) -> None:
    """Write mono/stereo/multichannel float audio as clipped PCM-16."""
    if not channels:
        raise ValueError("at least one channel is required")
    nframes = max(len(ch) for ch in channels)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(len(channels))
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        pcm = array.array("h")
        for i in range(nframes):
            for ch in channels:
                value = ch[i] if i < len(ch) else 0.0
                if value > 1.0:
                    value = 1.0
                elif value < -1.0:
                    value = -1.0
                pcm.append(int(round(value * 32767.0)))
        wf.writeframes(pcm.tobytes())


def stereo(x1: Sequence[float], x2: Sequence[float]) -> Audio:
    n = max(len(x1), len(x2))
    return [
        list(x1) + [0.0] * (n - len(x1)),
        list(x2) + [0.0] * (n - len(x2)),
    ]
