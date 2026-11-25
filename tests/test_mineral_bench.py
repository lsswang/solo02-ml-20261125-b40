"""Standard-library unittest wrapper.

Run with either:
    python run.py selftest
or:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mineral_bench.selftest import (
    test_minimal_optimal_rate,
    test_persistent_end_to_end_and_exit_status,
    test_pipeline_outputs_for_another_input,
    test_textbook_posteriors,
)


class MineralBenchTests(unittest.TestCase):
    def test_posteriors(self) -> None:
        test_textbook_posteriors()

    def test_minimal_rate(self) -> None:
        test_minimal_optimal_rate()

    def test_persistent_case(self) -> None:
        test_persistent_end_to_end_and_exit_status()

    def test_replacement_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            test_pipeline_outputs_for_another_input(Path(td))


if __name__ == "__main__":
    unittest.main()
