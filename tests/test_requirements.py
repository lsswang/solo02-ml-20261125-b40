import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RequirementScenarioTest(unittest.TestCase):
    def test_verify_script(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "verify.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            self.fail(f"verify.py failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        self.assertIn("ALL", result.stdout)
        self.assertIn("CHECKS PASSED", result.stdout)


if __name__ == "__main__":
    unittest.main()
