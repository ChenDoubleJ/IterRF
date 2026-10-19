import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_help(self, script):
        return subprocess.run(
            [sys.executable, str(ROOT / script), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_train_help_works_without_importing_torch(self):
        result = self.run_help("train.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--data-path-train", result.stdout)
        self.assertIn("--weights", result.stdout)

    def test_test_help_lists_all_pretrained_tasks(self):
        result = self.run_help("test.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        for task in ("RoadScene", "M3FD", "MRI-PET", "SPECT-MRI"):
            self.assertIn(task, result.stdout)
        self.assertIn("--output-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()

