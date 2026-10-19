import tempfile
import unittest
from pathlib import Path

from dataset import discover_samples


class DatasetLayoutTests(unittest.TestCase):
    def test_test_split_matches_files_by_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("ir", "vi", "warp_vi"):
                (root / folder).mkdir()
                (root / folder / "sample.png").touch()

            samples = discover_samples(root, training=False)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].name, "sample")

    def test_training_split_requires_displacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("ir", "vi", "warp_ir", "warp_vi", "disp"):
                (root / folder).mkdir()
            for folder in ("ir", "vi", "warp_ir", "warp_vi"):
                (root / folder / "sample.jpg").touch()

            with self.assertRaisesRegex(FileNotFoundError, "disp"):
                discover_samples(root, training=True)


if __name__ == "__main__":
    unittest.main()

