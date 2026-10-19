import tempfile
import unittest
from pathlib import Path

from project_config import TASKS, resolve_task_paths, validate_existing_dir, validate_existing_file


class ProjectConfigTests(unittest.TestCase):
    def test_all_released_tasks_have_stable_checkpoint_names(self):
        self.assertEqual(
            {name: config.checkpoint_name for name, config in TASKS.items()},
            {
                "RoadScene": "weight_roadscene.pth",
                "M3FD": "weight_m3fd.pth",
                "MRI-PET": "weight_mri_pet.pth",
                "SPECT-MRI": "weight_spect_mri.pth",
            },
        )

    def test_task_defaults_can_be_overridden(self):
        data_path, weights_path = resolve_task_paths(
            "M3FD", data_path="custom/data", weights="custom/model.pth"
        )
        self.assertEqual(data_path, Path("custom/data"))
        self.assertEqual(weights_path, Path("custom/model.pth"))

    def test_task_defaults_point_to_data_and_checkpoint(self):
        data_path, weights_path = resolve_task_paths("MRI-PET")
        self.assertEqual(data_path, Path("data/MRI-PET/test"))
        self.assertEqual(weights_path, Path("checkpoint/weight_mri_pet.pth"))

    def test_validation_reports_missing_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "Dataset directory"):
                validate_existing_dir(root / "missing", "Dataset directory")
            with self.assertRaisesRegex(FileNotFoundError, "Checkpoint"):
                validate_existing_file(root / "missing.pth", "Checkpoint")


if __name__ == "__main__":
    unittest.main()

