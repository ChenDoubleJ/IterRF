from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class TaskConfig:
    data_folder: str
    checkpoint_name: str


TASKS: Dict[str, TaskConfig] = {
    "RoadScene": TaskConfig("RoadScene", "weight_roadscene.pth"),
    "M3FD": TaskConfig("M3FD", "weight_m3fd.pth"),
    "MRI-PET": TaskConfig("MRI-PET", "weight_mri_pet.pth"),
    "SPECT-MRI": TaskConfig("SPECT-MRI", "weight_spect_mri.pth"),
}


def resolve_task_paths(
    task: str,
    data_path: Optional[str] = None,
    weights: Optional[str] = None,
) -> Tuple[Path, Path]:
    config = TASKS[task]
    resolved_data = Path(data_path) if data_path else Path("data") / config.data_folder / "test"
    resolved_weights = Path(weights) if weights else Path("checkpoint") / config.checkpoint_name
    return resolved_data, resolved_weights


def validate_existing_dir(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def validate_existing_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path

