from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class SamplePaths:
    name: str
    ir: Path
    vi: Path
    warp_vi: Path
    warp_ir: Optional[Path] = None
    disp: Optional[Path] = None


def _files_by_stem(folder: Path, extensions=IMAGE_EXTENSIONS) -> Dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    files = {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in extensions
    }
    if not files:
        raise FileNotFoundError(f"No supported files found in: {folder}")
    return files


def discover_samples(data_path, training: bool) -> List[SamplePaths]:
    root = Path(data_path)
    ir = _files_by_stem(root / "ir")
    vi = _files_by_stem(root / "vi")
    warp_vi = _files_by_stem(root / "warp_vi")
    required = {"vi": vi, "warp_vi": warp_vi}

    warp_ir = None
    disp = None
    if training:
        warp_ir = _files_by_stem(root / "warp_ir")
        disp = _files_by_stem(root / "disp", extensions=(".pt", ".pth"))
        required.update({"warp_ir": warp_ir, "disp": disp})

    missing = {
        folder: sorted(set(ir) - set(files))
        for folder, files in required.items()
        if set(ir) - set(files)
    }
    if missing:
        details = "; ".join(f"{folder}: {', '.join(names[:5])}" for folder, names in missing.items())
        raise FileNotFoundError(f"Unpaired dataset files ({details})")

    return [
        SamplePaths(
            name=name,
            ir=ir[name],
            vi=vi[name],
            warp_vi=warp_vi[name],
            warp_ir=warp_ir[name] if warp_ir else None,
            disp=disp[name] if disp else None,
        )
        for name in sorted(ir)
    ]


def _load_rgb(path: Path):
    import numpy as np
    import torch
    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1)).contiguous()


def _load_disp(path: Path):
    import torch

    disp = torch.load(path, map_location="cpu")
    if isinstance(disp, dict):
        for key in ("disp", "flow", "deformation"):
            if key in disp:
                disp = disp[key]
                break
    disp = torch.as_tensor(disp, dtype=torch.float32)
    if disp.ndim == 3 and disp.shape[-1] == 2:
        disp = disp.permute(2, 0, 1)
    if disp.ndim != 3 or disp.shape[0] != 2:
        raise ValueError(f"Expected displacement shape [2, H, W], got {tuple(disp.shape)} from {path}")
    return disp.contiguous()


class Registration_Test_Data:
    def __init__(self, data_path):
        self.samples = discover_samples(data_path, training=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return _load_rgb(sample.ir), _load_rgb(sample.vi), _load_rgb(sample.warp_vi), sample.name


class Registration_Train_Data:
    def __init__(self, data_path):
        self.samples = discover_samples(data_path, training=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (
            _load_rgb(sample.ir),
            _load_rgb(sample.vi),
            _load_rgb(sample.warp_ir),
            _load_rgb(sample.warp_vi),
            _load_disp(sample.disp),
        )

