import argparse
import os
from pathlib import Path

from checkpoint_utils import get_alpha_beta_state
from project_config import TASKS, resolve_task_paths, validate_existing_dir, validate_existing_file


def build_parser():
    parser = argparse.ArgumentParser(description="Run IterRF registration and fusion inference.")
    parser.add_argument("--task", choices=TASKS, default="RoadScene")
    parser.add_argument("--data-path", help="Test split containing ir/, vi/, and warp_vi/.")
    parser.add_argument("--weights", help="Checkpoint path. Defaults to checkpoint/<task weight>.")
    parser.add_argument("--output-dir", default="results", help="Directory for registered and fused images.")
    parser.add_argument("--stage", type=int, default=3, help="Number of IterRF refinement stages.")
    parser.add_argument("--eta1", type=float, default=0.1)
    parser.add_argument("--eta2", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu-id", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def _save_image(tensor, path: Path):
    import cv2
    import numpy as np

    image = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    image = np.rint(image * 255.0).astype(np.uint8)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write output image: {path}")


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import kornia.utils as KU

    from dataset import Registration_Test_Data
    from model.DUN_net import DUNnet
    from model.alpha_beta_net import alpha_beta_net
    from model.fusionnet import FusionNet
    from model.registration_net import registration_net

    data_path, weights_path = resolve_task_paths(args.task, args.data_path, args.weights)
    validate_existing_dir(data_path, "Dataset directory")
    validate_existing_file(weights_path, "Checkpoint")
    if args.stage < 1:
        raise ValueError("--stage must be at least 1")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or --device auto.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    class SpatialTransformer(nn.Module):
        def __init__(self, height, width):
            super().__init__()
            self.register_buffer("grid", KU.create_meshgrid(height, width, normalized_coordinates=True))

        def forward(self, source, displacement):
            if displacement.shape[1] == 2:
                displacement = displacement.permute(0, 2, 3, 1)
            if displacement.shape[1:3] != self.grid.shape[1:3]:
                self.grid = KU.create_meshgrid(
                    displacement.shape[1], displacement.shape[2], normalized_coordinates=True
                ).to(displacement.device)
            flow = self.grid + displacement
            return F.grid_sample(source, flow, mode="bilinear", align_corners=False), flow

    loader = torch.utils.data.DataLoader(
        Registration_Test_Data(data_path),
        batch_size=1,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
    )

    fusionnet = FusionNet(64).to(device)
    registration = registration_net().to(device)
    alpha_beta = alpha_beta_net().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    fusionnet.load_state_dict(checkpoint["fusionnet"])
    registration.load_state_dict(checkpoint["registration"])
    alpha_beta.load_state_dict(get_alpha_beta_state(checkpoint))
    fusionnet.eval()
    registration.eval()
    alpha_beta.eval()
    dunnet = DUNnet(args.eta1, args.eta2, fusionnet, registration, None, device)

    output_dir = Path(args.output_dir) / args.task
    with torch.no_grad():
        for image_ir, _image_vi, image_vi_warp, names in loader:
            image_ir = image_ir.to(device)
            image_vi_warp = image_vi_warp.to(device)
            image_vi_warp_y, _, _ = dunnet.RGB2YCrCb(image_vi_warp)
            image_ir_y, _, _ = dunnet.RGB2YCrCb(image_ir)
            eva_vi_y = image_vi_warp_y
            alpha = (image_ir_y < eva_vi_y).float()
            beta = 1 - alpha
            eva_f_y = alpha * image_vi_warp_y + beta * image_ir_y
            p_value = eva_f_y
            q_value = eva_vi_y
            eva_vi_rgb = image_vi_warp
            ir_warp_y = image_ir_y
            ir_warp_rgb = image_ir
            transformer = SpatialTransformer(image_ir.shape[2], image_ir.shape[3]).to(device)

            for _ in range(args.stage):
                alpha, beta = alpha_beta(eva_vi_y, image_ir_y)
                (
                    eva_vi_y,
                    eva_f_y,
                    p_value,
                    q_value,
                    alpha,
                    beta,
                    _defor1,
                    _defor2,
                    eva_vi_rgb,
                    ir_warp_y,
                    ir_warp_rgb,
                    eva_f_rgb,
                ) = dunnet(
                    image_ir_y,
                    image_vi_warp_y,
                    eva_vi_y,
                    eva_f_y,
                    p_value,
                    q_value,
                    alpha,
                    beta,
                    eva_vi_rgb,
                    ir_warp_y,
                    ir_warp_rgb,
                    transformer,
                )

            name = names[0]
            _save_image(eva_vi_rgb, output_dir / "registered" / f"{name}.png")
            _save_image(eva_f_rgb, output_dir / "fused" / f"{name}.png")
            print(f"Saved {name} to {output_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())

