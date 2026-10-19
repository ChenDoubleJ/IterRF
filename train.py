import argparse
import datetime
import os
from pathlib import Path

from checkpoint_utils import get_alpha_beta_state
from project_config import TASKS, validate_existing_dir, validate_existing_file


def build_parser():
    parser = argparse.ArgumentParser(description="Train IterRF on paired multi-modal images.")
    parser.add_argument("--task", choices=TASKS, default="RoadScene")
    parser.add_argument("--data-path-train", help="Training split containing ir/, vi/, warp_ir/, warp_vi/, disp/.")
    parser.add_argument("--data-path-test", help="Validation split containing ir/, vi/, and warp_vi/.")
    parser.add_argument("--weights", help="Optional model-only initialization checkpoint.")
    parser.add_argument("--resume", help="Checkpoint used to resume model, optimizer, scheduler, and epoch.")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta1", type=float, default=0.1)
    parser.add_argument("--eta2", type=float, default=0.1)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu-id", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser


def _load_model_state(checkpoint, fusionnet, registration, alpha_beta):
    fusionnet.load_state_dict(checkpoint["fusionnet"])
    registration.load_state_dict(checkpoint["registration"])
    alpha_beta.load_state_dict(get_alpha_beta_state(checkpoint))


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    import torch
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    from dataset import Registration_Test_Data, Registration_Train_Data
    from model.DUN_net import DUNnet
    from model.alpha_beta_net import alpha_beta_net
    from model.fusionnet import FusionNet
    from model.registration_net import registration_net
    from utils import create_lr_scheduler, evaluate, train_one_epoch

    task_root = Path("data") / args.task
    train_path = Path(args.data_path_train) if args.data_path_train else task_root / "train"
    test_path = Path(args.data_path_test) if args.data_path_test else task_root / "test"
    validate_existing_dir(train_path, "Training dataset directory")
    validate_existing_dir(test_path, "Validation dataset directory")
    if args.weights:
        validate_existing_file(Path(args.weights), "Initialization checkpoint")
    if args.resume:
        validate_existing_file(Path(args.resume), "Resume checkpoint")
    if min(args.epochs, args.batch_size, args.stage, args.val_interval, args.save_interval) < 1:
        raise ValueError("Epoch, batch, stage, validation, and save values must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or --device auto.")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    run_name = datetime.datetime.now().strftime(f"{args.task}_%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    image_dir = run_dir / "images"
    weights_dir = run_dir / "weights"
    log_dir = run_dir / "log"
    for directory in (image_dir, weights_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    train_dataset = Registration_Train_Data(train_path)
    test_dataset = Registration_Test_Data(test_path)
    workers = args.num_workers
    if workers is None:
        workers = min(os.cpu_count() or 1, args.batch_size if args.batch_size > 1 else 0, 8)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=workers,
    )

    fusionnet = FusionNet(64).to(device)
    registration = registration_net().to(device)
    alpha_beta = alpha_beta_net().to(device)
    dunnet = DUNnet(args.eta1, args.eta2, fusionnet, registration, None, device)
    parameters = [
        parameter
        for parameter in list(fusionnet.parameters())
        + list(registration.parameters())
        + list(alpha_beta.parameters())
        if parameter.requires_grad
    ]
    optimizer = optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-5)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    start_epoch = 0
    best_psnr = float("-inf")
    best_ssim = float("-inf")
    if args.weights:
        _load_model_state(torch.load(args.weights, map_location=device), fusionnet, registration, alpha_beta)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        _load_model_state(checkpoint, fusionnet, registration, alpha_beta)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint.get("best_psnr", best_psnr)
        best_ssim = checkpoint.get("best_ssim", best_ssim)

    def save_checkpoint(path, epoch):
        torch.save(
            {
                "fusionnet": fusionnet.state_dict(),
                "registration": registration.state_dict(),
                "alpha_beta": alpha_beta.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "args": vars(args),
            },
            path,
        )

    try:
        for epoch in range(start_epoch, args.epochs):
            values = train_one_epoch(
                fusionnet=fusionnet,
                registration=registration,
                alpha_beta=alpha_beta,
                dunnet=dunnet,
                encoder=None,
                optimizer=optimizer,
                data_loader=train_loader,
                lr_scheduler=scheduler,
                stage=args.stage,
                device=device,
                epoch=epoch,
            )
            metric_names = (
                "total_loss", "loss_ssim", "loss_max", "loss_reg_field", "loss_grad",
                "loss_reg_img", "loss_border_re", "loss_smoothness", "loss_color", "lr",
            )
            for name, value in zip(metric_names, values):
                writer.add_scalar(f"Train/{name}", value, epoch)

            if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
                psnr, ssim = evaluate(
                    fusionnet=fusionnet,
                    registration=registration,
                    alpha_beta=alpha_beta,
                    dunnet=dunnet,
                    encoder=None,
                    optimizer=optimizer,
                    data_loader=test_loader,
                    stage=args.stage,
                    device=device,
                    epoch=epoch,
                    lr_scheduler=scheduler,
                    filefold_path=str(image_dir),
                )
                writer.add_scalar("Validation/psnr", psnr, epoch)
                writer.add_scalar("Validation/ssim", ssim, epoch)
                if psnr > best_psnr or ssim > best_ssim:
                    best_psnr = max(best_psnr, psnr)
                    best_ssim = max(best_ssim, ssim)
                    save_checkpoint(weights_dir / "checkpoint_best.pth", epoch)

            if epoch % args.save_interval == 0 or epoch == args.epochs - 1:
                save_checkpoint(weights_dir / f"checkpoint_{epoch}.pth", epoch)
    finally:
        writer.close()


if __name__ == "__main__":
    main(build_parser().parse_args())

