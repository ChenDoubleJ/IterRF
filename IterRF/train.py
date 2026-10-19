import os
import argparse

import torch.nn as nn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from dataset import Registration_Train_Data, Registration_Test_Data
from utils import data_transform, inverse_data_transform, create_lr_scheduler, train_one_epoch, evaluate
import datetime
# from model.fusion_net import Fusionnet
# from model.fusionnet import FusionNet
from model.fusionnet import FusionNet
from model.registrationnet import RegistrationNet
from model.DUN_net import DUNnet
from model.alpha_beta_net import alpha_beta_net
from model.registration_net import registration_net
from model.encoder import AE_Encoder
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # torch.backends.cudnn.enabled = False

    if os.path.exists("./experiments") is False:
        os.makedirs("./experiments")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./experiments/Registration_and_Fusion_{}".format(file_name)
    os.makedirs(filefold_path)
    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    best_psnr = 1e-5
    best_ssim = 1e-5
    start_epoch = 0

    train_dataset = Registration_Train_Data(data_path=args.data_path_train)
    test_dataset = Registration_Test_Data(data_path=args.data_path_test)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=nw)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=nw)
    # encoder = AE_Encoder().to(device)
    # encoder.load_state_dict(torch.load('checkpoint/Encoder.pth'))
    # encoder.eval()
    # for p1 in encoder.parameters():
    #     p1.requires_grad = False
    encoder = None
    fusionnet = FusionNet(64).to(device)
    registration = registration_net().to(device)
    # alpha = nn.Sigmoid(nn.parameter())
    alpha_beta = alpha_beta_net().to(device)
    dunnet = DUNnet(args.eta1, args.eta2, fusionnet, registration, encoder, device)
    pg = [p for p in list(fusionnet.parameters()) + list(registration.parameters()) + list(alpha_beta.parameters()) if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5E-5)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    total_params = sum(p.numel() for p in list(fusionnet.parameters()) + list(registration.parameters()) + list(alpha_beta.parameters()))
    print(f'Total number of parameters: {total_params}')

    for epoch in range(start_epoch, args.epochs):
        # train
        # train_loss, train_loss_ssim, train_loss_max, \
        # train_loss_color, train_loss_grad, train_loss_l1, train_loss_smooth, lr = train_one_epoch(fusionnet=fusionnet,
        #                                                                 registration=registration,
        #                                                                 alpha_beta = alpha_beta,
        #                                                                 dunnet=dunnet,
        #                                                                 optimizer=optimizer,
        #                                                                 data_loader=train_loader,
        #                                                                 lr_scheduler=lr_scheduler,
        #                                                                 stage=args.stage,
        #                                                                 device=device,
        #                                                                 epoch=epoch)
        total_loss, loss_ssim, loss_max, loss_reg_field, loss_grad, loss_reg_img, loss_border_re, loss_smoothness, loss_color, lr = train_one_epoch(fusionnet=fusionnet,
                                    registration=registration,
                                    alpha_beta = alpha_beta,
                                    dunnet=dunnet,
                                    encoder=encoder,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    lr_scheduler=lr_scheduler,
                                    stage=args.stage,
                                    device=device,
                                    epoch=epoch)
        tb_writer.add_scalar("Train/total_loss", total_loss, epoch)
        tb_writer.add_scalar("Train/loss_ssim", loss_ssim, epoch)
        tb_writer.add_scalar("Train/loss_max", loss_max, epoch)
        tb_writer.add_scalar("Train/loss_reg_field", loss_reg_field, epoch)
        tb_writer.add_scalar("Train/loss_grad", loss_grad, epoch)
        tb_writer.add_scalar("Train/loss_reg_img", loss_reg_img, epoch)
        tb_writer.add_scalar("Train/loss_smoothness", loss_smoothness, epoch)
        tb_writer.add_scalar("Train/loss_border_re", loss_border_re, epoch)
        tb_writer.add_scalar("Train/loss_color", loss_color, epoch)
        tb_writer.add_scalar("Train/train_lr", lr, epoch)
        # validate
        val_epoch = -1
        if epoch == 0 or epoch % val_epoch == 0 or epoch == 999: 
            psnr, ssim = evaluate(fusionnet=fusionnet,
                    registration=registration,
                     alpha_beta = alpha_beta,
                    dunnet=dunnet,
                    encoder=encoder,
                    optimizer=optimizer,
                    data_loader=test_loader,
                    stage=args.stage,
                    device=device,
                    epoch=epoch, lr_scheduler=lr_scheduler, filefold_path=file_img_path)

        if psnr > best_psnr or ssim > best_ssim:
            print(f'-----------{epoch}----------------')
            best_psnr = psnr
            best_ssim = ssim
            save_file = {"fusionnet": fusionnet.state_dict(),
                            "registration": registration.state_dict(),
                            "alpha_beat": alpha_beta.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args}
            torch.save(save_file, file_weights_path + "/" + "checkpoint.pth")
        if epoch % 100 == 0 or epoch == args.epochs -1:
            save_file = {"fusionnet": fusionnet.state_dict(),
                            "registration": registration.state_dict(),
                            "alpha_beat": alpha_beta.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args}
            torch.save(save_file, file_weights_path + "/" + f"checkpoint_{epoch}.pth")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--eta1', type=float, default=0.1)
    parser.add_argument('--eta2', type=float, default=0.1)
    parser.add_argument('--stage', type=float, default=3)
    parser.add_argument('--data_path_train', type=str,
                        default="data/RoadScene/train")
    parser.add_argument('--data_path_test', type=str,
                        default="data/RoadScene/test")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default=False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--gpu_id', default='6', help='device id (i.e. 0, 1, 2 or 3)')
    opt = parser.parse_args()

    main(opt)
