import os
import argparse

import torch.nn as nn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import json
import pickle
import random
from scipy.signal import convolve2d
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# import numpy as np
import math
import cv2
from loss_function import losses
from metrics import calculate_psnr, calculate_ssim
import kornia.utils as KU
import torch.nn as nn
from dataset import Registration_Train_Data, Registration_Test_Data
from utils import data_transform, inverse_data_transform, create_lr_scheduler, train_one_epoch, evaluate
import datetime
from model.fusionnet import FusionNet
from model.registrationnet import RegistrationNet
from model.DUN_net import DUNnet
from model.alpha_beta_net import alpha_beta_net
from model.registration_net import registration_net
from model.encoder import AE_Encoder
class SpatialTransformer(nn.Module):
    def __init__(self, h, w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h, w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        if disp.shape[1] == 2:
            disp = disp.permute(0, 2, 3, 1)
        if disp.shape[1] != self.grid.shape[1] or disp.shape[2] != self.grid.shape[2]:
            self.grid = KU.create_meshgrid(disp.shape[1], disp.shape[2]).to(disp.device)
        flow = self.grid + disp
        return F.grid_sample(src, flow, mode=self.mode, align_corners=False), flow
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_dataset = Registration_Test_Data(data_path=args.data_path_test)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=False,
                                                num_workers=1)
    model_weight_path = args.weights
    encoder = None
    fusionnet = FusionNet(64).to(device)
    registration = registration_net().to(device)
    alpha_beta = alpha_beta_net().to(device)
    fusionnet.load_state_dict(torch.load(model_weight_path)['fusionnet'])
    registration.load_state_dict(torch.load(model_weight_path)['registration'])
    alpha_beta.load_state_dict(torch.load(model_weight_path)['alpha_beat'])
    fusionnet = fusionnet.eval()
    registration = registration.eval()
    alpha_beta = alpha_beta.eval()
    dunnet = DUNnet(args.eta1, args.eta2, fusionnet, registration, encoder, device)
    best_epoch = torch.load(model_weight_path)['epoch']
    with torch.no_grad():
        for step, data in  enumerate(test_loader):
                image_ir, image_vi, image_vi_warp, name = data
                tensors = [image_ir, image_vi, image_vi_warp]
                tensors = [tensor.to(device).detach() for tensor in tensors]
                tensors = [tensor.squeeze(1) if len(tensor.shape) > 4 else tensor for tensor in tensors]
                (image_ir, image_vi, image_vi_warp) = tensors
                image_ir = image_ir.to(device)
                image_vi = image_vi.to(device)
                image_vi_warp = image_vi_warp.to(device)
                image_vi_warp_y, _,_ = RGB2YCrCb(image_vi_warp)
                image_ir_y, _,_ = RGB2YCrCb(image_ir)
                # image_vi_y, _,_ = RGB2YCrCb(image_vi)

                eva_vi_y = image_vi_warp_y
                alpha0 = (image_ir_y < eva_vi_y).float()
                beta0 = 1 - alpha0
                eva_F_y = alpha0 * image_vi_warp_y + beta0 * image_ir_y
                P = eva_F_y
                Q = eva_vi_y
                eva_vi_rgb = image_vi_warp
                ir_warp_fake_rgb = image_ir
                ir_warp_fake_y = image_ir_y
                ST = SpatialTransformer(image_ir.shape[2], image_ir.shape[3], True)
                stage_values = {
                "P": [P.detach().clone()],
                "Q": [Q.detach().clone()],
                "eva_vi_y": [eva_vi_y.detach().clone()],
                "eva_F_y": [eva_F_y.detach().clone()]
}
                for i in range(args.stage):
                    alpha, beta = alpha_beta(eva_vi_y, image_ir_y)
                # eva_vi_y, eva_F_y, P, Q, alpha, beta, eva_defor,eva_vi_rgb, ir_warp_fake_rgb, eva_F_rgb  = dunnet(image_ir_y, eva_vi_y, eva_F_y, P, Q, alpha, beta, eva_vi_rgb, ir_warp_fake_rgb)
                    eva_vi_y, eva_F_y, P, Q, alpha, beta, eva_defor1, eva_defor2, eva_vi_rgb, ir_warp_fake_y, ir_warp_fake_rgb, eva_F_rgb  = dunnet(image_ir_y,image_vi_warp_y, eva_vi_y, eva_F_y, P, Q, alpha, beta, eva_vi_rgb, ir_warp_fake_y, ir_warp_fake_rgb, ST)



                # print(eva_defor1['vis2ir'].size())
                eva_vi = tensor2numpy_R(eva_vi_rgb)

                eva_F = tensor2numpy_R(eva_F_rgb)
                # ir_warp1 = tensor2numpy_R(ir_warp_fake_rgb)
            
                save_pic(eva_vi, name[0], flag = "eva_vi")
                save_pic(eva_F, name[0], flag = "eva_F")
                # save_pic(ir_warp1, evalfold_path, name[0] + "_ir_warp")
def save_pic(outputpic, name, flag):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]

    root = "./seu-results/"
    root_path = os.path.join(root, flag)

    if os.path.exists("./seu-results") is False:
        os.makedirs("./seu-results")
    if os.path.exists(root) is False:
        os.makedirs(root)
    if os.path.exists(root_path) is False:
        os.makedirs(root_path)
    path = root_path + "/{}".format(name) +".png"
    cv2.imwrite(path, outputpic)
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
    print("complete compute {}.png and save".format(name))


def tensor2numpy_R(R_tensor):
    R = R_tensor.squeeze(0).cpu().detach().numpy()
    R = np.transpose(R, [1, 2, 0])
    return R

def tensor2numpy_L(L_tensor):
    L = L_tensor.squeeze(0)
    L_3 = torch.cat([L, L, L], dim=0)
    L_3 = L_3.cpu().detach().numpy()
    L_3 = np.transpose(L_3, [1, 2, 0])
    return L_3
def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--eta1', type=float, default=0.1)
    parser.add_argument('--eta2', type=float, default=0.1)
    parser.add_argument('--stage', type=float, default=3)
    # parser.add_argument('--data_path_train', type=str,
    #                     default="data/RealScene/train")
    # parser.add_argument('--data_path_test', type=str,
    #                     default="data/RealScene/test")
    parser.add_argument('--data_path_train', type=str,
                        default="data/RoadScene/test")
    parser.add_argument('--data_path_test', type=str,
                        default="data/RoadScene/test")
    parser.add_argument('--weights', type=str, default='/home/xuhan/jiajunchen/Reg_Fusion/Registration_Fusion/experiments/best/weights/checkpoint_1400.pth',
                        help='test weights path')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default=False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--gpu_id', default='2', help='device id (i.e. 0, 1, 2 or 3)')
    opt = parser.parse_args()

    main(opt)
