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
def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
    # return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,1000, eta_min=1e-6)

# def save_pic(outputpic, path, index : str):
#     outputpic[outputpic > 1.] = 1
#     outputpic[outputpic < 0.] = 0
#     outputpic = cv2.UMat(outputpic).get()
#     outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
#     outputpic=outputpic[:, :, ::-1]
#     save_path = os.path.join(path, index + ".png")
#     cv2.imwrite(save_path, outputpic)

# def tensor2numpy_R(R_tensor):
#     R = R_tensor.squeeze(0).cpu().detach().numpy()
#     R = np.transpose(R, [1, 2, 0])
#     return R

# def tensor2numpy_L(L_tensor):
#     L = L_tensor.squeeze(0)
#     L_3 = torch.cat([L, L, L], dim=0)
#     L_3 = L_3.cpu().detach().numpy()
#     L_3 = np.transpose(L_3, [1, 2, 0])
#     return L_3
# def rgb_to_ycbcr(image):
#         r = image[:, :1, :, :]
#         g = image[:, 1:2, :, :]
#         b = image[:, 2:3, :, :]

#         y = 0.299 * r + 0.587 * g + 0.114 * b
#         cb = -0.168736 * r - 0.331264 * g + 0.5 * b
#         cr = 0.5 * r - 0.418688 * g - 0.081312 * b

#         # ycbcr_image = torch.stack((y, cb, cr), dim=1)
#         return y, cb, cr
def rgb_to_ycbcr(image):
    r = image[:, 0:1, :, :]
    g = image[:, 1:2, :, :]
    b = image[:, 2:3, :, :]

    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128/255.0
    cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 128/255.0

    return y, cb, cr   # 注意顺序与OpenCV对齐
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
def generate_mask(ST, disp, border_mask):
    AP = nn.AvgPool2d(5, stride=1, padding=2)
    flow = ST.grid + disp
    goodmask = torch.logical_and(flow >= -1, flow <= 1)
    # if self.border_mask.device != goodmask.device:
    border_mask = border_mask.to(goodmask.device)
    goodmask = torch.logical_and(goodmask[..., 0], goodmask[..., 1]).unsqueeze(1) * 1.0
    for i in range(2):
        goodmask = (AP(goodmask) > 0.3).float()
    flow = ST.grid - disp
    goodmask0 = F.grid_sample(goodmask, flow)
    goodmask_inverse = goodmask0
    return goodmask_inverse, goodmask, border_mask
import os
import torch
import matplotlib.pyplot as plt

# 保存函数
def save_first_sample(image_ir_warp, image_vi_warp, goodmask_inverse, goodmask, border_mask, save_dir="./vis_results"):
    os.makedirs(save_dir, exist_ok=True)

    # 取 batch 中第一个样本
    ir = image_ir_warp[0].detach().cpu()
    vi = image_vi_warp[0].detach().cpu()
    ginv = goodmask_inverse[0].detach().cpu()
    g = goodmask[0].detach().cpu()
    b = border_mask[0].detach().cpu()

    # 如果是 (C,H,W)，转成 (H,W,C)
    if ir.dim() == 3 and ir.shape[0] in [1, 3]:
        ir = ir.permute(1, 2, 0)
    if vi.dim() == 3 and vi.shape[0] in [1, 3]:
        vi = vi.permute(1, 2, 0)

    # 归一化到 [0,1]
    def norm(x):
        x = x.float()
        if x.max() > 1.0:
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x

    ir = norm(ir)
    vi = norm(vi)

    # 保存 image_ir_warp
    plt.imsave(os.path.join(save_dir, "image_ir_warp.png"), ir.numpy(), cmap="gray" if ir.shape[-1]==1 else None)
    # 保存 image_vi_warp
    plt.imsave(os.path.join(save_dir, "image_vi_warp.png"), vi.numpy(), cmap="gray" if vi.shape[-1]==1 else None)

    # 保存 mask，mask=1 白色，mask=0 黑色
    for mask, name in zip([ginv, g, b], ["goodmask_inverse", "goodmask", "border_mask"]):
        mask_img = (mask.squeeze().numpy() > 0.5).astype("uint8") * 255  # 二值化后变 0/255
        plt.imsave(os.path.join(save_dir, f"{name}.png"), mask_img, cmap="gray")

    print(f"保存完成，结果在 {save_dir}")

# 调用示例
def gradient_magnitude(img):
    # 计算梯度幅值
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)


def train_one_epoch(fusionnet, registration, alpha_beta, dunnet, encoder, optimizer, lr_scheduler, data_loader, stage, device, epoch):
    fusionnet = fusionnet.train()
    registration = registration.train()
    # encoder = encoder.eval()
    ST = SpatialTransformer(256, 256, True)
    loss_function = losses()
    border_mask = torch.zeros([1, 1, 256,256])
    border_mask[:, :, 10:-10, 10:-10] = 1
    
    if torch.cuda.is_available():
        loss_function = loss_function.to(device)


    total_loss = torch.zeros(1).to(device)
    total_loss_ssim = torch.zeros(1).to(device)
    total_loss_max = torch.zeros(1).to(device)
    # loss_color = torch.zeros(1).to(device)
    total_loss_grad = torch.zeros(1).to(device)
    # loss_l1 = torch.zeros(1).to(device)
    total_loss_smoothness = torch.zeros(1).to(device)
    # total_loss_smoothness = torch.zeros(1).to(device)
    total_reg_img = torch.zeros(1).to(device)
    total_reg_field = torch.zeros(1).to(device)
    total_border_re = torch.zeros(1).to(device)
    total_loss_color = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step,data in  enumerate(data_loader):
        # print(deformation.shape)
        image_ir, image_vi, image_ir_warp, image_vi_warp, deformation = data
        tensors = [image_ir, image_vi, image_ir_warp, image_vi_warp, deformation]
        tensors = [tensor.to(device).detach() for tensor in tensors]
        tensors = [tensor.squeeze(1) if len(tensor.shape) > 4 else tensor for tensor in tensors]
        (image_ir, image_vi, image_ir_warp, image_vi_warp, deformation) = tensors
        goodmask_inverse, goodmask,border_mask = generate_mask(ST, deformation, border_mask)
        # save_first_sample(image_ir_warp, image_vi_warp, goodmask_inverse, goodmask, border_mask, save_dir="./output_vis")
        # break
        # image_vi_y, image_vi_cb, image_vi_cr = rgb_to_ycbcr(image_vi) 
        # image_vi = torch.cat((image_vi_y, image_vi_y, image_vi_y), dim=1)
        image_vi_warp_y, image_vi_warp_b, image_vi_warp_r = RGB2YCrCb(image_vi_warp)
        image_ir_y, image_ir_b, image_ir_r = RGB2YCrCb(image_ir)

        eva_vi_y = image_vi_warp_y
        alpha0 = (image_ir_y < eva_vi_y).float()
        # beta0 = 1 - alpha0
        # 计算梯度幅值
        # grad_ir = gradient_magnitude(image_ir_y)
        # grad_vi = gradient_magnitude(eva_vi_y)

        # # 在梯度更大的位置取对应掩码
        # alpha0 = (grad_ir < grad_vi).float()
        beta0 = 1 - alpha0
        eva_F_y = alpha0 * image_vi_warp_y + beta0 * image_ir_y
        P = eva_F_y
        Q = eva_vi_y
        eva_vi_rgb = image_vi_warp
        ir_warp_fake_rgb = image_ir
        ir_warp_fake_y = image_ir_y
        sum_reg_img = torch.zeros(1).to(device)
        sum_reg_field = torch.zeros(1).to(device)
        sum_border_re = torch.zeros(1).to(device)
        sum_loss = torch.zeros(1).to(device)
        sum_ssim = torch.zeros(1).to(device)
        sum_max = torch.zeros(1).to(device)

        sum_grad = torch.zeros(1).to(device)

        sum_smooth = torch.zeros(1).to(device)
        sum_color = torch.zeros(1).to(device)
        ST = SpatialTransformer(image_ir.shape[2], image_ir.shape[3] , True)
        for i in range(stage):
            # grad_ir = gradient_magnitude(image_ir_y)
            # grad_vi = gradient_magnitude(eva_vi_y)
            # alpha = (grad_ir < grad_vi).float()
            # alpha = (image_ir_y < eva_vi_y).float()
            # beta = 1 - alpha
            alpha, beta = alpha_beta(eva_vi_y, image_ir_y)
            eva_vi_y, eva_F_y, P, Q, alpha, beta, eva_defor1, eva_defor2, eva_vi_rgb, ir_warp_fake_y, ir_warp_fake_rgb, eva_F_rgb  = dunnet(image_ir_y,image_vi_warp_y, eva_vi_y, eva_F_y, P, Q, alpha, beta, eva_vi_rgb, ir_warp_fake_y, ir_warp_fake_rgb, ST)

            
            # eva_warp_vi, _ = ST(eva_vi_y, deformation)
            # eva_warp_ir = F.grid_sample(eva_ir, deformation, align_corners=False, mode='bilinear')
            # loss, loss_ssim, loss_max, loss_color, loss_grad, loss_l1, loss_smooth  = loss_function(image_vi_warp, image_vi, image_ir, 
            #                                                                                         image_ir_warp, ir_warp_fake_rgb, 
            #                                                                                         eva_vi_y, eva_vi_rgb, eva_F_y, 
            #                                                                                         eva_F_rgb, eva_defor, goodmask_inverse, 
            #                                                                                         goodmask, deformation, border_mask)
            loss, loss_reg_img, loss_reg_field, loss_smooth, loss_border_re, loss_max, loss_ssim, loss_grad, loss_color = loss_function(image_vi_warp, image_vi, image_ir, 
                                                                                                                            image_ir_warp, ir_warp_fake_rgb, 
                                                                                                                            eva_vi_y, eva_vi_rgb, eva_F_y, 
                                                                                                                            eva_F_rgb, eva_defor1, eva_defor2, goodmask_inverse, 
                                                                                                                            goodmask, deformation, border_mask)
            sum_reg_img += loss_reg_img
            sum_reg_field += loss_reg_field
            sum_border_re += loss_border_re
            sum_loss += loss
            sum_ssim += loss_ssim
            sum_max += loss_max
            # sum_color += loss_color
            sum_grad += loss_grad
            # sum_l1 += loss_l1
            sum_smooth += loss_smooth
            sum_color += loss_color

        sum_loss.backward()
        # print(sum_ssim)
        # print(total_loss)
        total_loss += sum_loss.detach().item()
        total_loss_ssim += sum_ssim.detach().item()
        total_loss_max += sum_max.detach().item()
        # loss_color += sum_color.detach().item()
        total_loss_grad += sum_grad.detach().item()
        # loss_l1 += sum_l1.detach().item()
        total_loss_smoothness += sum_smooth.detach().item()
        total_reg_img += sum_reg_img.detach().item()
        total_reg_field += sum_reg_field.detach().item()
        total_border_re += sum_border_re.detach().item()
        total_loss_color += sum_color.detach().item()
        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  loss_ssim: {:.3f}  loss_max: {:.3f}  loss_reg_field: {:.3f} loss_grad: {:.3f} loss_reg_img: {:.3f}  loss_smooth: {:.3f} loss_border_re: {:.3f} loss_color : {:.3f} lr: {:.6f}".format(epoch, total_loss.item() / (step + 1), 
            total_loss_ssim.item() / (step + 1), total_loss_max.item() / (step + 1), total_reg_field.item() / (step + 1), 
            total_loss_grad.item() / (step + 1), total_reg_img.item() / (step + 1), total_border_re.item() / (step + 1), 
            total_loss_smoothness.item() / (step + 1),  total_loss_color.item() / (step + 1), lr)

        # data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, sum_loss.item() / (step + 1))


        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()

        optimizer.zero_grad()

    lr_scheduler.step()
    
    # return total_loss.item() / (step + 1), loss_ssim.item() / (step + 1), loss_max.item() / (step + 1), loss_color.item() / (step + 1), loss_grad.item() / (step + 1), loss_l1.item() / (step + 1), loss_smoothness.item() / (step + 1), lr
    return total_loss.item() / (step + 1), total_loss_ssim.item() / (step + 1), total_loss_max.item() / (step + 1), total_reg_field.item() / (step + 1), total_loss_grad.item() / (step + 1), total_reg_img.item() / (step + 1), total_border_re.item() / (step + 1), total_loss_smoothness.item() / (step + 1), total_loss_color.item() / (step + 1),lr

@torch.no_grad()
def evaluate(fusionnet, registration, alpha_beta, dunnet,encoder, optimizer, lr_scheduler, data_loader, stage, device, epoch, filefold_path):
    fusionnet = fusionnet.eval()
    registration = registration.eval()

    save_epoch = 30

    if epoch % save_epoch == 0 or epoch == 999:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    avg_psnr_val = 0.0
    avg_ssim_val = 0.0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in  enumerate(data_loader):
        image_ir, image_vi, image_vi_warp, name = data
        tensors = [image_ir, image_vi, image_vi_warp]
        tensors = [tensor.to(device).detach() for tensor in tensors]
        tensors = [tensor.squeeze(1) if len(tensor.shape) > 4 else tensor for tensor in tensors]
        (image_ir, image_vi, image_vi_warp) = tensors
        image_ir = image_ir.to(device)
        image_vi = image_vi.to(device)
        image_vi_warp = image_vi_warp.to(device)
        # goodmask_inverse, goodmask,border_mask = generate_mask(ST, deformation)
        # image_vi_y, image_vi_cb, image_vi_cr = rgb_to_ycbcr(image_vi) 
        # image_vi = torch.cat((image_vi_y, image_vi_y, image_vi_y), dim=1)
        image_vi_warp_y, image_vi_warp_b, image_vi_warp_r = RGB2YCrCb(image_vi_warp)
        image_ir_y, image_ir_b, image_ir_r = RGB2YCrCb(image_ir)
        image_vi_y, image_vi_b, image_vi_r = RGB2YCrCb(image_vi)

        eva_vi_y = image_vi_warp_y
        alpha0 = (image_ir_y < eva_vi_y).float()
        # grad_ir = gradient_magnitude(image_ir_y)
        # grad_vi = gradient_magnitude(eva_vi_y)
        # alpha0 = (grad_ir < grad_vi).float()
        beta0 = 1 - alpha0
        eva_F_y = alpha0 * image_vi_warp_y + beta0 * image_ir_y
        P = eva_F_y
        Q = eva_vi_y
        eva_vi_rgb = image_vi_warp
        ir_warp_fake_rgb = image_ir
        ir_warp_fake_y = image_ir_y
        # sum_loss = torch.zeros(1).to(device)
        # sum_ssim = torch.zeros(1).to(device)
        # sum_max = torch.zeros(1).to(device)
        # sum_color = torch.zeros(1).to(device)
        # sum_grad = torch.zeros(1).to(device)
        # sum_l1 = torch.zeros(1).to(device)
        # sum_smooth = torch.zeros(1).to(device)
        ST = SpatialTransformer(image_ir.shape[2], image_ir.shape[3], True)
        for i in range(stage):
            # grad_ir = gradient_magnitude(image_ir_y)
            # grad_vi = gradient_magnitude(eva_vi_y)
            # alpha = (grad_ir < grad_vi).float()
            # alpha = (image_ir_y < eva_vi_y).float()
            # beta = 1 - alpha
            alpha, beta = alpha_beta(eva_vi_y, image_ir_y)
           # eva_vi_y, eva_F_y, P, Q, alpha, beta, eva_defor,eva_vi_rgb, ir_warp_fake_rgb, eva_F_rgb  = dunnet(image_ir_y, eva_vi_y, eva_F_y, P, Q, alpha, beta, eva_vi_rgb, ir_warp_fake_rgb)
            eva_vi_y, eva_F_y, P, Q, alpha, beta, eva_defor1, eva_defor2, eva_vi_rgb, ir_warp_fake_y, ir_warp_fake_rgb, eva_F_rgb  = dunnet(image_ir_y,image_vi_warp_y, eva_vi_y, eva_F_y, P, Q, alpha, beta, eva_vi_rgb, ir_warp_fake_y, ir_warp_fake_rgb, ST)

        # # image_vi_y, image_vi_cb, image_vi_cr = rgb_to_ycbcr(image_vi) 
        # # image_vi = torch.cat((image_vi_y, image_vi_y, image_vi_y), dim=1)
        # eva_vi = image_vi_warp
        # alpha0 = (image_ir < eva_vi).float()
        # # print(alpha0.size())
        # beta0 = 1 - alpha0
        # eva_F = alpha0 * image_vi_warp + beta0 * image_ir
        # P = eva_F
        # Q = eva_vi
        n = image_vi.size(0)
        # for i in range(stage):
        #     if i == 0:
        #         eva_defor = 0
        #     alpha, beta = alpha_beta(eva_vi, image_ir)
        #     eva_vi, eva_F, P, Q, alpha, beta, eva_defor = dunnet(image_ir, image_vi_warp, eva_vi, eva_F, P, Q, alpha, beta, eva_defor)
        for i in range(n):
            avg_psnr1 = calculate_psnr(image_vi_y[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0, eva_F_y[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0, test_y_channel=True)
            avg_psnr2 = calculate_psnr(image_ir_y[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0, eva_F_y[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0, test_y_channel=True)
            avg_psnr = (avg_psnr1 + avg_psnr2) / 2
            avg_ssim1 = calculate_ssim(image_vi[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0, eva_F_rgb[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0)
            avg_ssim2 = calculate_ssim(image_ir[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0, eva_F_rgb[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0)
            avg_ssim = (avg_ssim1 + avg_ssim2) / 2
            #avg_scd = SCD(rgb2gray(image_vi[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0), rgb2gray(image_ir[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0), rgb2gray(eva_F_rgb[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0))            # avg_ssim = calculate_ssim(eva_vi_y[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0, image_vi_y[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0)
             
            #avg_qabf = get_Qabf(rgb2gray(image_vi[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0), rgb2gray(image_ir[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0), rgb2gray(eva_F_rgb[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0))            # avg_ssim = calculate_ssim(eva_vi_y[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0, image_vi_y[i, ::].permute(1, 2, 0).cpu().numpy() * 255.0)
        avg_psnr_val += avg_psnr
        avg_ssim_val += avg_ssim

            
        if epoch % save_epoch == 0 or epoch == 1500:
            # R_low_img = tensor2numpy_R(vi_r)
            # R_high_img = tensor2numpy_R(vi_l)
            # L_low_img = tensor2numpy_L(ir_l)
            # L_high_img = tensor2numpy_L(ir_l)
            # save_pic(R_low_img, evalfold_path, str(step) + "vi_r")
            # save_pic(R_high_img, evalfold_path, str(step) + "vi_l")
            # save_pic(L_low_img, evalfold_path, str(step) + "ir_l")
            # save_pic(L_high_img, evalfold_path, str(step) + "ir_l1")
            # R_high_img = torch.cat((vi_l, vi_l, vi_l),dim=1)
            # image_vi_cb = tensor2numpy_R(image_vi_cb)
            # image_vi_cr = tensor2numpy_R(image_vi_cr)
            eva_vi = tensor2numpy_R(eva_vi_rgb)
            # print(eva_ir.shape)
            eva_F = tensor2numpy_R(eva_F_rgb)
            ir_warp1 = tensor2numpy_R(ir_warp_fake_rgb)
            # L_low_img = torch.cat((ir_l, ir_l, ir_l),dim=1)
            # L_low_img = ir_l
            # P = tensor2numpy_R(P)
            # Q = tensor2numpy_R(Q)
            # P_merge = cv2.merge([P[:,:,:1], image_vi_cr, image_vi_cb])
            # P_rgb = cv2.cvtColor(P_merge, cv2.COLOR_YCrCb2RGB)
            # eva_F_merge = cv2.merge([eva_F[:, :,:1], image_vi_cr, image_vi_cb])
            # eva_F_rgb = cv2.cvtColor(eva_F_merge, cv2.COLOR_YCrCb2RGB)

            # L_high_img= torch.cat((ir_l, ir_l, ir_l),dim=1)
            # L_high_img = ir_l
            # L_high_img = tensor2numpy_L(L_high_img)
            save_pic(eva_vi, evalfold_path, name[0] + "_eva_vi")
            save_pic(eva_F, evalfold_path, name[0] + "_eva_F")
            save_pic(ir_warp1, evalfold_path, name[0] + "_ir_warp")
            # save_pic(P, evalfold_path, name[0] + "_P")
            # save_pic(Q, evalfold_path, name[0] + "_Q")
            # avg_ssim = 0
    print(f'Epoch: {epoch}: PSNR: {avg_psnr_val / (step+1)}, SSIM: {avg_ssim_val / (step+1)}')
    return avg_psnr_val / (step+1), avg_ssim_val / (step+1)
def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index + ".png")
    cv2.imwrite(save_path, outputpic)

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
def rgb2gray(img):
    # 输入 (H, W, 3)，输出 (H, W)
    if img.ndim == 3 and img.shape[2] == 3:
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    else:
        return img  # 如果本身是灰度图则直接返回
def get_Qabf(pA, pB, pF):
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5;
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # Sobel Operator Sobel算子
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    # if y is the response to h1 and x is the response to h3;then the intensity is sqrt(x^2+y^2) and  is arctan(y/x);
    # 如果y对应h1，x对应h2，则强度为sqrt(x^2+y^2)，方向为arctan(y/x)

    strA = pA
    strB = pB
    strF = pF

    # 数组旋转180度
    def flip180(arr):
        return np.flip(arr)

    # 相当于matlab的Conv2
    def convolution(k, data):
        k = flip180(k)
        data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        img_new = convolve2d(data, k, mode='valid')
        return img_new

    def getArray(img):
        SAx = convolution(h3, img)
        SAy = convolution(h1, img)
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        n, m = img.shape
        aA = np.zeros((n, m))
        zero_mask = SAx == 0
        aA[~zero_mask] = np.arctan(SAy[~zero_mask] / SAx[~zero_mask])
        aA[zero_mask] = np.pi / 2
        # for i in range(n):
        #     for j in range(m):
        #         if (SAx[i, j] == 0):
        #             aA[i, j] = math.pi / 2
        #         else:
        #             aA[i, j] = math.atan(SAy[i, j] / SAx[i, j])
        return gA, aA

    # 对strB和strF进行相同的操作
    gA, aA = getArray(strA)
    gB, aB = getArray(strB)
    gF, aF = getArray(strF)

    # the relative strength and orientation value of GAF,GBF and AAF,ABF;
    def getQabf(aA, gA, aF, gF):
        mask = (gA > gF)
        GAF = np.where(mask, gF / gA, np.where(gA == gF, gF, gA / gF))

        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)

        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))

        QAF = QgAF * QaAF
        return QAF

    QAF = getQabf(aA, gA, aF, gF)
    QBF = getQabf(aB, gB, aF, gF)

    # 计算QABF
    deno = np.sum(gA + gB)
    nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
    output = nume / deno
    return output
def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r

def SCD(A, B, F):
    r = corr2(F - B, A) + corr2(F - A, B)
    return r