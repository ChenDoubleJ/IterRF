import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import kornia.utils as KU
import kornia.filters as KF


class losses(nn.Module):
    def __init__(self):
        super(losses, self).__init__()
        self.ssim = L_SSIM(window_size=11)
    def rgb_to_ycbcr(self, image):
        r = image[:, 0:1, :, :]
        g = image[:, 1:2, :, :]
        b = image[:, 2:3, :, :]

        y  =  0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128/255.0
        cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 128/255.0

        return y, cb, cr   # 注意顺序与OpenCV对齐
    def RGB2YCrCb(self, rgb_image):
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
    def YCbCr2RGB(self, Y, Cb, Cr):
        ycrcb = torch.cat([Y, Cr, Cb], dim=1)
        B, C, W, H = ycrcb.shape
        im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
        mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
        ).to(Y.device)
        bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
        temp = (im_flat + bias).mm(mat)
        out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
        out = out.clamp(0,1.0)
        return out

    def l1loss(self, img1, img2, mask=1, eps=1e-2):
        mask_ = torch.logical_and(img1 > 1e-2, img2 > 1e-2)
        mean_ = img1.mean(dim=[-1, -2], keepdim=True) + img2.mean(dim=[-1, -2], keepdim=True)
        mean_ = mean_.detach() / 2
        std_ = img1.std(dim=[-1, -2], keepdim=True) + img2.std(dim=[-1, -2], keepdim=True)
        std_ = std_.detach() / 2
        img1 = (img1 - mean_) / std_
        img2 = (img2 - mean_) / std_
        img1 = KF.gaussian_blur2d(img1, (3, 3), (1, 1)) * mask_
        img2 = KF.gaussian_blur2d(img2, (3, 3), (1, 1)) * mask_
        return ((img1 - img2) * mask).abs().clamp(min=eps).mean()


    def l2loss(self, img1, img2, mask=1, eps=1e-2):
        mask_ = torch.logical_and(img1 > 1e-2, img2 > 1e-2)
        mean_ = img1.mean(dim=[-1, -2], keepdim=True) + img2.mean(dim=[-1, -2], keepdim=True)
        mean_ = mean_.detach() / 2
        std_ = img1.std(dim=[-1, -2], keepdim=True) + img2.std(dim=[-1, -2], keepdim=True)
        std_ = std_.detach() / 2
        img1 = (img1 - mean_) / std_
        img2 = (img2 - mean_) / std_
        img1 = KF.gaussian_blur2d(img1, (3, 3), (1, 1)) * mask_
        img2 = KF.gaussian_blur2d(img2, (3, 3), (1, 1)) * mask_
        return ((img1 - img2) * mask).abs().clamp(min=eps).pow(2).mean()
        
    def gradientloss(self, img1, img2, mask=1, eps=1e-2):
        # img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
        # mask_ = torch.logical_and(img1 > 1e-2, img2 > 1e-2)
        mean_ = img1.mean(dim=[-1, -2], keepdim=True) + img2.mean(dim=[-1, -2], keepdim=True)
        mean_ = mean_.detach() / 2
        std_ = img1.std(dim=[-1, -2], keepdim=True) + img2.std(dim=[-1, -2], keepdim=True)
        std_ = std_.detach() / 2
        img1 = (img1 - mean_) / std_
        img2 = (img2 - mean_) / std_
        grad1 = KF.spatial_gradient(img1, order=2)
        grad2 = KF.spatial_gradient(img2, order=2)
        mask = mask.unsqueeze(1)
        # grad1 = self.AP5(self.MP5(grad1))
        # grad2 = self.AP5(self.MP5(grad2))
        # print((grad1-grad2).abs().mean())
        l = (((grad1 - grad2) + (grad1 - grad2).pow(2) * 10) * mask).abs().clamp(min=eps).mean()
        # l = l[...,5:-5,10:-10].mean()
        return l
 

    def imgloss(self, src, tgt, mask=1, weights=[0.3, 0.7]):
        return weights[0] * (self.l1loss(src, tgt, mask) + self.l2loss(src, tgt, mask)) + \
               weights[1] * self.gradientloss(src, tgt, mask)

    def weightfiledloss(self, ref, tgt, disp, disp_gt, border_mask):
        ref = (ref - ref.mean(dim=[-1, -2], keepdim=True)) / (ref.std(dim=[-1, -2], keepdim=True) + 1e-5)
        tgt = (tgt - tgt.mean(dim=[-1, -2], keepdim=True)) / (tgt.std(dim=[-1, -2], keepdim=True) + 1e-5)
        g_ref = KF.spatial_gradient(ref, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        g_tgt = KF.spatial_gradient(tgt, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        w = (((g_ref + g_tgt)) * 2 + 1) * border_mask
        return (w * (1000 * (disp - disp_gt).abs().clamp(min=1e-2).pow(2))).mean()

    def border_suppression(self, img, mask):
        return (img * (1 - mask)).mean() # [0.6, 0.00, 0.3, 0.0]
    def fusloss(self, ir, vi, fu, fu_rgb, vi_rgb, mask, weights=[0.6, 0.00, 0.3, 0.0]):
        grad_ir = KF.spatial_gradient(ir, order=2).abs().sum(dim=[1, 2])
        grad_vi = KF.spatial_gradient(vi, order=2).abs().sum(dim=[1, 2])
        grad_fus = KF.spatial_gradient(fu, order=2).abs().sum(dim=[1, 2])
        grad_joint = torch.max(grad_ir, grad_vi)
        loss_grad = 0.5 * F.l1_loss(mask * grad_fus, mask * grad_joint) + 0.5 * F.l1_loss((1 - mask)*grad_fus, (1 - mask)*grad_ir)
        #loss_ssim = 0.5 * self.ssim(ir, fu) + 0.5 * self.ssim(mask*vi, mask*fu)
        int_joint = torch.max(vi, ir)
        loss_intensity = 0.5 * F.l1_loss(mask * fu, mask * int_joint) + 0.5 * F.l1_loss((1 - mask)*fu, (1 -mask) * ir)
        # loss_intensity = torch.zeros(1).cuda()
        _, fu_cb, fu_cr = self.RGB2YCrCb(fu_rgb)
        _, vi_cb, vi_cr = self.RGB2YCrCb(vi_rgb)
        #loss_color = 0.5 * (F.l1_loss(mask * fu_cb, mask * vi_cb)) + 0.5 * (F.l1_loss(mask * fu_cr, mask * vi_cr))
        loss_color = torch.zeros(1).cuda()
        loss_ssim = torch.zeros(1).cuda()
        loss_total = weights[0] * loss_grad + weights[1] * loss_ssim + weights[2] * loss_intensity + weights[3] * loss_color
        return loss_intensity, loss_ssim, loss_grad, loss_color, loss_total
    # def fusloss(self, ir, vi, fu, mask, weights=[0.25, 0.1, 0.3, 0]):
    #     grad_ir = KF.spatial_gradient(ir, order=2).abs().sum(dim=[1, 2])
    #     grad_vi = KF.spatial_gradient(vi, order=2).abs().sum(dim=[1, 2])
    #     grad_fus = KF.spatial_gradient(fu, order=2).abs().sum(dim=[1, 2])
    #     mask = (torch.logical_and(ir > 0, vi > 0)*mask)
    #     # mm = mask
    #     # if (fu > 2.0 / 255).sum() < 100:
    #     #     mask = 1
    #     # grad_joint = torch.max(grad_ir, grad_vi)
    #     # loss_grad = (((grad_joint - grad_fus).abs().clamp(min=1e-9)) * mask).mean() + (((grad_ir - grad_fus).abs().clamp(min=1e-9)) * (1 - mask)).mean()
    #     # loss_ssim = (self.ssim(ir, fu) + self.ssim(vi, fu))
    #     # # print(loss_ssim)
    #     # intensity_joint = torch.max(vi, ir) * mask
    #     # loss_intensity = F.l1_loss(fu * mask, intensity_joi_border'_re:605
    def border_suppression(self, img, mask):
        return (img * (1 - mask)).mean()
    # self.border_suppression(ir_warp_fake_rgb, goo
    # loss_intensity, loss_ssim, loss_grad, loss_fusion = selfnt) + F.l1_loss(fu * (1- mask), ir)
    #     # mm = (mask * grad_vi > mask * grad_ir).float()
    #     # loss_grad = 0.5 * F.l1_loss(mm * grad_fus, mm * grad_vi) + 0.5 * F.l1_loss((1 - mm) * grad_fus, (1 - mm) * grad_ir)
    #     loss_grad = 0.5 * F.l1_loss(grad_fus, grad_ir) + 0.5 * F.l1_loss(mask*grad_fus, mask*grad_vi)
        
    #     loss_ssim = 0.5 * self.ssim(ir, fu) + 0.5 * self.ssim(mask*vi, mask*fu)
    #     # loss_intensity = 0.5 * F.l1_loss(fu, ir) + 0.5 * F.l1_loss(mask*fu, mask*vi)
    #     # m = (ir > vi).float()
    #     # loss_intensity = 0.5 *(F.l1_loss((1 - mask) * fu, (1 - mask) * ir))
    #     m = (mask * vi > mask * ir).float()
    #     # f = m * vi + (1 - m) * ir
    #     loss_intensity = 0.5 * F.l1_loss(m * fu, m * vi) + 0.5 * F.l1_loss((1 - m) * fu, (1 - m) * ir)

    #     loss_total = weights[0] * loss_grad + weights[1] * loss_ssim + weights[2] * loss_intensity
    #     return loss_intensity, loss_ssim, loss_grad, loss_total
    def smoothloss(self, disp, img=None):
        smooth_d = [3 * 3, 7 * 3, 15 * 3]
        b, c, h, w = disp.shape
        grad = KF.spatial_gradient(disp, order=2).abs().sum(dim=2)[:, :, 3:-3, 3:-3].clamp(min=1e-9).mean()
        local_smooth_re = 0
        for d in smooth_d:
            local_mean = KF.gaussian_blur2d(disp, (d, d), (d // 6, d // 6), border_type='replicate')
            # local_mean_pow2 = F.avg_pool2d(disp.pow(2),kernel_size=d,stride=1,padding=d//2)
            local_smooth_re += 1 / (d * 1.0 + 1) * (disp - local_mean)[:, :, d // 2:-d // 2, d // 2:-d // 2].pow(2).mean()
            # local_smooth_re += 1/(d*1.0+1)*(disp.pow(2)-local_mean_pow2)[:,:,5:-5,5:-5].pow(2).mean()
        # global_var = disp[...,2:-2,2:-2].var(dim=[-1,-2]).clamp(1e-5).mean()
        # std = img.std(dim=[-1,-2]).mean().clamp(min=0.003)
        # grad = grad[...,10:-10,10:-10]
        return 5000 * local_smooth_re + 500 * grad
    def forward(self, image_vi_warp, image_vi, image_ir, image_ir_warp, ir_warp_fake_rgb, eva_vi_y, eva_vi_rgb, 
                eva_F_y, eva_F_rgb, eva_defor1, eva_defor2, goodmask_inverse, goodmask, disp, border_mask):
        image_ir_y, _, _ = self.RGB2YCrCb(image_ir)
        image_vi_y, _, _ = self.RGB2YCrCb(image_vi)
        mask_ = torch.logical_and(image_ir_y > 1e-5, image_vi_y > 1e-5)
        mm = (eva_vi_rgb > 1e-5).float()
        mm = mm[:, :1, ::]
        mask_ = torch.logical_and(mm, mask_)
        mask_ = goodmask * goodmask_inverse
        # yy, _, _ = self.RGB2YCrCb(image_vi_warp)
        # photometric loss
        loss_reg_img = self.imgloss(eva_vi_rgb, image_vi, goodmask * goodmask_inverse) #  + \
                    #    self.imgloss(self.image_fusion_1, self.image_fusion_gt, mask_)
        loss_reg_field = torch.zeros(1).cuda()
        # loss_reg_field = self.weightfiledloss(image_vi_warp, ir_warp_fake_rgb,
        #                                       eva_defor2['ir2vis'], disp.permute(0, 3, 1, 2), border_mask)
        loss_smooth = self.smoothloss(eva_defor1['down2']) + self.smoothloss(eva_defor1['down4']) + self.smoothloss(eva_defor1['down8'])
        loss_border_re = self.border_suppression(eva_vi_rgb, goodmask_inverse) 
        loss_intensity, loss_ssim, loss_grad, loss_color, loss_fusion = self.fusloss(image_ir_y, image_vi_y, eva_F_y,eva_F_rgb, image_vi, mask_) # mask_
        loss_total = loss_reg_img * 20.0 + loss_reg_field * 16 + loss_smooth * 25 + loss_border_re * 40 + 5 * loss_fusion

        return loss_total, loss_reg_img, loss_reg_field, loss_smooth, loss_border_re, loss_intensity, loss_ssim, loss_grad, loss_color

class registration_and_fusion_loss(nn.Module):
    def __init__(self):
        super(registration_and_fusion_loss, self).__init__()
        self.loss_func_ssim = L_SSIM(window_size=48)  # 48
        self.loss_func_Grad = L_Grad_position()
        self.loss_func_Max = L_Intensity()
        self.loss_func_color = L_color()
    def forward(self, image_infrared, eva_image_visible, image_fused, warp_vi, eva_warp_vi, img_vi,eva_defor, max_ratio=0.6, ssim_vis_ratio=1, ssim_ir_ratio=1, ssim_ratio=0.2, color_ratio=4.0, grad_ratio=1.0, l1_ratio = 1.4):
        image_visible_gray = self.rgb2gray(eva_image_visible)
        image_infrared_gray = self.rgb2gray(image_infrared)
        image_fused_gray = self.rgb2gray(image_fused)
        loss_max = max_ratio * self.loss_func_Max(eva_image_visible, image_infrared, image_fused)
        loss_ssim = ssim_ratio * (ssim_vis_ratio * self.loss_func_ssim(eva_image_visible, image_fused) + ssim_ir_ratio * self.loss_func_ssim(image_infrared_gray, image_fused_gray))
        loss_color = color_ratio * self.loss_func_color(eva_image_visible, image_fused)
        loss_grad = grad_ratio * self.loss_func_Grad(image_visible_gray, image_infrared_gray, image_fused_gray)
        # loss_max  = torch.zeros(1).cuda()
        # 上一个是ssim1 l1 4
        loss_l1 = l1_ratio * (F.l1_loss(eva_warp_vi, warp_vi) + 1.0 * F.l1_loss(eva_image_visible, img_vi))
        loss_smooth = 0.25 * self.smoothloss(eva_defor)
        # loss_color  = torch.zeros(1).cuda()
        total_loss = loss_max + loss_ssim + loss_color + loss_grad + loss_l1 + loss_smooth
        return total_loss, loss_ssim, loss_max, loss_color, loss_grad, loss_l1, loss_smooth

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray
    def smoothloss(self, disp, img=None):
        smooth_d = [3 * 3, 7 * 3, 15 * 3]
        b, c, h, w = disp.shape
        grad = KF.spatial_gradient(disp, order=2).abs().sum(dim=2)[:, :, 5:-5, 5:-5].clamp(min=1e-9).mean()
        local_smooth_re = 0
        for d in smooth_d:
            local_mean = KF.gaussian_blur2d(disp, (d, d), (d // 6, d // 6), border_type='replicate')
            # local_mean_pow2 = F.avg_pool2d(disp.pow(2),kernel_size=d,stride=1,padding=d//2)
            local_smooth_re += 1 / (d * 1.0 + 1) * (disp - local_mean)[:, :, d // 2:-d // 2, d // 2:-d // 2].pow(2).mean()
            # local_smooth_re += 1/(d*1.0+1)*(disp.pow(2)-local_mean_pow2)[:,:,5:-5,5:-5].pow(2).mean()
        # global_var = disp[...,2:-2,2:-2].var(dim=[-1,-2]).clamp(1e-5).mean()
        # std = img.std(dim=[-1,-2]).mean().clamp(min=0.003)
        # grad = grad[...,10:-10,10:-10]
        return 5000 * local_smooth_re + 500 * grad

class fusion_prompt_loss(nn.Module):
    def __init__(self):
        super(fusion_prompt_loss, self).__init__()
        self.fusion_loss = registration_and_fusion_loss()

    def forward(self, image_A, image_B, image_fused, task):
        total_loss = 0
        total_ssim_loss = 0
        total_max_loss = 0
        total_color_loss = 0
        total_grad_loss = 0

        num_tasks = len(task)

        for idx, task_type in enumerate(task):
            img_A = self._get_image(image_A, idx)
            img_B = self._get_image(image_B, idx)
            img_fused = self._get_image(image_fused, idx)

            if task_type == "low_light":
                loss, ssim_loss, max_loss, color_loss, grad_loss = self.fusion_loss(img_A, img_B, img_fused,
                                                                                    max_ratio=8, ssim_ratio=1, text_ratio=10)
            elif task_type == "over_exposure":
                loss, ssim_loss, max_loss, color_loss, grad_loss = self.fusion_loss(img_A, img_B, img_fused,
                                                                                    max_ratio=4, ssim_ratio=0, text_ratio=2)
            elif task_type == "ir_low_contrast":
                loss, ssim_loss, max_loss, color_loss, grad_loss = self.fusion_loss(img_A, img_B, img_fused,
                                                                                    max_ratio=8, ssim_ratio=1, text_ratio=10)
            elif task_type == "ir_noise":
                loss, ssim_loss, max_loss, color_loss, grad_loss = self.fusion_loss(img_A, img_B, img_fused,
                                                                                    max_ratio=6, ssim_ratio=1, text_ratio=10)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            total_loss += loss
            total_ssim_loss += ssim_loss
            total_max_loss += max_loss
            total_color_loss += color_loss
            total_grad_loss += grad_loss

        # Calculate the average for each loss component
        return total_loss / num_tasks, total_ssim_loss / num_tasks, total_max_loss / num_tasks, total_color_loss / num_tasks, total_grad_loss / num_tasks,

    @staticmethod
    def _get_image(images, index):
        return images[index].unsqueeze(0)

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):

        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        loss_color = loss_cb + loss_cr
        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128/255.0
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128/255.0

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = torch.mean(image_visible, dim=1, keepdim=True)
        gray_infrared = torch.mean(image_infrared, dim=1, keepdim=True)

        mask = (gray_infrared > gray_visible).float()

        fused_image = mask * image_infrared + (1 - mask) * image_visible
        Loss_intensity = F.l1_loss(fused_image, image_fused)
        return Loss_intensity

# Use it only if you have a consistent modal preference
class L_Intensity_Consist(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, vis_compose, ir_compose, consist_mode="l1"):
        if consist_mode == "l2":
            Loss_intensity = (vis_compose * F.mse_loss(image_visible, image_fused) + ir_compose * F.mse_loss(image_infrared, image_fused))/2
        else:
            Loss_intensity = (vis_compose * F.l1_loss(image_visible, image_fused) + ir_compose * F.l1_loss(image_infrared, image_fused))/2
        return Loss_intensity

# use the L_Grad_position or L_Grad
class L_Grad_position(nn.Module):
    def __init__(self):
        super(L_Grad_position, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        gradient_A_x, gradient_A_y = self.gradient(image_A)
        gradient_B_x, gradient_B_y = self.gradient(image_B)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)
        loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(gradient_fuse_y, torch.max(gradient_A_y, gradient_B_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = self.tensor_RGB2GRAY(image_visible)
        gray_infrared = self.tensor_RGB2GRAY(image_infrared)
        gray_fused = self.tensor_RGB2GRAY(image_fused)

        d1 = self.gradient(gray_visible)
        d2 = self.gradient(gray_infrared)
        df = self.gradient(gray_fused)
        edge_loss = F.l1_loss(torch.max(d1, d2), df)
        return edge_loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x) + torch.abs(gradient_y)
    
    def tensor_RGB2GRAY(self, image):
        b,c,h,w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret


class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.cuda()
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window.cuda()
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
