import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import kornia.utils as KU
from einops import rearrange
# from fusion_net import Fusionnet
# from registration_net import Registration

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
class DUNnet(nn.Module):
    def __init__(self, eta1, eta2, fusionnet, registration,encoder, device):

        super(DUNnet, self).__init__()
        # self.ST = SpatialTransformer(256, 256, True)
        self.fusionnet = fusionnet
        self.registration = registration
        self.encoder = encoder
        self.eta1 = eta1
        self.eta2 = eta2
        self.device = device
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

    def forward(self, ir_y, vi_warp_y, eva_vi_y, eva_F_y, P, Q, alpha, beta, eva_vi_rgb, ir_warp_fake_y, ir_warp_fake_rgb, ST):
        Q = self.eta1 * eva_vi_y - alpha * (beta * ir_y - P) 
        Q = Q / (alpha * alpha + self.eta1)
        # _, _, f1, _, _, _, f2, _ = self.encoder(eva_vi_y, ir_y)
        defor1 = self.registration(eva_vi_y, ir_y, Q, type='vis2ir')
        # _, _, f3, _, _, _, f4, _ = self.encoder(vi_warp_y, ir_warp_fake_y)
        # defor2 = self.registration(vi_warp_y, ir_warp_fake_y,  type='ir2vis')
        eva_vi_y, _ = ST(eva_vi_y, defor1['vis2ir'])
        eva_vi_rgb, _ = ST(eva_vi_rgb, defor1['vis2ir'])
        defor2 = defor1
        # ir_warp_fake_y, _ = ST(ir_warp_fake_y, defor2['ir2vis'])
        # ir_warp_fake_rgb, _ = ST(ir_warp_fake_rgb, defor2['ir2vis'])
        P = beta * ir_y + alpha * Q + self.eta2 * eva_F_y
        P = P / (1 + self.eta2)
        eva_F_y = self.fusionnet(eva_vi_y, ir_y, P)
        _, F_b, F_r = self.RGB2YCrCb(eva_vi_rgb)
        eva_F_rgb = self.YCbCr2RGB(eva_F_y, F_b, F_r)
        return eva_vi_y, eva_F_y, P, Q, alpha, beta, defor1, defor2, eva_vi_rgb, ir_warp_fake_y, ir_warp_fake_rgb, eva_F_rgb
    
