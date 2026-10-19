import torch
import numpy as np
import torch.nn as nn
# from network.architecture import get_conv2d_layer
import torch.nn.functional as F
from utils import *


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        #self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        #self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        # x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
        #                   .contiguous().reshape(b, c, h, w))
        # x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
        #                  .contiguous().reshape(b, c, h, w))
        x = x + self.attn(x.reshape(b, c, -1).transpose(-2, -1).contiguous().transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(x.reshape(b, c, -1).transpose(-2, -1).contiguous().transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)




class RegistrationNet(nn.Module):
    def __init__(self, num_blocks=[2, 3, 4], num_heads=[1, 2, 4], channels=[18, 36, 72, 144], num_refinement=2,
                 expansion_factor=2.66):
        super(RegistrationNet, self).__init__()

        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

# ---------------------------------------------------------------------------------------------------------------
        self.encoders_vis = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])

        self.encoders_ir = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
        
        self.encoders_Q = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
# ---------------------------------------------------------------------------------------------------------------
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])

        # the number of reduce block == the number of decoder - 1
        # self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
        #                               for i in reversed(range(2, len(channels)))])
# ---------------------------------------------------------------------------------------------------------------
        self.reduces1 = nn.ModuleList([nn.Conv2d(channels[2] * 2, channels[2], kernel_size=1, bias=False)])
        self.reduces1.append(nn.Conv2d(channels[1] * 3, channels[1], kernel_size=1, bias=False))
        # self.reduces.append(nn.Conv2d(channels[0] * 3, channels[0], kernel_size=1, bias=False))

        # the number of decoder == the number of encoder - 1
        self.decoders1 = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])

        self.decoders1.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))

        self.decoders1.append(nn.Sequential(*[TransformerBlock(channels[0] * 3, num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))
# ---------------------------------------------------------------------------------------------------------------
        self.reduces2 = nn.ModuleList([nn.Conv2d(channels[2] * 2, channels[2], kernel_size=1, bias=False)])
        self.reduces2.append(nn.Conv2d(channels[1] * 3, channels[1], kernel_size=1, bias=False))
        # self.reduces.append(nn.Conv2d(channels[0] * 3, channels[0], kernel_size=1, bias=False))

        # the number of decoder == the number of encoder - 1
        self.decoders2 = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])

        self.decoders2.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))

        self.decoders2.append(nn.Sequential(*[TransformerBlock(channels[0] * 3, num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))
# ---------------------------------------------------------------------------------------------------------------
        self.refinement = nn.Sequential(*[TransformerBlock(channels[0] * 3, num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[0] * 3, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, ir, Q):
        # fo_vis = self.embed_conv(vis)
        fo_ir = self.embed_conv(ir)
        fo_Q = self.embed_conv(Q)
# --------------------------------------------------------------------------------------
        # vis_enc1 = self.encoders_vis[0](fo_vis)
        # vis_enc2 = self.encoders_vis[1](self.downs[0](vis_enc1))
        # vis_enc3 = self.encoders_vis[2](self.downs[1](vis_enc2))

        ir_enc1 = self.encoders_ir[0](fo_ir)
        ir_enc2 = self.encoders_ir[1](self.downs[0](ir_enc1))
        ir_enc3 = self.encoders_ir[2](self.downs[1](ir_enc2))

        Q_enc1 = self.encoders_Q[0](fo_Q)
        Q_enc2 = self.encoders_Q[1](self.downs[0](Q_enc1))
        Q_enc3 = self.encoders_Q[2](self.downs[1](Q_enc2))
# ----------------------------------------------------------------------------------------
        # reduce31 = self.reduces1[0](torch.cat([vis_enc3, ir_enc3], dim=1))
        # out_dec3 = self.decoders1[0](reduce31)

        # out_dec31_up = self.ups[1](out_dec3)
        # reduce21 = self.reduces1[1](torch.cat([out_dec31_up, vis_enc2, ir_enc2], dim=1))
        # out_dec21 = self.decoders1[1](reduce21)

        # out_dec21_up = self.ups[2](out_dec21)
        # out_dec11 = self.decoders1[2](torch.cat([out_dec21_up, vis_enc1, ir_enc1], dim=1))
# ------------------------------------------------------------------------------------------
        reduce32 = self.reduces1[0](torch.cat([Q_enc3, ir_enc3], dim=1))
        out_dec32 = self.decoders1[0](reduce32)

        out_dec32_up = self.ups[1](out_dec32)
        reduce22 = self.reduces1[1](torch.cat([out_dec32_up, Q_enc2, ir_enc2], dim=1))
        out_dec22 = self.decoders1[1](reduce22)

        out_dec22_up = self.ups[2](out_dec22)
        out_dec12 = self.decoders1[2](torch.cat([out_dec22_up, Q_enc1, ir_enc1], dim=1))
# ------------------------------------------------------------------------------------------
        out = self.output(self.refinement(out_dec12))



        return out





