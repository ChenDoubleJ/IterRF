import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

import torch
import torch.nn as nn

class alpha_beta_net(nn.Module):
    def __init__(self, 
        inp_channels=2,   
        out_channels=1, 
        channel=32,
    ):
        super(alpha_beta_net, self).__init__()

        self.conv0 = nn.Conv2d(inp_channels, channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channel, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, vis, ir):
        x = torch.cat((vis, ir), dim=1)  
        out1 = self.conv0(x)
        out2 = self.conv1(self.relu(out1))
        alpha = self.sigmoid(out2)
        beta = 1 - alpha
        return alpha, beta
