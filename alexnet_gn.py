from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace

__all__ = ['alexnet_gn']

class AlexnetGN(nn.Module):
    def __init__(self, in_channel=3, out_dim=128, l2norm=True):
        super(AlexnetGN, self).__init__()
        self._l2norm = l2norm
        conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 96, 11, 4, 2, bias=False),
            nn.GroupNorm(32, 96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        conv_block_2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 384),
            nn.ReLU(inplace=True),
        )
        conv_block_4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 384),
            nn.ReLU(inplace=True),
        )
        conv_block_5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        ave_pool = nn.AdaptiveAvgPool2d((6,6))
                
        fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        fc8 = nn.Sequential(
            nn.Linear(4096, out_dim)
        )
        head = [fc6, fc7, fc8]
        if self._l2norm: 
            head.append(Normalize(2))
        
        self.backbone = nn.Sequential(
            conv_block_1,
            conv_block_2,
            conv_block_3,
            conv_block_4,
            conv_block_5,
            ave_pool,
        )
        
        self.head = nn.Sequential(*head)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.head(x)
        return x

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def alexnet_gn(**kwargs):
    model = AlexnetGN(**kwargs)
    
    return model
