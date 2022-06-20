"""
  custom wrappers used when loading full models as feature extractors (including mlp head)
"""
import torch
import torch.nn as nn
import torchvision

from alexnet_gn import alexnet_gn

class BarlowTwins(nn.Module):
    def __init__(self, projector_sizes=[8192,8192,8192]):
        super().__init__()
        
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        
        # projector
        sizes = [2048] + projector_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x):
        z = self.projector(self.backbone(x))
        z = self.bn(z)
        return z
    
class BarlowTwinsAlexnetGN(nn.Module):
    def __init__(self, projector_sizes=[4096,4096,4096]):
        super().__init__()
        self.backbone = alexnet_gn().backbone
        self.flatten = nn.Flatten()
        
        # projector
        sizes = [256*6*6] + projector_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x):
        z = self.projector(self.flatten(self.backbone(x)))
        z = self.bn(z)
        return z
