'''
    modified from original to:
        - return val transform
        - enable loading the full checkpoints (with mlp head)
        - enable loading custom models
        - check hashid
'''

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision.models.resnet import resnet50 as _resnet50
from alexnet_gn import alexnet_gn as _alexnet_gn

dependencies = ['torch', 'torchvision']

def _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)    
    ])
    
    return transform

def resnet50_barlowtwins_backbone(pretrained=True, **kwargs):
    """Official resnet50 barlowtwins model (resnet50 backbone only) from `Barlow Twins: Self-Supervised Learning 
    via Redundancy Reduction <https://arxiv.org/abs/2103.03230>`__."""
    
    model = _resnet50(pretrained=False, **kwargs)
    if pretrained:
        cache_file_name="resnet50_barlowtwins-cf015f40cc.pth"
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/resnet50.pth', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        model.fc = torch.nn.Identity() # backbone excludes .fc classifier, set to Identity
        model.load_state_dict(state_dict, strict=True)
        model.hashid = 'cf015f40cc'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform()
    
    return model, transform
