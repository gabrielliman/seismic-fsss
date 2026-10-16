import torch
import torch.nn as nn
from torchvision import models

# Import your existing grayscale conversion utility
from model.utils.backbone_patch import convert_resnet_to_grayscale

class SiameseNetwork(nn.Module):
    def __init__(self, args=None, backbone="resnet18"):
        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # 1. Initialize the standard pretrained backbone
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        # 2. Convert to 1-channel (seismic patches are single-channel)
        self.backbone = convert_resnet_to_grayscale(self.backbone)

        out_features = list(self.backbone.modules())[-1].out_features

        # 3. Keep the authors' classification head exactly as designed
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1, img2):
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        # The authors explicitly multiply the features before classifying
        combined_features = feat1 * feat2
        
        output = self.cls_head(combined_features)
       
        return output