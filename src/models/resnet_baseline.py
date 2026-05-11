"""
ResNet-50 baseline feature extractor adapted for 10-channel input.

AI-assist prompt:
"Write a ResNet-50 feature extractor in PyTorch that accepts 10-channel input
by replacing the first convolutional layer, keeps the rest pretrained from
ImageNet, freezes the backbone, and outputs 2048-d embeddings."
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, in_chans=10, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Replace first conv to accept in_chans
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_chans, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        # Initialize new conv weights by averaging ImageNet RGB weights across channels
        if pretrained and in_chans != 3:
            with torch.no_grad():
                # Average RGB weights across the 3 input channels
                avg_weight = old_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                # Replicate the averaged weight to all input channels
                for i in range(in_chans):
                    backbone.conv1.weight[:, i, :, :] = avg_weight.squeeze(1) * (3.0 / in_chans)

        # Remove classifier head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = self.backbone(x)
        return x.squeeze(-1).squeeze(-1)
