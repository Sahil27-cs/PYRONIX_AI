"""
PyTorch U-Net Architectures for Multi-Spectral Wildfire Segmentation
Supports:
1. ResNet-34 U-Net (with pretrained encoder adapted to 6 input channels)
2. Standard U-Net (from scratch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvBlock(nn.Module):
    """Double Conv + BatchNorm + ReLU block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    """Decoder block: Bilinear Upsample + Concat Skip + Double Conv"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle small spatial dimension mismatches if any
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet34UNet(nn.Module):
    """
    ResNet-34 Encoder U-Net for 6-channel Sentinel-2 optical imagery:
    Channels: B2, B3, B4, B8, B11, B12
    Output: 1-channel binary segmentation logits (without sigmoid)
    """
    def __init__(self, in_channels=6, num_classes=1, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet34(weights=weights)

        # Modify first conv layer to accept in_channels (e.g. 6 instead of 3)
        orig_conv1 = base_resnet.conv1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            if pretrained:
                if in_channels >= 3:
                    # Copy RGB weights to first 3 channels
                    self.conv1.weight[:, :3, :, :] = orig_conv1.weight[:, :3, :, :]
                    # For remaining channels (NIR, SWIR-1, SWIR-2, SAR, etc.), initialize with mean of RGB weights
                    rgb_mean = orig_conv1.weight[:, :3, :, :].mean(dim=1, keepdim=True)
                    for c in range(3, in_channels):
                        self.conv1.weight[:, c:c+1, :, :] = rgb_mean
                else:
                    # For 1 or 2 channels (e.g. single-pol or dual-pol SAR)
                    self.conv1.weight[:, :in_channels, :, :] = orig_conv1.weight[:, :in_channels, :, :]
            else:
                nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        # Encoder stages
        self.layer1 = base_resnet.layer1  # 64 channels,  stride 4
        self.layer2 = base_resnet.layer2  # 128 channels, stride 8
        self.layer3 = base_resnet.layer3  # 256 channels, stride 16
        self.layer4 = base_resnet.layer4  # 512 channels, stride 32

        # Decoder stages
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64,  out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64,  skip_channels=64,  out_channels=32)

        # Final upsample from stride 2 to stride 1
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(32, 16),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Stem
        x0 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, H/2, W/2)
        x_pool = self.maxpool(x0)                # (B, 64, H/4, W/4)

        # Encoder forward
        x1 = self.layer1(x_pool)                 # (B, 64, H/4, W/4)
        x2 = self.layer2(x1)                     # (B, 128, H/8, W/8)
        x3 = self.layer3(x2)                     # (B, 256, H/16, W/16)
        x4 = self.layer4(x3)                     # (B, 512, H/32, W/32)

        # Decoder forward with skip connections
        d4 = self.dec4(x4, x3)                   # (B, 256, H/16, W/16)
        d3 = self.dec3(d4, x2)                   # (B, 128, H/8, W/8)
        d2 = self.dec2(d3, x1)                   # (B, 64, H/4, W/4)
        d1 = self.dec1(d2, x0)                   # (B, 32, H/2, W/2)

        # Final projection to full resolution (B, num_classes, H, W)
        out = self.final_up(d1)
        return out

class StandardUNet(nn.Module):
    """
    Standard U-Net from scratch (Lightweight baseline without pretrained backbone)
    """
    def __init__(self, in_channels=6, num_classes=1, base_filters=32):
        super().__init__()
        bf = base_filters
        self.inc = ConvBlock(in_channels, bf)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(bf, bf * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(bf * 2, bf * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(bf * 4, bf * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(bf * 8, bf * 16))

        self.up1 = DecoderBlock(bf * 16, bf * 8, bf * 8)
        self.up2 = DecoderBlock(bf * 8, bf * 4, bf * 4)
        self.up3 = DecoderBlock(bf * 4, bf * 2, bf * 2)
        self.up4 = DecoderBlock(bf * 2, bf, bf)
        self.outc = nn.Conv2d(bf, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)
        return self.outc(d4)
