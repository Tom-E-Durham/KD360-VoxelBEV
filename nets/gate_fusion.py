import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, channels):
        super(GatedFusion, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, lidar_feat, img_feat):
        cat_feat = torch.cat([img_feat, lidar_feat], dim=1)
        gate = self.gate_conv(cat_feat)
        fused = gate * img_feat + (1 - gate) * lidar_feat
        out = self.fuse_conv(torch.cat([fused, fused], dim=1))
        return out
