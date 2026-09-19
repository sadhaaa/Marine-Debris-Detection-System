"""
MixStructureBlock and EMA (Efficient Multi-scale Attention)
From: An improved YOLOv11 network for marine debris detection in underwater environment
      (Jing Yuanwei et al., Scientific Reports 2026)

Components:
1. MixStructureBlock: Replaces standard C3k2/bottleneck blocks.
   - Multi-scale dilated convolutions (parallel 3x3, 5x5, 7x7 paths with dilations).
   - Compound attention: Channel Attention (CA) + Pixel Attention (PA) + Simple Pixel Attention (SPA).
2. EMA (Efficient Multi-scale Attention):
   - Replaces or augments PSA / C2PSA blocks in the head / backbone.
   - 1x1 branch + 3x3 branch with grouped spatial-channel feature interaction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention (EMA).
    Groups channels, cross-interacts across 1x1 and 3x3 spatial branches to retain multi-scale cues.
    """
    def __init__(self, channels: int, factor: int = 8):
        super().__init__()
        self.groups = factor
        assert channels % factor == 0, f"channels ({channels}) must be divisible by factor ({factor})"
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        group_c = c // self.groups
        x_group = x.view(b * self.groups, group_c, h, w)

        # Coordinate attention-style pooling
        x_h = self.pool_h(x_group)
        x_w = self.pool_w(x_group).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 1x1 path + 3x3 path
        out1 = self.gn(x_group * x_h.sigmoid() * x_w.sigmoid())
        out2 = self.conv3x3(x_group)

        # Cross spatial attention
        w1 = self.softmax(self.agp(out1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        w2 = out2.reshape(b * self.groups, group_c, -1)
        weights = (torch.matmul(w1, w2)).reshape(b * self.groups, 1, h, w).sigmoid()

        out = (out1 + out2) * weights
        return out.view(b, c, h, w)


class MixStructureBlock(nn.Module):
    """
    MixStructureBlock from Improved YOLOv11 for marine debris detection.
    Combines multi-branch multi-scale dilated convolutions and hybrid channel-pixel attention.
    """
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        # Parallel multi-scale / dilated branches: 3x3, 5x5 (effective via 3x3 rate=2), 7x7 (effective via 3x3 rate=3)
        self.branch1 = nn.Sequential(
            nn.Conv2d(c_, c_ // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_ // 2),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(c_, c_ // 2, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c_ // 2),
            nn.SiLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(c_, c_ // 2, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(c_ // 2),
            nn.SiLU()
        )

        fused_c = (c_ // 2) * 3
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_c, fused_c // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_c // 4, fused_c, 1, bias=False),
            nn.Sigmoid()
        )
        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(fused_c, 1, 1, bias=False),
            nn.Sigmoid()
        )

        self.cv2 = nn.Sequential(
            nn.Conv2d(fused_c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        self.shortcut = (c1 == c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.cv1(x)
        b1 = self.branch1(res)
        b2 = self.branch2(res)
        b3 = self.branch3(res)
        cat = torch.cat([b1, b2, b3], dim=1)

        # Hybrid Channel & Pixel Attention
        att = cat * self.ca(cat) * self.pa(cat)
        out = self.cv2(att)
        return x + out if self.shortcut else out
