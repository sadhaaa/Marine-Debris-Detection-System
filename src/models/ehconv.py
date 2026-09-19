"""
Efficient Hybrid Convolution (EHConv).

This module implements the lightweight convolutional operation used
as the basic spatial-processing component of SvelteNeck.
"""

import torch
import torch.nn as nn


class EHConv(nn.Module):
    """
    Efficient Hybrid Convolution.

    Architecture:
        Input
          ↓
        Depthwise Conv
          ↓
        Pointwise 1×1 Conv
          ↓
        BatchNorm
          ↓
        Mish
          ↓
        Output

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int): Depthwise convolution kernel size.
        s (int): Stride.

    Input:
        Tensor of shape [B, c1, H, W]

    Output:
        Tensor of shape [B, c2, H_out, W_out]
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
    ) -> None:
        super().__init__()

        if k % 2 == 0:
            raise ValueError("Kernel size k must be odd.")

        if c1 <= 0 or c2 <= 0:
            raise ValueError("Channel dimensions must be positive.")

        if s <= 0:
            raise ValueError("Stride must be positive.")

        padding = k // 2

        # Depthwise convolution:
        # Each input channel is spatially filtered independently.
        #
        # Input:
        #   [B, c1, H, W]
        #
        # Output:
        #   [B, c1, H_out, W_out]
        self.depthwise = nn.Conv2d(
            in_channels=c1,
            out_channels=c1,
            kernel_size=k,
            stride=s,
            padding=padding,
            groups=c1,
            bias=False,
        )

        # Pointwise convolution:
        # Mixes information across channels using a 1×1 convolution.
        #
        # [B, c1, H_out, W_out]
        #          ↓
        # [B, c2, H_out, W_out]
        self.pointwise = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(c2)

        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EHConv.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)

        return x