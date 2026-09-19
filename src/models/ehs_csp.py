"""
Efficient Hybrid Squeeze CSP (EHSCSP).

This module provides the feature-processing block used inside
the proposed SvelteNeck architecture.
"""

import torch
import torch.nn as nn

from .ehs_bottleneck import EHSBottleneck


class EHSCSP(nn.Module):
    """
    Efficient Hybrid Squeeze CSP block.

    The input is projected into two branches.

    Branch 1:
        Preserved feature information.

    Branch 2:
        Feature transformation through a sequence of
        EHSBottleneck blocks.

    The intermediate states are concatenated and fused
    using a final 1×1 convolution.

    Args:
        c1: Number of input channels.
        c2: Number of output channels.
        n: Number of EHSBottleneck blocks.
        shortcut: Enable residual connections inside bottlenecks.
        e: Expansion ratio.
        k: Kernel size used by EHConv.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 1.0,
        k: int = 5,
    ) -> None:
        super().__init__()

        if c1 <= 0 or c2 <= 0:
            raise ValueError(
                "Channel dimensions must be positive."
            )

        if n < 1:
            raise ValueError(
                "Number of bottlenecks n must be at least 1."
            )

        # Internal channel width.
        #
        # The projection produces 2 * hidden_channels.
        hidden_channels = int(c2 * e)

        self.hidden_channels = hidden_channels

        # Initial projection.
        #
        # [B, c1, H, W]
        #       ↓
        # [B, 2*hidden, H, W]
        self.cv1 = nn.Conv2d(
            c1,
            2 * hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            2 * hidden_channels
        )

        self.act1 = nn.Mish()

        # Transformed branch.
        #
        # Each bottleneck maintains hidden_channels.
        self.blocks = nn.ModuleList(
            [
                EHSBottleneck(
                    c1=hidden_channels,
                    c2=hidden_channels,
                    shortcut=shortcut,
                    e=1.0,
                    k=k,
                    s=1,
                )
                for _ in range(n)
            ]
        )

        # Final feature fusion.
        #
        # We concatenate:
        #
        #   preserved branch
        #   + transformed states
        #
        # This creates:
        #
        #   (n + 2) * hidden_channels
        #
        # channels.
        fusion_channels = (
            (n + 2) * hidden_channels
        )

        self.cv2 = nn.Conv2d(
            fusion_channels,
            c2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(c2)

        self.act2 = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EHSCSP.
        """

        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Split projected tensor into two equal branches.
        preserved, transformed = torch.chunk(
            x,
            chunks=2,
            dim=1,
        )

        states = [
            preserved,
            transformed,
        ]

        # Process only the transformed branch.
        for block in self.blocks:
            transformed = block(transformed)
            states.append(transformed)

        # Concatenate preserved and transformed states.
        x = torch.cat(
            states,
            dim=1,
        )

        x = self.cv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x