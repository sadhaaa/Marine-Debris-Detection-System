"""
Efficient Hybrid Squeeze Bottleneck used by SvelteNeck.
"""

import torch
import torch.nn as nn

from .ehconv import EHConv


class EHSBottleneck(nn.Module):
    """
    Efficient Hybrid Squeeze Bottleneck.

    Structure:

        Input
          │
          ▼
        1×1 Conv
          │
          ▼
        EHConv
          │
          ▼
        Output

    When input/output dimensions match and stride=1,
    a residual shortcut is added.

    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: Whether to allow residual connection.
        e: Channel expansion ratio.
        k: EHConv kernel size.
        s: Stride.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        e: float = 1.0,
        k: int = 5,
        s: int = 1,
    ) -> None:
        super().__init__()

        hidden_channels = int(c2 * e)

        self.cv1 = nn.Conv2d(
            c1,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.act1 = nn.Mish()

        self.ehconv = EHConv(
            c1=hidden_channels,
            c2=c2,
            k=k,
            s=s,
        )

        self.use_shortcut = (
            shortcut
            and c1 == c2
            and s == 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """

        identity = x

        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.ehconv(x)

        if self.use_shortcut:
            x = x + identity

        return x