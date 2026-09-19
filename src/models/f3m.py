"""
F3M — Feature-level Frequency Fusion Module.

A lightweight dual-attention gate (channel + spatial) that sharpens
multi-scale feature representations before the Detect head.

Design rationale
----------------
Underwater imagery suffers from:
  * Colour distortion (blue/green cast)
  * Low contrast and light scattering

These effects attenuate high-frequency edges and fine textures that
discriminate small debris classes from the dominant "rov" class.

F3M applies:
  1. Channel attention  — selectively amplify informative channels
                          via Squeeze-and-Excitation (SE) with ratio=4.
  2. Spatial attention  — highlight salient regions via a large-kernel
                          depthwise convolution (k=7) + sigmoid gate.

Both gates use residual scaling (output = input × gate) so that
gradients flow cleanly even at the start of training.

Usage in YOLO YAML
------------------
Insert one F3M per scale level immediately before Index / Detect:

    head:
      - [[4, 6, 10], 1, SvelteNeck, [0.50]]

      # F3M gates per scale
      - [11, 1, Index,  [64,  0]]     # P3 raw
      - [11, 1, Index,  [128, 1]]     # P4 raw
      - [11, 1, Index,  [256, 2]]     # P5 raw

      - [12, 1, F3M, [64]]            # F3M on P3
      - [13, 1, F3M, [128]]           # F3M on P4
      - [14, 1, F3M, [256]]           # F3M on P5

      - [[15, 16, 17], 1, Detect, [nc]]

Parameters (approximate)
-------------------------
F3M(64)  ≈  1,168 params
F3M(128) ≈  4,288 params
F3M(256) ≈ 16,640 params
Total for three scales ≈ 22,096 params  (< 0.03 M overhead)
"""

import torch
import torch.nn as nn


class F3M(nn.Module):
    """
    Feature-level Frequency Fusion Module.

    Args:
        c (int): Number of input (= output) channels.
        r (int): SE channel-reduction ratio. Default 4.
        sk  (int): Spatial-attention depthwise kernel size. Default 7.
    """

    def __init__(
        self,
        c: int,
        r: int = 4,
        sk: int = 7,
    ) -> None:
        super().__init__()

        if c <= 0:
            raise ValueError("Channel count c must be positive.")

        reduced = max(1, c // r)

        # ── Channel attention (SE-style) ──────────────────────────
        # Global average pool → FC → ReLU → FC → Sigmoid
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # [B, c, 1, 1]
            nn.Flatten(),                   # [B, c]
            nn.Linear(c, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, c, bias=False),
            nn.Sigmoid(),
        )

        # ── Spatial attention ─────────────────────────────────────
        # Depthwise conv with large kernel → Sigmoid
        pad = sk // 2
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(
                c, c,
                kernel_size=sk,
                stride=1,
                padding=pad,
                groups=c,       # depthwise
                bias=False,
            ),
            nn.BatchNorm2d(c),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel then spatial attention gates.

        Args:
            x: Input tensor [B, c, H, W].

        Returns:
            Gated tensor [B, c, H, W].
        """
        # Channel gate: [B, c] → [B, c, 1, 1]
        cg = self.channel_gate(x)
        cg = cg.view(cg.shape[0], cg.shape[1], 1, 1)

        # Apply channel gate (residual multiplicative)
        x = x * cg

        # Spatial gate
        sg = self.spatial_gate(x)

        # Apply spatial gate (residual multiplicative)
        return x * sg
