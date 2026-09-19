"""
SvelteNeck adapted for YOLO26n.

The module receives the three YOLO26 backbone feature levels:

    P3: [B, 128, 80, 80]
    P4: [B, 128, 40, 40]
    P5: [B, 256, 20, 20]

Ultralytics passes multiple `from` layers as a list:

    [P3, P4, P5]

The module returns:

    P3: [B, 64, 80, 80]
    P4: [B, 128, 40, 40]
    P5: [B, 256, 20, 20]
"""

import torch
import torch.nn as nn

from .ehconv import EHConv
from .ehs_csp import EHSCSP


class SvelteNeck(nn.Module):
    """
    Lightweight multi-scale feature-fusion neck.

    The architecture performs:

        Top-down pathway:
            P5 -> P4 -> P3

        Bottom-up pathway:
            P3 -> P4 -> P5

    Args:
        e: Expansion ratio used inside EHSCSP blocks.
    """

    def __init__(self, e: float = 0.75) -> None:
        super().__init__()

        self.upsample_p5 = nn.Upsample(
            scale_factor=2,
            mode="nearest",
        )

        self.td_p4 = EHSCSP(
            c1=384,
            c2=128,
            n=1,
            shortcut=True,
            e=e,
            k=5,
        )

        self.upsample_p4 = nn.Upsample(
            scale_factor=2,
            mode="nearest",
        )

        self.td_p3 = EHSCSP(
            c1=256,
            c2=64,
            n=1,
            shortcut=True,
            e=e,
            k=5,
        )

        self.bu_p3_down = EHConv(
            c1=64,
            c2=64,
            k=3,
            s=2,
        )

        self.bu_p4 = EHSCSP(
            c1=192,
            c2=128,
            n=1,
            shortcut=True,
            e=e,
            k=5,
        )

        self.bu_p4_down = EHConv(
            c1=128,
            c2=128,
            k=3,
            s=2,
        )

        self.bu_p5 = EHSCSP(
            c1=384,
            c2=256,
            n=1,
            shortcut=True,
            e=e,
            k=5,
        )

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Fuse the three YOLO26 backbone feature levels.

        Args:
            features:
                List containing P3, P4, and P5 tensors.

        Returns:
            List containing the fused P3, P4, and P5 tensors.
        """

        if len(features) != 3:
            raise ValueError(
                "SvelteNeck expects exactly three feature maps: "
                "P3, P4, and P5."
            )

        # Ultralytics supplies multi-input layers as a list.
        p3, p4, p5 = features

        # ---------------------------------------------------------
        # Top-down pathway
        # ---------------------------------------------------------

        # P5:
        # [B, 256, 20, 20]
        #
        # Upsample:
        # [B, 256, 40, 40]
        p5_up = self.upsample_p5(p5)

        # Concatenate with backbone P4:
        #
        # [B, 256, 40, 40]
        # +
        # [B, 128, 40, 40]
        #
        # = [B, 384, 40, 40]
        p4_td = self.td_p4(
            torch.cat(
                [p5_up, p4],
                dim=1,
            )
        )

        # P4 top-down output:
        # [B, 128, 40, 40]
        #
        # Upsample to P3 resolution:
        # [B, 128, 80, 80]
        p4_up = self.upsample_p4(p4_td)

        # Concatenate with backbone P3:
        #
        # [B, 128, 80, 80]
        # +
        # [B, 128, 80, 80]
        #
        # = [B, 256, 80, 80]
        p3_td = self.td_p3(
            torch.cat(
                [p4_up, p3],
                dim=1,
            )
        )

        # P3 detection feature:
        # [B, 64, 80, 80]

        # ---------------------------------------------------------
        # Bottom-up pathway
        # ---------------------------------------------------------

        # Downsample P3:
        #
        # [B, 64, 80, 80]
        #       ↓
        # [B, 64, 40, 40]
        p3_down = self.bu_p3_down(p3_td)

        # Fuse bottom-up P3 with top-down P4:
        #
        # [B, 64, 40, 40]
        # +
        # [B, 128, 40, 40]
        #
        # = [B, 192, 40, 40]
        p4_bu = self.bu_p4(
            torch.cat(
                [p3_down, p4_td],
                dim=1,
            )
        )

        # P4 detection feature:
        # [B, 128, 40, 40]

        # Downsample P4:
        #
        # [B, 128, 40, 40]
        #       ↓
        # [B, 128, 20, 20]
        p4_down = self.bu_p4_down(p4_bu)

        # Fuse bottom-up P4 with original backbone P5:
        #
        # [B, 128, 20, 20]
        # +
        # [B, 256, 20, 20]
        #
        # = [B, 384, 20, 20]
        p5_bu = self.bu_p5(
            torch.cat(
                [p4_down, p5],
                dim=1,
            )
        )

        # P5 detection feature:
        # [B, 256, 20, 20]

        return [
            p3_td,
            p4_bu,
            p5_bu,
        ]