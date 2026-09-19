"""
Unit tests for EHConv.
"""

import torch
import torch.nn as nn

from ehconv import EHConv


def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print("=" * 80)
    print("EHCONV VALIDATION")
    print("=" * 80)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print()

    # ------------------------------------------------------------
    # Test 1: Standard spatial processing
    # 128 channels, 5×5 kernel, stride 1
    # ------------------------------------------------------------

    ehconv_1 = EHConv(
        c1=128,
        c2=128,
        k=5,
        s=1,
    ).to(device)

    x1 = torch.randn(
        1,
        128,
        40,
        40,
        device=device,
    )

    with torch.no_grad():
        y1 = ehconv_1(x1)

    print("Test 1")
    print("-" * 40)
    print(f"Input shape      : {tuple(x1.shape)}")
    print(f"Output shape     : {tuple(y1.shape)}")
    print(f"Parameters       : {count_parameters(ehconv_1):,}")

    assert y1.shape == (1, 128, 40, 40)

    print("Shape test       : PASS")
    print()

    # ------------------------------------------------------------
    # Test 2: Downsampling
    # 64 channels, 3×3 kernel, stride 2
    # ------------------------------------------------------------

    ehconv_2 = EHConv(
        c1=64,
        c2=64,
        k=3,
        s=2,
    ).to(device)

    x2 = torch.randn(
        1,
        64,
        80,
        80,
        device=device,
    )

    with torch.no_grad():
        y2 = ehconv_2(x2)

    print("Test 2")
    print("-" * 40)
    print(f"Input shape      : {tuple(x2.shape)}")
    print(f"Output shape     : {tuple(y2.shape)}")
    print(f"Parameters       : {count_parameters(ehconv_2):,}")

    assert y2.shape == (1, 64, 40, 40)

    print("Shape test       : PASS")
    print()

    # ------------------------------------------------------------
    # Test 3: Compare EHConv with standard convolution
    # ------------------------------------------------------------

    standard_conv = nn.Conv2d(
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False,
    ).to(device)

    eh_params = count_parameters(ehconv_2)
    standard_params = count_parameters(standard_conv)

    print("Test 3")
    print("-" * 40)
    print(f"EHConv parameters       : {eh_params:,}")
    print(f"Standard Conv parameters: {standard_params:,}")

    reduction = (
        1.0 - (eh_params / standard_params)
    ) * 100.0

    print(f"Parameter reduction     : {reduction:.2f}%")

    assert eh_params < standard_params

    print("Efficiency test : PASS")
    print()

    # ------------------------------------------------------------
    # Test 4: Check output for invalid numerical values
    # ------------------------------------------------------------

    assert torch.isfinite(y1).all()
    assert torch.isfinite(y2).all()

    print("Numerical stability test: PASS")
    print()

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    print("=" * 80)
    print("EHCONV VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()