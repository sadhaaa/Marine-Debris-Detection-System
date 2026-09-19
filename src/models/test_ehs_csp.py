"""
Unit tests for EHSCSP.
"""

import torch
import torch.nn as nn

from ehs_csp import EHSCSP


def count_parameters(module: nn.Module) -> int:
    """Return trainable parameter count."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print("=" * 80)
    print("EHSCSP VALIDATION")
    print("=" * 80)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print()

    # ------------------------------------------------------------
    # Test 1
    # 384 → 128
    # This matches the first top-down fusion point.
    # ------------------------------------------------------------

    block_1 = EHSCSP(
        c1=384,
        c2=128,
        n=1,
        shortcut=True,
        e=1.0,
        k=5,
    ).to(device)

    x1 = torch.randn(
        1,
        384,
        40,
        40,
        device=device,
    )

    with torch.no_grad():
        y1 = block_1(x1)

    print("Test 1 — Top-down P4 fusion")
    print("-" * 40)
    print(f"Input shape      : {tuple(x1.shape)}")
    print(f"Output shape     : {tuple(y1.shape)}")
    print(f"Parameters       : {count_parameters(block_1):,}")

    assert y1.shape == (1, 128, 40, 40)

    print("Shape test       : PASS")
    print()

    # ------------------------------------------------------------
    # Test 2
    # 256 → 64
    # This matches the second top-down fusion point.
    # ------------------------------------------------------------

    block_2 = EHSCSP(
        c1=256,
        c2=64,
        n=1,
        shortcut=True,
        e=1.0,
        k=5,
    ).to(device)

    x2 = torch.randn(
        1,
        256,
        80,
        80,
        device=device,
    )

    with torch.no_grad():
        y2 = block_2(x2)

    print("Test 2 — Top-down P3 fusion")
    print("-" * 40)
    print(f"Input shape      : {tuple(x2.shape)}")
    print(f"Output shape     : {tuple(y2.shape)}")
    print(f"Parameters       : {count_parameters(block_2):,}")

    assert y2.shape == (1, 64, 80, 80)

    print("Shape test       : PASS")
    print()

    # ------------------------------------------------------------
    # Test 3
    # 192 → 128
    # Matches bottom-up P4 fusion.
    # ------------------------------------------------------------

    block_3 = EHSCSP(
        c1=192,
        c2=128,
        n=1,
        shortcut=True,
        e=1.0,
        k=5,
    ).to(device)

    x3 = torch.randn(
        1,
        192,
        40,
        40,
        device=device,
    )

    with torch.no_grad():
        y3 = block_3(x3)

    print("Test 3 — Bottom-up P4 fusion")
    print("-" * 40)
    print(f"Input shape      : {tuple(x3.shape)}")
    print(f"Output shape     : {tuple(y3.shape)}")
    print(f"Parameters       : {count_parameters(block_3):,}")

    assert y3.shape == (1, 128, 40, 40)

    print("Shape test       : PASS")
    print()

    # ------------------------------------------------------------
    # Test 4
    # 384 → 256
    # Matches bottom-up P5 fusion.
    # ------------------------------------------------------------

    block_4 = EHSCSP(
        c1=384,
        c2=256,
        n=1,
        shortcut=True,
        e=1.0,
        k=5,
    ).to(device)

    x4 = torch.randn(
        1,
        384,
        20,
        20,
        device=device,
    )

    with torch.no_grad():
        y4 = block_4(x4)

    print("Test 4 — Bottom-up P5 fusion")
    print("-" * 40)
    print(f"Input shape      : {tuple(x4.shape)}")
    print(f"Output shape     : {tuple(y4.shape)}")
    print(f"Parameters       : {count_parameters(block_4):,}")

    assert y4.shape == (1, 256, 20, 20)

    print("Shape test       : PASS")
    print()

    # ------------------------------------------------------------
    # Test 5: Numerical stability
    # ------------------------------------------------------------

    assert torch.isfinite(y1).all()
    assert torch.isfinite(y2).all()
    assert torch.isfinite(y3).all()
    assert torch.isfinite(y4).all()

    print("Numerical stability test: PASS")
    print()

    print("=" * 80)
    print("EHSCSP VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()