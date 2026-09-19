"""
Unit tests for EHSBottleneck.
"""

import torch
import torch.nn as nn

from ehs_bottleneck import EHSBottleneck


def count_parameters(module: nn.Module) -> int:
    """Return trainable parameter count."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print("=" * 80)
    print("EHSBOTTLENECK VALIDATION")
    print("=" * 80)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print()

    # ------------------------------------------------------------
    # Test 1: Residual bottleneck
    # ------------------------------------------------------------

    block = EHSBottleneck(
        c1=128,
        c2=128,
        shortcut=True,
        e=1.0,
        k=5,
        s=1,
    ).to(device)

    x = torch.randn(
        1,
        128,
        40,
        40,
        device=device,
    )

    with torch.no_grad():
        y = block(x)

    print("Test 1 — Residual bottleneck")
    print("-" * 40)
    print(f"Input shape      : {tuple(x.shape)}")
    print(f"Output shape     : {tuple(y.shape)}")
    print(f"Parameters       : {count_parameters(block):,}")
    print(f"Shortcut enabled : {block.use_shortcut}")

    assert y.shape == x.shape
    assert block.use_shortcut

    print("Shape test       : PASS")
    print("Shortcut test    : PASS")
    print()

    # ------------------------------------------------------------
    # Test 2: Channel-changing bottleneck
    # ------------------------------------------------------------

    block_2 = EHSBottleneck(
        c1=128,
        c2=64,
        shortcut=True,
        e=1.0,
        k=5,
        s=1,
    ).to(device)

    x2 = torch.randn(
        1,
        128,
        40,
        40,
        device=device,
    )

    with torch.no_grad():
        y2 = block_2(x2)

    print("Test 2 — Channel-changing bottleneck")
    print("-" * 40)
    print(f"Input shape      : {tuple(x2.shape)}")
    print(f"Output shape     : {tuple(y2.shape)}")
    print(f"Parameters       : {count_parameters(block_2):,}")
    print(f"Shortcut enabled : {block_2.use_shortcut}")

    assert y2.shape == (1, 64, 40, 40)
    assert not block_2.use_shortcut

    print("Shape test       : PASS")
    print("Shortcut test    : PASS")
    print()

    # ------------------------------------------------------------
    # Test 3: Stride-2 bottleneck
    # ------------------------------------------------------------

    block_3 = EHSBottleneck(
        c1=64,
        c2=64,
        shortcut=True,
        e=1.0,
        k=3,
        s=2,
    ).to(device)

    x3 = torch.randn(
        1,
        64,
        80,
        80,
        device=device,
    )

    with torch.no_grad():
        y3 = block_3(x3)

    print("Test 3 — Stride-2 bottleneck")
    print("-" * 40)
    print(f"Input shape      : {tuple(x3.shape)}")
    print(f"Output shape     : {tuple(y3.shape)}")
    print(f"Parameters       : {count_parameters(block_3):,}")
    print(f"Shortcut enabled : {block_3.use_shortcut}")

    assert y3.shape == (1, 64, 40, 40)
    assert not block_3.use_shortcut

    print("Shape test       : PASS")
    print("Shortcut test    : PASS")
    print()

    # ------------------------------------------------------------
    # Test 4: Numerical stability
    # ------------------------------------------------------------

    assert torch.isfinite(y).all()
    assert torch.isfinite(y2).all()
    assert torch.isfinite(y3).all()

    print("Numerical stability test: PASS")
    print()

    print("=" * 80)
    print("EHSBOTTLENECK VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()