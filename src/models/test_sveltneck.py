"""
Standalone validation for SvelteNeck.
"""

import torch
import torch.nn as nn

from sveltneck import SvelteNeck


def count_parameters(module: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print("=" * 80)
    print("SVELTENECK VALIDATION")
    print("=" * 80)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print()

    neck = SvelteNeck(
        e=0.75,
    ).to(device)

    neck.eval()

    # Exact feature-map interfaces obtained from YOLO26n.
    p3 = torch.randn(
        1,
        128,
        80,
        80,
        device=device,
    )

    p4 = torch.randn(
        1,
        128,
        40,
        40,
        device=device,
    )

    p5 = torch.randn(
        1,
        256,
        20,
        20,
        device=device,
    )

    print("Input feature maps")
    print("-" * 40)
    print(f"P3: {tuple(p3.shape)}")
    print(f"P4: {tuple(p4.shape)}")
    print(f"P5: {tuple(p5.shape)}")
    print()

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------

    with torch.no_grad():
        p3_out, p4_out, p5_out = neck(
            p3,
            p4,
            p5,
        )

    print("Output feature maps")
    print("-" * 40)
    print(f"P3: {tuple(p3_out.shape)}")
    print(f"P4: {tuple(p4_out.shape)}")
    print(f"P5: {tuple(p5_out.shape)}")
    print()

    # ------------------------------------------------------------
    # Interface validation
    # ------------------------------------------------------------

    assert p3_out.shape == (
        1,
        64,
        80,
        80,
    )

    assert p4_out.shape == (
        1,
        128,
        40,
        40,
    )

    assert p5_out.shape == (
        1,
        256,
        20,
        20,
    )

    print("P3 interface test : PASS")
    print("P4 interface test : PASS")
    print("P5 interface test : PASS")
    print()

    # ------------------------------------------------------------
    # Numerical validation
    # ------------------------------------------------------------

    assert torch.isfinite(p3_out).all()
    assert torch.isfinite(p4_out).all()
    assert torch.isfinite(p5_out).all()

    print("Numerical stability test: PASS")
    print()

    # ------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------

    total_params = count_parameters(neck)

    print("Parameter budget")
    print("-" * 40)
    print(f"SvelteNeck parameters : {total_params:,}")
    print(
        f"SvelteNeck parameters : "
        f"{total_params / 1e6:.3f} M"
    )

    native_neck_params = 897_152

    reduction = (
        1.0
        - total_params / native_neck_params
    ) * 100.0

    print(
        f"Compared with native neck: "
        f"{reduction:.2f}% reduction"
    )
    print()

    # ------------------------------------------------------------
    # Estimated complete model parameter count
    # ------------------------------------------------------------

    native_total = 2_572_280

    estimated_total = (
        native_total
        - native_neck_params
        + total_params
    )

    print("Estimated complete YOLO26 + SvelteNeck")
    print("-" * 40)
    print(
        f"Estimated parameters : "
        f"{estimated_total:,}"
    )
    print(
        f"Estimated parameters : "
        f"{estimated_total / 1e6:.3f} M"
    )
    print()

    print("=" * 80)
    print("SVELTENECK VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()