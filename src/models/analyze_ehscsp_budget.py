"""
Analyze EHSCSP parameter budgets for different expansion ratios.
"""

import torch.nn as nn

from ehs_csp import EHSCSP


def count_parameters(module: nn.Module) -> int:
    """Return the total number of parameters."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def main() -> None:
    configurations = [
        ("P4 top-down", 384, 128),
        ("P3 top-down", 256, 64),
        ("P4 bottom-up", 192, 128),
        ("P5 bottom-up", 384, 256),
    ]

    expansion_ratios = [
        0.25,
        0.50,
        0.75,
        1.00,
    ]

    print("=" * 90)
    print("EHSCSP PARAMETER BUDGET ANALYSIS")
    print("=" * 90)

    print(
        f"{'Block':<20}"
        f"{'c1':>8}"
        f"{'c2':>8}"
        f"{'e':>8}"
        f"{'hidden':>10}"
        f"{'params':>15}"
    )

    print("-" * 90)

    for name, c1, c2 in configurations:

        for e in expansion_ratios:

            block = EHSCSP(
                c1=c1,
                c2=c2,
                n=1,
                shortcut=True,
                e=e,
                k=5,
            )

            params = count_parameters(block)

            hidden = int(c2 * e)

            print(
                f"{name:<20}"
                f"{c1:>8}"
                f"{c2:>8}"
                f"{e:>8.2f}"
                f"{hidden:>10}"
                f"{params:>15,}"
            )

        print("-" * 90)

    print()
    print("Native YOLO26n neck:")
    print("897,152 parameters")
    print()


if __name__ == "__main__":
    main()