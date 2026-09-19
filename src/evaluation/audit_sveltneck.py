"""
Computational audit for YOLO26n vs SvelteNeck.

This script compares:
    1. Native YOLO26n neck
    2. Proposed SvelteNeck

It measures:
    - Trainable parameters
    - MACs
    - GFLOPs
    - Parameter reduction
    - GFLOP reduction
    - Feature-map interface compatibility
    - Numerical stability

The native YOLO26n backbone and Detect head are treated as frozen.
Only the neck is replaced in the proposed design.
"""

from pathlib import Path
import sys

import torch
from thop import profile
from ultralytics import YOLO


# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_MODELS = PROJECT_ROOT / "src" / "models"

if str(SRC_MODELS) not in sys.path:
    sys.path.insert(0, str(SRC_MODELS))


from sveltneck import SvelteNeck


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

MODEL_NAME = "yolo26n.pt"

IMAGE_SIZE = 640

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float32


# ---------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------

def count_parameters(module: torch.nn.Module) -> int:
    """
    Count all trainable and non-trainable parameters.
    """
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def gflops_from_macs(macs: float) -> float:
    """
    Convert MACs to GFLOPs.

    THOP reports MACs.
    Convention used here:

        1 MAC = 2 FLOPs

    Therefore:

        GFLOPs = MACs * 2 / 1e9
    """
    return (macs * 2.0) / 1e9


def print_separator() -> None:
    print("=" * 80)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:

    print_separator()
    print("YOLO26n vs SvelteNeck COMPUTATIONAL AUDIT")
    print_separator()

    print(f"Device: {DEVICE}")
    print(f"Input : {IMAGE_SIZE} x {IMAGE_SIZE}")

    if DEVICE == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    print()

    # -----------------------------------------------------------------
    # LOAD YOLO26n
    # -----------------------------------------------------------------

    print("Loading YOLO26n...")

    yolo = YOLO(MODEL_NAME)

    model = yolo.model.to(DEVICE)
    model.eval()

    print("YOLO26n loaded successfully.")
    print()

    # -----------------------------------------------------------------
    # TOTAL YOLO26n PARAMETERS
    # -----------------------------------------------------------------

    native_total_params = count_parameters(model)

    if native_total_params <= 0:
        raise RuntimeError(
            "Unable to obtain native YOLO26n parameter count."
        )

    print("YOLO26n TOTAL PARAMETERS")
    print("-" * 60)

    print(
        f"Total parameters : {native_total_params:,}"
    )

    print(
        f"Total parameters : "
        f"{native_total_params / 1e6:.3f} M"
    )

    print()

    # -----------------------------------------------------------------
    # EXTRACT BACKBONE FEATURES
    # -----------------------------------------------------------------

    print("Extracting YOLO26n backbone feature interfaces...")
    print()

    input_tensor = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
        dtype=DTYPE,
    )

    saved_features = {}

    hooks = []

    def make_hook(layer_index: int):
        def hook(module, inputs, output):
            saved_features[layer_index] = output

        return hook

    # YOLO26n backbone interfaces:
    #
    # Layer 4  -> P3 = 128 x 80 x 80
    # Layer 6  -> P4 = 128 x 40 x 40
    # Layer 10 -> P5 = 256 x 20 x 20

    for layer_index in [4, 6, 10]:

        hook = model.model[layer_index].register_forward_hook(
            make_hook(layer_index)
        )

        hooks.append(hook)

    with torch.no_grad():
        _ = model(input_tensor)

    for hook in hooks:
        hook.remove()

    p3 = saved_features[4]
    p4 = saved_features[6]
    p5 = saved_features[10]

    print("Backbone interfaces")
    print("-" * 60)

    print(f"P3: {tuple(p3.shape)}")
    print(f"P4: {tuple(p4.shape)}")
    print(f"P5: {tuple(p5.shape)}")

    print()

    # -----------------------------------------------------------------
    # NATIVE YOLO26n NECK
    # -----------------------------------------------------------------

    print_separator()
    print("NATIVE YOLO26n NECK")
    print_separator()

    native_neck_layers = {
        11: model.model[11],
        12: model.model[12],
        13: model.model[13],
        14: model.model[14],
        15: model.model[15],
        16: model.model[16],
        17: model.model[17],
        18: model.model[18],
        19: model.model[19],
        20: model.model[20],
        21: model.model[21],
        22: model.model[22],
    }

    native_neck_params = 0
    native_neck_macs = 0.0

    for layer_index, layer in native_neck_layers.items():

        layer_params = count_parameters(layer)

        native_neck_params += layer_params

        # -------------------------------------------------------------
        # Profile only computational layers.
        # Upsample and Concat do not have trainable parameters.
        # -------------------------------------------------------------

        if layer_index in [13, 16, 17, 19, 20, 22]:

            try:

                if layer_index == 13:
                    dummy = torch.randn(
                        1,
                        384,
                        40,
                        40,
                        device=DEVICE,
                    )

                elif layer_index == 16:
                    dummy = torch.randn(
                        1,
                        256,
                        80,
                        80,
                        device=DEVICE,
                    )

                elif layer_index == 17:
                    dummy = torch.randn(
                        1,
                        64,
                        80,
                        80,
                        device=DEVICE,
                    )

                elif layer_index == 19:
                    dummy = torch.randn(
                        1,
                        192,
                        40,
                        40,
                        device=DEVICE,
                    )

                elif layer_index == 20:
                    dummy = torch.randn(
                        1,
                        128,
                        40,
                        40,
                        device=DEVICE,
                    )

                elif layer_index == 22:
                    dummy = torch.randn(
                        1,
                        384,
                        20,
                        20,
                        device=DEVICE,
                    )

                macs, _ = profile(
                    layer,
                    inputs=(dummy,),
                    verbose=False,
                )

                layer_macs = float(macs)

                native_neck_macs += layer_macs

                layer_gflops = gflops_from_macs(
                    layer_macs
                )

                print(
                    f"Layer {layer_index:02d} | "
                    f"{layer.__class__.__name__:10s} | "
                    f"params={layer_params:,} | "
                    f"GFLOPs={layer_gflops:.3f}"
                )

            except Exception as error:

                print(
                    f"Layer {layer_index:02d} | "
                    f"{layer.__class__.__name__:10s} | "
                    f"params={layer_params:,} | "
                    f"profiling failed: {error}"
                )

        else:

            print(
                f"Layer {layer_index:02d} | "
                f"{layer.__class__.__name__:10s} | "
                f"params={layer_params:,}"
            )

    native_neck_gflops = gflops_from_macs(
        native_neck_macs
    )

    print()

    print("Native YOLO26n neck")
    print("-" * 60)

    print(
        f"Parameters : {native_neck_params:,}"
    )

    print(
        f"Parameters : "
        f"{native_neck_params / 1e6:.3f} M"
    )

    print(
        f"MACs       : "
        f"{native_neck_macs / 1e9:.3f} G"
    )

    print(
        f"GFLOPs     : "
        f"{native_neck_gflops:.3f} G"
    )

    print()

    # -----------------------------------------------------------------
    # SVELTENECK
    # -----------------------------------------------------------------

    print_separator()
    print("SVELTENECK")
    print_separator()

    svelte_neck = SvelteNeck(
        e=0.75
    ).to(DEVICE)

    svelte_neck.eval()

    svelte_params = count_parameters(
        svelte_neck
    )

    # -------------------------------------------------------------
    # Profile SvelteNeck as one complete module.
    #
    # Inputs:
    #     P3 = 128 x 80 x 80
    #     P4 = 128 x 40 x 40
    #     P5 = 256 x 20 x 20
    # -------------------------------------------------------------

    with torch.no_grad():

        svelte_macs, _ = profile(
            svelte_neck,
            inputs=(
                p3,
                p4,
                p5,
            ),
            verbose=False,
        )

    svelte_macs = float(svelte_macs)

    svelte_gflops = gflops_from_macs(
        svelte_macs
    )

    print("SvelteNeck")
    print("-" * 60)

    print(
        f"Parameters : {svelte_params:,}"
    )

    print(
        f"Parameters : "
        f"{svelte_params / 1e6:.3f} M"
    )

    print(
        f"MACs       : "
        f"{svelte_macs / 1e9:.3f} G"
    )

    print(
        f"GFLOPs     : "
        f"{svelte_gflops:.3f} G"
    )

    print()

    # -----------------------------------------------------------------
    # PARAMETER COMPARISON
    # -----------------------------------------------------------------

    parameter_reduction = (
        (native_neck_params - svelte_params)
        / native_neck_params
    ) * 100.0

    # -----------------------------------------------------------------
    # GFLOP COMPARISON
    # -----------------------------------------------------------------

    gflop_reduction = (
        (native_neck_gflops - svelte_gflops)
        / native_neck_gflops
    ) * 100.0

    print_separator()
    print("EFFICIENCY COMPARISON")
    print_separator()

    print(
        f"Native neck parameters : "
        f"{native_neck_params:,}"
    )

    print(
        f"SvelteNeck parameters  : "
        f"{svelte_params:,}"
    )

    print(
        f"Parameter reduction    : "
        f"{parameter_reduction:.2f}%"
    )

    print()

    print(
        f"Native neck GFLOPs     : "
        f"{native_neck_gflops:.3f}"
    )

    print(
        f"SvelteNeck GFLOPs      : "
        f"{svelte_gflops:.3f}"
    )

    print(
        f"GFLOP reduction        : "
        f"{gflop_reduction:.2f}%"
    )

    print()

    # -----------------------------------------------------------------
    # ESTIMATED COMPLETE MODEL PARAMETER COUNT
    # -----------------------------------------------------------------

    estimated_svelte_total = (
        native_total_params
        - native_neck_params
        + svelte_params
    )

    print_separator()
    print("ESTIMATED COMPLETE YOLO26n + SVELTENECK")
    print_separator()

    print(
        f"Native YOLO26n parameters : "
        f"{native_total_params:,}"
    )

    print(
        f"Removed native neck       : "
        f"{native_neck_params:,}"
    )

    print(
        f"Added SvelteNeck          : "
        f"{svelte_params:,}"
    )

    print(
        f"Estimated final params    : "
        f"{estimated_svelte_total:,}"
    )

    print(
        f"Estimated final params    : "
        f"{estimated_svelte_total / 1e6:.3f} M"
    )

    print()

    # -----------------------------------------------------------------
    # FEATURE INTERFACE VALIDATION
    # -----------------------------------------------------------------

    print_separator()
    print("FEATURE INTERFACE VALIDATION")
    print_separator()

    with torch.no_grad():

        svelte_p3, svelte_p4, svelte_p5 = (
            svelte_neck(
                p3,
                p4,
                p5,
            )
        )

    print(
        f"SvelteNeck P3 output : "
        f"{tuple(svelte_p3.shape)}"
    )

    print(
        f"SvelteNeck P4 output : "
        f"{tuple(svelte_p4.shape)}"
    )

    print(
        f"SvelteNeck P5 output : "
        f"{tuple(svelte_p5.shape)}"
    )

    expected_p3 = (
        1,
        64,
        80,
        80,
    )

    expected_p4 = (
        1,
        128,
        40,
        40,
    )

    expected_p5 = (
        1,
        256,
        20,
        20,
    )

    p3_pass = tuple(svelte_p3.shape) == expected_p3
    p4_pass = tuple(svelte_p4.shape) == expected_p4
    p5_pass = tuple(svelte_p5.shape) == expected_p5

    print()

    print(
        f"P3 interface test : "
        f"{'PASS' if p3_pass else 'FAIL'}"
    )

    print(
        f"P4 interface test : "
        f"{'PASS' if p4_pass else 'FAIL'}"
    )

    print(
        f"P5 interface test : "
        f"{'PASS' if p5_pass else 'FAIL'}"
    )

    if not all(
        [
            p3_pass,
            p4_pass,
            p5_pass,
        ]
    ):
        raise RuntimeError(
            "SvelteNeck feature interface "
            "validation failed."
        )

    print()

    # -----------------------------------------------------------------
    # NUMERICAL STABILITY
    # -----------------------------------------------------------------

    print_separator()
    print("NUMERICAL STABILITY TEST")
    print_separator()

    finite_outputs = all(
        torch.isfinite(output).all().item()
        for output in [
            svelte_p3,
            svelte_p4,
            svelte_p5,
        ]
    )

    print(
        f"Numerical stability : "
        f"{'PASS' if finite_outputs else 'FAIL'}"
    )

    if not finite_outputs:
        raise RuntimeError(
            "SvelteNeck produced NaN or Inf values."
        )

    print()

    # -----------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------

    print_separator()
    print("FINAL AUDIT SUMMARY")
    print_separator()

    print(
        f"Native neck parameters : "
        f"{native_neck_params / 1e6:.3f} M"
    )

    print(
        f"SvelteNeck parameters  : "
        f"{svelte_params / 1e6:.3f} M"
    )

    print(
        f"Parameter reduction    : "
        f"{parameter_reduction:.2f}%"
    )

    print()

    print(
        f"Native neck GFLOPs     : "
        f"{native_neck_gflops:.3f}"
    )

    print(
        f"SvelteNeck GFLOPs      : "
        f"{svelte_gflops:.3f}"
    )

    print(
        f"GFLOP reduction        : "
        f"{gflop_reduction:.2f}%"
    )

    print()

    print(
        f"Estimated complete model : "
        f"{estimated_svelte_total / 1e6:.3f} M parameters"
    )

    print()

    print_separator()
    print("COMPUTATIONAL AUDIT COMPLETE")
    print_separator()


if __name__ == "__main__":
    main()