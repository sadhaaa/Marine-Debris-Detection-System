"""
Display the software and hardware environment used for
the SvelteNeck-YOLO26 research project.

This information should be recorded before baseline training
to ensure experiments are reproducible.
"""

import platform
import sys

import torch
import ultralytics


def main() -> None:
    """Print complete project environment information."""

    print("=" * 70)
    print("DL Project Marine - Environment Verification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Python
    # ------------------------------------------------------------------
    print("\n[PYTHON]")
    print(f"Python version : {sys.version}")
    print(f"Python path    : {sys.executable}")

    # ------------------------------------------------------------------
    # Operating system
    # ------------------------------------------------------------------
    print("\n[OPERATING SYSTEM]")
    print(f"System         : {platform.system()}")
    print(f"Release        : {platform.release()}")
    print(f"Platform       : {platform.platform()}")

    # ------------------------------------------------------------------
    # PyTorch
    # ------------------------------------------------------------------
    print("\n[PYTORCH]")
    print(f"PyTorch version: {torch.__version__}")

    # ------------------------------------------------------------------
    # CUDA
    # ------------------------------------------------------------------
    print("\n[CUDA]")
    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"PyTorch CUDA   : {torch.version.cuda}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()

        print(f"GPU count      : {gpu_count}")

        for gpu_index in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(gpu_index)
            gpu_properties = torch.cuda.get_device_properties(
                gpu_index
            )

            total_memory_gb = (
                gpu_properties.total_memory
                / (1024 ** 3)
            )

            print(f"\nGPU {gpu_index}")
            print(f"Name           : {gpu_name}")
            print(f"VRAM           : {total_memory_gb:.2f} GB")
            print(f"Compute Cap.   : "
                  f"{gpu_properties.major}."
                  f"{gpu_properties.minor}")

    else:
        print("WARNING: CUDA is NOT available.")
        print("Training will not use an NVIDIA GPU.")

    # ------------------------------------------------------------------
    # Ultralytics
    # ------------------------------------------------------------------
    print("\n[ULTRALYTICS]")
    print(f"Ultralytics    : {ultralytics.__version__}")

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)

    if torch.cuda.is_available():
        print("✓ CUDA GPU environment detected")
        print("✓ Environment is ready for GPU verification")
    else:
        print("⚠ CUDA GPU not detected")
        print("⚠ Do NOT start baseline training yet")

    print("=" * 70)


if __name__ == "__main__":
    main()