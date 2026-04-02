"""
Apple Silicon MPS (Metal) sanity check: confirm PyTorch can use the Mac GPU.

Runs a tiny tensor op on ``mps`` to verify the backend works (not only device creation).
"""

from __future__ import annotations

from typing import Any

import torch


def verify_mps_gpu_ready() -> dict[str, Any]:
    """
    Verify PyTorch MPS is built, available, and can run a small op on the GPU.

    Prints a short report to stdout.

    Returns a dict with keys: mps_built, mps_available, pytorch_version.
    """
    if not torch.backends.mps.is_built():
        raise RuntimeError(
            "PyTorch was built without MPS support. Install a Mac build of PyTorch "
            "with Metal (MPS) enabled."
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available. Use an Apple Silicon Mac with a supported macOS "
            "and PyTorch version, or pass --device cpu."
        )

    # Minimal MPS compute to confirm the GPU path works (not just device creation)
    a = torch.randn(32, 32, device="mps", dtype=torch.float32)
    b = torch.randn(32, 32, device="mps", dtype=torch.float32)
    _ = (a @ b).sum().item()

    info = {
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "pytorch_version": torch.__version__,
    }
    print(
        f"[GPU check] MPS tensor matmul OK | PyTorch {info['pytorch_version']} | "
        f"device=mps"
    )
    return info
