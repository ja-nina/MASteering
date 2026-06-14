import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def _gpu_available() -> bool:
    if os.environ.get("TESTBED_FORCE_GPU_TESTS") == "1":
        return True
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.gpu tests unless a GPU is available.

    These tests load a real model (Qwen) which needs CUDA and a modern
    transformers install. Set TESTBED_FORCE_GPU_TESTS=1 to force them on.
    """
    if _gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires GPU / modern transformers (CPU-only env)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
