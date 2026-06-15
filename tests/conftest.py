import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.gpu tests unless TESTBED_FORCE_GPU_TESTS=1.

    GPU tests load a real model (Qwen) and are slow — always opt-in explicitly
    so that 'pytest -q' stays fast regardless of hardware availability.
    Set TESTBED_FORCE_GPU_TESTS=1 to run them.
    """
    if os.environ.get("TESTBED_FORCE_GPU_TESTS") == "1":
        return
    skip_gpu = pytest.mark.skip(reason="opt-in only: set TESTBED_FORCE_GPU_TESTS=1")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
