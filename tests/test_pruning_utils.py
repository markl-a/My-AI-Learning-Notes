import sys
import os
import pytest

# Attempt to import torch; skip tests if unavailable or fails to load
try:
    import torch
except Exception as e:  # covers ImportError and OSError
    pytest.skip(f"torch not available: {e}", allow_module_level=True)

# Add path to import training_utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
UTILS_PATH = os.path.join(
    PROJECT_ROOT,
    '2.\u6df1\u5165LLM\u6a21\u578b\u5de5\u7a0b\u8207LLM\u904b\u7dad',
    'GaLore_Demo', 'GaLore-master', 'peft_pretraining'
)
sys.path.append(UTILS_PATH)

try:
    from training_utils import random_pruning, magnitude_pruning
except Exception as e:
    pytest.skip(f"training_utils import failed: {e}", allow_module_level=True)


def test_random_pruning_zero_ratio():
    torch.manual_seed(0)
    tensor = torch.ones(10000)
    pruned = random_pruning(tensor.clone(), 0.3)
    zero_ratio = (pruned == 0).float().mean().item()
    assert abs(zero_ratio - 0.3) < 0.05


def test_magnitude_pruning_smallest_elements_pruned():
    tensor = torch.tensor([0.1, -0.2, 0.3, -4.0, 5.0])
    pruned = magnitude_pruning(tensor.clone(), 0.4)
    magnitudes = torch.abs(tensor)
    threshold = torch.quantile(magnitudes.to(torch.float32), 0.4).to(tensor.dtype)
    for idx, value in enumerate(tensor):
        if torch.abs(value) > threshold:
            assert pruned[idx] == value
        else:
            assert pruned[idx] == 0
