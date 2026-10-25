import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_remote_rmu import masked_mse, attach_rmu


def test_masked_mse_ignores_prompt_and_backpropagates_answer():
    current = torch.tensor([[[100., 100.], [2., 4.]]], requires_grad=True)
    result = masked_mse(current, torch.zeros_like(current), torch.tensor([[False, True]]))
    assert result.item() == 10.
    result.backward()
    assert current.grad[0, 0].abs().sum() == 0
    assert current.grad[0, 1].abs().sum() > 0


def test_empty_mask_rejected():
    import pytest
    with pytest.raises(RuntimeError, match='Empty RMU'):
        masked_mse(torch.ones(1, 2, 3), torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool))


def test_missing_rmu_modules_rejected():
    import pytest
    with pytest.raises(RuntimeError, match='three RMU'):
        attach_rmu(torch.nn.Linear(2, 2))
