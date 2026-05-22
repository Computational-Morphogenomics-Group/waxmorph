import importlib

import pytest
import torch

torch_train_module = importlib.import_module("waxmorph.torch.train")


def test_stable_clip_handles_large_finite_float32_gradients():
    param = torch.nn.Parameter(torch.zeros(1024, dtype=torch.float32))
    param.grad = torch.full_like(param, 1e20)

    grad_norm = torch_train_module._clip_grad_norm_stable([("param", param)], max_norm=1.0, epoch=0)

    assert torch.isfinite(grad_norm)
    assert grad_norm.item() > 1e20
    assert torch.isfinite(param.grad).all()
    assert torch.linalg.vector_norm(param.grad.double()).item() == pytest.approx(1.0)
