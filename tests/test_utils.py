"""Unit tests for utility helpers that rely on optional dependencies."""
from __future__ import annotations

import argparse

import pytest


torch = pytest.importorskip("torch", reason="PyTorch is required for utility tests")
functions_utils = pytest.importorskip(
    "mcbo.utils.functions_utils", reason="functions_utils requires PyTorch"
)
runner_utils = pytest.importorskip("mcbo.utils.runner_utils")


def test_noise_scales_to_normals():
    from torch.distributions.normal import Normal

    # test scalar noise_scales
    distribution_list = functions_utils.noise_scales_to_normals(2.0, 2)
    for distribution in distribution_list:
        assert distribution.variance == 2.0**2

    # test a torch tensor of noise scales
    distribution_list = functions_utils.noise_scales_to_normals(torch.ones((2,)), 2)
    for distribution in distribution_list:
        assert distribution.variance == 1.0

    # test zero
    distribution_list = functions_utils.noise_scales_to_normals(0.0, 2)
    for distribution in distribution_list:
        assert distribution.variance == pytest.approx(1e-6)

    # ensure non-zero noise uses Normal distributions
    distribution_list = functions_utils.noise_scales_to_normals(torch.tensor([0.2, 0.3]), 2)
    assert all(isinstance(distribution, Normal) for distribution in distribution_list)


def test_check_nonnegative():
    with pytest.raises(argparse.ArgumentTypeError):
        runner_utils.check_nonnegative(-2)
    with pytest.raises(argparse.ArgumentTypeError):
        runner_utils.check_nonnegative("hello world")
    assert runner_utils.check_nonnegative(0.0) == 0.0
    assert runner_utils.check_nonnegative(2) == 2.0
