"""Regression tests for functions defined in ``scripts.functions``."""
from __future__ import annotations

import pytest

scripts_functions = pytest.importorskip(
    "scripts.functions", reason="scripts package is required for function tests"
)
torch = pytest.importorskip("torch", reason="PyTorch is required for function tests")

Env = scripts_functions.Env
Dropwave = scripts_functions.Dropwave
Alpine2 = scripts_functions.Alpine2
Ackley = scripts_functions.Ackley
Rosenbrock = scripts_functions.Rosenbrock
ToyGraph = scripts_functions.ToyGraph
PSAGraph = scripts_functions.PSAGraph


def test_function():
    env = Env()
    x = torch.tensor([0.5, 0.5])
    with pytest.raises(NotImplementedError):
        env.evaluate(x)


def test_dropwave():
    function = Dropwave(noise_scales=0.0)
    x = torch.tensor([0.5, 0.5])
    assert abs(function.evaluate(x)[-1] - 1.0) < 0.0001


def test_alpine2():
    function = Alpine2(noise_scales=0.0)
    x = torch.zeros(6)
    assert abs(function.evaluate(x)[-1] - 0.0) < 0.0001


def test_ackley():
    function = Ackley(noise_scales=0.0)
    # test that putting 0.5 for all actions gives output 0
    x = torch.full((6,), 0.5)
    assert abs(function.evaluate(x)[-1] - 0.0) < 0.0001
    with pytest.raises(ValueError):
        function.evaluate(torch.tensor([0.5, 0.5]))


def test_rosenbrock():
    function = Rosenbrock(noise_scales=0.0)
    x = torch.full((5,), 0.75)
    assert abs(function.evaluate(x)[-1] - 0.0) < 0.0001


def test_toy():
    function = ToyGraph(noise_scales=0.0)
    # test that the max found in Aglietti has the right output value
    x = torch.tensor([0.0, 1.0, 0.0, 0.0, (-3.16053512 + 5) / 25, 0.0])
    assert abs(function.evaluate(x)[-1] - 2.1710) < 0.01
    x2 = torch.tensor([0.0, 1.0, 0.0, 0.9109, 0.3227, 0.0])
    assert abs(function.evaluate(x2)[-1] - 1.85562) < 0.01


def test_psa():
    function = PSAGraph()
    x = torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0])
    mean = 0.0
    n = 1000
    for _ in range(n):
        mean += function.evaluate(x)[-1] / n
    assert abs(mean - (-5.151712726)) < 0.05
