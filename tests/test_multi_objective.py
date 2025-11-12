import torch

from mcbo.utils.multi_objective import compute_hypervolume, compute_pareto_front


def test_compute_pareto_front_returns_mask_and_front():
    torch.manual_seed(0)
    observations = torch.rand(32, 3)
    front, mask = compute_pareto_front(observations)

    assert front.ndim == 2
    assert mask.shape == torch.Size([observations.shape[0]])
    assert mask.dtype == torch.bool
    assert front.shape[1] == observations.shape[1]


def test_compute_hypervolume_is_stable_for_large_samples():
    torch.manual_seed(0)
    observations = torch.rand(3000, 3, dtype=torch.double)
    ref_point = torch.zeros(3, dtype=torch.double)
    hv = compute_hypervolume(observations, ref_point)

    assert isinstance(hv, float)
    assert hv >= 0.0
