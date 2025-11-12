"""Utility helpers for multi-objective Bayesian optimization."""

from typing import Optional, Tuple
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor


def compute_pareto_front(Y: Tensor) -> Tuple[Tensor, Tensor]:
    """Return the Pareto front and mask for the provided outcomes."""

    if Y.ndim != 2:
        raise ValueError("Expected a 2D tensor of observations.")
    mask = is_non_dominated(Y)
    return Y[mask], mask


def compute_hypervolume(
    Y: Tensor, ref_point: Tensor, pareto_front: Optional[Tensor] = None
) -> float:
    """Compute the dominated hypervolume for outcomes ``Y``."""

    if ref_point.shape[-1] != Y.shape[-1]:
        raise ValueError("Reference point dimension must match outcome dimension.")
    if pareto_front is None:
        pareto_front, _ = compute_pareto_front(Y)
    hv = Hypervolume(ref_point=ref_point.to(Y))
    return hv.compute(pareto_front).item()
