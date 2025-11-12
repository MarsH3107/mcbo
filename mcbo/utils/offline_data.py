"""Utilities for working with pre-generated offline datasets.

This module provides helper classes and functions to split an offline dataset
of MCBO evaluations into an initial training set and a candidate pool. The
resulting dataset can subsequently be used by the optimisation loop to emulate
online interaction with an environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor


OfflineDataLike = Union[str, Dict[str, Tensor]]


@dataclass
class OfflineBatch:
    """Container returned when querying the candidate pool."""

    X: Tensor
    network_observation: Tensor
    observation: Tensor
    mean_objective: Tensor
    indices: Tensor


class OfflineDataset:
    """In-memory representation of an offline dataset.

    Parameters
    ----------
    X : Tensor
        Action vectors used to query the environment / function network.
    network_observation : Tensor
        Observations for each node in the causal network.
    observation : Tensor
        Objective values derived from ``network_observation``.
    mean_objective : Tensor
        Estimated noise-free objective values. When not provided the observed
        values are used instead.
    initial_count : int
        Number of samples that should form the initial training dataset.
    seed : Optional[int]
        Random seed controlling the shuffling prior to splitting the dataset.
    """

    def __init__(
        self,
        X: Tensor,
        network_observation: Tensor,
        observation: Tensor,
        mean_objective: Optional[Tensor],
        initial_count: int,
        seed: Optional[int] = None,
    ) -> None:
        if X.ndim != 2:
            raise ValueError("Expected X to have shape (n_samples, dim)")
        if network_observation.shape[0] != X.shape[0]:
            raise ValueError("network_observation must align with X")
        if observation.shape[0] != X.shape[0]:
            raise ValueError("observation must align with X")

        if mean_objective is None:
            mean_objective = observation.clone()
        elif mean_objective.shape[0] != X.shape[0]:
            raise ValueError("mean_objective must align with X")

        n_samples = X.shape[0]
        if initial_count <= 0 or initial_count >= n_samples:
            raise ValueError(
                "initial_count must be positive and smaller than the total number of samples"
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=X.device)
            generator.manual_seed(seed)

        permutation = torch.randperm(n_samples, generator=generator)
        init_idx = permutation[:initial_count]
        cand_idx = permutation[initial_count:]

        self.initial_X = X[init_idx]
        self.initial_network_observation = network_observation[init_idx]
        self.initial_observation = observation[init_idx]
        self.initial_mean = mean_objective[init_idx]

        self.candidate_X = X[cand_idx]
        self.candidate_network_observation = network_observation[cand_idx]
        self.candidate_observation = observation[cand_idx]
        self.candidate_mean = mean_objective[cand_idx]

        self._available_mask = torch.ones(
            self.candidate_X.shape[0], dtype=torch.bool, device=self.candidate_X.device
        )

    def get_initial_data(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return tensors corresponding to the initial training dataset."""

        return (
            self.initial_X,
            self.initial_network_observation,
            self.initial_observation,
            self.initial_mean,
        )

    def get_candidates(self, target: Optional[Tensor] = None) -> OfflineBatch:
        """Return the currently available candidate points.

        Parameters
        ----------
        target : Optional[Tensor]
            When provided, only candidates matching the (binary) intervention
            target are returned. The comparison is made using the first
            ``target.shape[-1]`` entries of ``X``. This mirrors the
            representation used by the interventional environments.
        """

        available_indices = torch.where(self._available_mask)[0]
        X = self.candidate_X[available_indices]
        network_observation = self.candidate_network_observation[available_indices]
        observation = self.candidate_observation[available_indices]
        mean_objective = self.candidate_mean[available_indices]

        if target is not None and target.numel() > 0:
            target = target.to(X)
            target_dim = target.shape[-1]
            if target_dim > X.shape[-1]:
                raise ValueError("target dimensionality exceeds candidate dimensionality")
            candidate_targets = X[..., :target_dim]
            match_mask = torch.all(torch.isclose(candidate_targets, target), dim=-1)
            available_indices = available_indices[match_mask]
            X = X[match_mask]
            network_observation = network_observation[match_mask]
            observation = observation[match_mask]
            mean_objective = mean_objective[match_mask]

        return OfflineBatch(
            X=X,
            network_observation=network_observation,
            observation=observation,
            mean_objective=mean_objective,
            indices=available_indices,
        )

    def pop(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Remove and return a candidate by its absolute index."""

        if index < 0 or index >= self._available_mask.shape[0]:
            raise IndexError("Candidate index out of range")
        if not self._available_mask[index]:
            raise ValueError("Candidate at the provided index has already been used")

        self._available_mask[index] = False

        return (
            self.candidate_X[index].unsqueeze(0),
            self.candidate_network_observation[index].unsqueeze(0),
            self.candidate_observation[index].unsqueeze(0),
            self.candidate_mean[index].unsqueeze(0),
        )

    def is_empty(self) -> bool:
        """Return ``True`` if no candidates are left."""

        return not torch.any(self._available_mask)


def _to_tensor(value: Union[Tensor, "np.ndarray"]) -> Tensor:
    if isinstance(value, Tensor):
        return value
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise TypeError("NumPy support is required to load numpy arrays") from exc

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    raise TypeError("Unsupported data type for offline dataset")


def load_offline_dataset(
    data: OfflineDataLike,
    initial_count: int,
    seed: Optional[int] = None,
) -> OfflineDataset:
    """Load and split an offline dataset.

    ``data`` can either be a mapping already containing the tensors or a path to
    a Torch serialised object. The mapping is expected to provide at least the
    ``X``, ``network_observation`` and ``observation`` entries. An optional
    ``mean_objective`` entry can be supplied to distinguish the noise-free
    objective from the noisy observations.
    """

    if isinstance(data, str):
        payload = torch.load(data)
    elif isinstance(data, dict):
        payload = data
    else:
        raise TypeError("Unsupported offline data specification")

    try:
        X = _to_tensor(payload["X"])
        network_observation = _to_tensor(payload["network_observation"])
        observation = _to_tensor(payload["observation"])
    except KeyError as exc:
        raise KeyError(
            "Offline dataset must contain 'X', 'network_observation' and 'observation'"
        ) from exc

    mean_objective = payload.get("mean_objective")
    if mean_objective is not None:
        mean_objective = _to_tensor(mean_objective)

    return OfflineDataset(
        X=X,
        network_observation=network_observation,
        observation=observation,
        mean_objective=mean_objective,
        initial_count=initial_count,
        seed=seed,
    )

