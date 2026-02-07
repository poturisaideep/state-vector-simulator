"""Measurement sampling utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def sample_measurements(
    probabilities: np.ndarray,
    num_qubits: int,
    shots: int,
    rng: np.random.Generator,
) -> List[str]:
    probabilities = probabilities / probabilities.sum()
    basis_indices = rng.choice(len(probabilities), size=shots, p=probabilities)
    return [format(index, f"0{num_qubits}b") for index in basis_indices]


def counts_from_samples(samples: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for measurement in samples:
        counts[measurement] = counts.get(measurement, 0) + 1
    return counts

