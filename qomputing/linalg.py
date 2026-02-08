"""Shared tensor and linear algebra helpers."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def initial_state(num_qubits: int, *, dtype: np.dtype) -> np.ndarray:
    size = 1 << num_qubits
    state = np.zeros(size, dtype=dtype)
    state[0] = 1.0 + 0j
    return state


def axis_for_qubit(qubit: int) -> int:
    """Return tensor axis index for the given little-endian qubit."""
    return -(qubit + 1)


def reshape_state(state: np.ndarray, num_qubits: int) -> np.ndarray:
    return state.reshape((2,) * num_qubits)


def swap_axes(tensor: np.ndarray, axis_a: int, axis_b: int) -> np.ndarray:
    return np.swapaxes(tensor, axis_a, axis_b)


def select_indices(num_qubits: int, indices: Dict[int, int]) -> Tuple[slice, ...]:
    slices = [slice(None)] * num_qubits
    for axis, value in indices.items():
        slices[axis] = value
    return tuple(slices)


def apply_unitary(
    state: np.ndarray,
    qubits: List[int],
    matrix: np.ndarray,
    num_qubits: int,
) -> np.ndarray:
    qubits = list(qubits)
    if not qubits:
        return state
    matrix = np.asarray(matrix, dtype=state.dtype)
    dimension = 1 << len(qubits)
    if matrix.shape != (dimension, dimension):
        raise ValueError("Unitary size does not match qubit count")

    tensor = reshape_state(state, num_qubits)
    axes = [axis_for_qubit(q) for q in qubits]
    dest_axes = list(range(-len(qubits), 0))
    tensor = np.moveaxis(tensor, axes, dest_axes)

    leading_shape = tensor.shape[:-len(qubits)]
    reshaped = tensor.reshape(-1, dimension).T
    transformed = matrix @ reshaped
    tensor = transformed.T.reshape(*leading_shape, *([2] * len(qubits)))
    tensor = np.moveaxis(tensor, dest_axes, axes)
    return tensor.reshape(state.shape)

