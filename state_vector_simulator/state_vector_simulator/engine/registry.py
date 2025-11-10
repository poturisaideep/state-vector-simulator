"""Gate registry mapping names to handler implementations."""

from __future__ import annotations

import numpy as np

from ..circuit import Gate
from ..gates import (
    MULTI_QUBIT_HANDLERS,
    SINGLE_QUBIT_HANDLERS,
    TWO_QUBIT_HANDLERS,
)


def apply_gate(
    state: np.ndarray,
    gate: Gate,
    num_qubits: int,
    dtype: np.dtype,
) -> np.ndarray:
    name = gate.name
    if name in SINGLE_QUBIT_HANDLERS and not gate.controls:
        handler = SINGLE_QUBIT_HANDLERS[name]
        return handler(state, gate, num_qubits, dtype)
    if name in TWO_QUBIT_HANDLERS:
        handler = TWO_QUBIT_HANDLERS[name]
        return handler(state, gate, num_qubits, dtype)
    if name in MULTI_QUBIT_HANDLERS:
        handler = MULTI_QUBIT_HANDLERS[name]
        return handler(state, gate, num_qubits, dtype)
    raise ValueError(f"Unsupported gate: {gate.name}")

