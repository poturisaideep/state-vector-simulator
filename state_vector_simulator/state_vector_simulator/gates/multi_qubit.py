"""Multi-qubit (three or more qubits) gate implementations and registry."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .. import linalg
from ..circuit import Gate

MultiQubitHandler = Callable[[np.ndarray, Gate, int, np.dtype], np.ndarray]


def apply_ccx(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.controls) != 2 or len(gate.targets) != 1:
        raise ValueError("ccx gate requires two controls and one target")
    controls = list(gate.controls)
    target = gate.targets[0]
    matrix = np.eye(8, dtype=dtype)
    matrix[6, 6] = 0
    matrix[7, 7] = 0
    matrix[6, 7] = 1
    matrix[7, 6] = 1
    return linalg.apply_unitary(state, [*controls, target], matrix, num_qubits)


def apply_ccz(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.controls) != 2 or len(gate.targets) != 1:
        raise ValueError("ccz gate requires two controls and one target")
    controls = list(gate.controls)
    target = gate.targets[0]
    matrix = np.eye(8, dtype=dtype)
    matrix[7, 7] = -1
    return linalg.apply_unitary(state, [*controls, target], matrix, num_qubits)


def apply_cswap(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.controls) != 1 or len(gate.targets) != 2:
        raise ValueError("cswap gate requires one control and two targets")
    control = gate.controls[0]
    q1, q2 = gate.targets
    matrix = np.eye(8, dtype=dtype)
    matrix[5, 5] = 0
    matrix[6, 6] = 0
    matrix[5, 6] = 1
    matrix[6, 5] = 1
    return linalg.apply_unitary(state, [control, q1, q2], matrix, num_qubits)


HANDLERS: Dict[str, MultiQubitHandler] = {
    "ccx": apply_ccx,
    "ccz": apply_ccz,
    "cswap": apply_cswap,
}

GATE_NAMES: List[str] = list(HANDLERS.keys())

__all__ = ["HANDLERS", "GATE_NAMES", "MultiQubitHandler"]

