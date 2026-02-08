"""Gate registries and default gate sets."""

from __future__ import annotations

from .single_qubit import GATE_NAMES as SINGLE_QUBIT_NAMES
from .single_qubit import HANDLERS as SINGLE_QUBIT_HANDLERS
from .two_qubit import GATE_NAMES as TWO_QUBIT_NAMES
from .two_qubit import HANDLERS as TWO_QUBIT_HANDLERS
from .multi_qubit import GATE_NAMES as MULTI_QUBIT_NAMES
from .multi_qubit import HANDLERS as MULTI_QUBIT_HANDLERS

DEFAULT_SINGLE_QUBIT_GATES = ["h", "rx", "ry", "rz", "s", "t"]
DEFAULT_TWO_QUBIT_GATES = ["cx", "cz", "swap"]
DEFAULT_MULTI_QUBIT_GATES = []

__all__ = [
    "SINGLE_QUBIT_HANDLERS",
    "TWO_QUBIT_HANDLERS",
    "MULTI_QUBIT_HANDLERS",
    "DEFAULT_SINGLE_QUBIT_GATES",
    "DEFAULT_TWO_QUBIT_GATES",
    "DEFAULT_MULTI_QUBIT_GATES",
    "SINGLE_QUBIT_NAMES",
    "TWO_QUBIT_NAMES",
    "MULTI_QUBIT_NAMES",
]

