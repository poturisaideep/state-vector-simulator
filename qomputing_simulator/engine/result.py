"""Simulation result dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..circuit import QuantumCircuit


@dataclass
class SimulationResult:
    """Results of a simulation run."""

    circuit: QuantumCircuit
    final_state: np.ndarray
    probabilities: np.ndarray
    shots: Optional[int] = None
    samples: Optional[List[str]] = None
    counts: Optional[Dict[str, int]] = None

    def bitstring_probabilities(self) -> Dict[str, float]:
        """Return probabilities keyed by little-endian bitstrings."""
        probs: Dict[str, float] = {}
        for index, probability in enumerate(self.probabilities):
            bitstring = format(index, f"0{self.circuit.num_qubits}b")
            probs[bitstring] = float(probability)
        return probs

