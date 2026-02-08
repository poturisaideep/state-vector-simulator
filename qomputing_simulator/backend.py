"""Backend API: QomputingSimulator.get_backend(), backend.run(), result.get_counts()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from .engine.statevector import StateVectorSimulator
from .engine.result import SimulationResult

if TYPE_CHECKING:
    from .circuit import QuantumCircuit


class Result:
    """Result of a backend run. Use get_counts() for measurement counts."""

    def __init__(self, sim_result: SimulationResult) -> None:
        self._result = sim_result

    def get_counts(self) -> Dict[str, int]:
        """Return measurement counts (bitstring -> count)."""
        if self._result.counts is None:
            return {}
        return dict(self._result.counts)

    def get_statevector(self) -> np.ndarray:
        """Return the final state vector."""
        return self._result.final_state.copy()

    def get_probabilities(self) -> np.ndarray:
        """Return the measurement probabilities."""
        return self._result.probabilities.copy()


class Backend:
    """State-vector backend. Create via QomputingSimulator.get_backend(\"state_vector\")."""

    def __init__(self) -> None:
        self._sim = StateVectorSimulator()

    def run(
        self,
        qc: "QuantumCircuit",
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> Result:
        """Run the circuit and return a Result with get_counts(), get_statevector(), etc.
        Use shots=0 for state-vector only (no sampling); get_counts() will then return {}.
        """
        sim_result = self._sim.run(qc, shots=shots if shots and shots > 0 else None, seed=seed)
        return Result(sim_result)


class QomputingSimulator:
    """Library name: Qomputing. Use get_backend() to get a backend, then backend.run(qc, shots=...)."""

    @staticmethod
    def get_backend(name: str = "state_vector") -> Backend:
        """Return a backend. Currently only \"state_vector\" is supported."""
        if name != "state_vector":
            raise ValueError(f"Unknown backend: {name!r}. Use 'state_vector'.")
        return Backend()
