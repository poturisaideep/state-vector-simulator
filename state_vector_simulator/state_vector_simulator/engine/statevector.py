"""NumPy-based state vector simulator."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .. import linalg
from ..circuit import QuantumCircuit
from . import measurements, registry
from .result import SimulationResult


class StateVectorSimulator:
    """State vector simulator delegating gate logic to handler modules."""

    def __init__(self, *, dtype: np.dtype = np.complex128) -> None:
        self.dtype = dtype

    def run(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> SimulationResult:
        state = linalg.initial_state(circuit.num_qubits, dtype=self.dtype)
        for gate in circuit.gates:
            state = registry.apply_gate(state, gate, circuit.num_qubits, self.dtype)

        probabilities = np.abs(state) ** 2
        result_shots = shots if shots and shots > 0 else None
        samples: Optional[List[str]] = None
        counts: Optional[Dict[str, int]] = None
        if result_shots:
            rng = np.random.default_rng(seed)
            samples = measurements.sample_measurements(probabilities, circuit.num_qubits, result_shots, rng)
            counts = measurements.counts_from_samples(samples)

        return SimulationResult(
            circuit=circuit,
            final_state=state,
            probabilities=probabilities,
            shots=result_shots,
            samples=samples,
            counts=counts,
        )

