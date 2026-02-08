"""High-level library API to run the state vector simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .circuit import QuantumCircuit
from .circuit_builders import random_circuit
from .simulator import SimulationResult, StateVectorSimulator
from .xeb import LinearXEBResult, run_linear_xeb_experiment


def load_circuit(path: Union[Path, str]) -> QuantumCircuit:
    """Load a circuit from a JSON file.

    Args:
        path: Path to a JSON file with keys "num_qubits" and "gates".

    Returns:
        QuantumCircuit ready for run() or run_xeb().
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return QuantumCircuit.from_dict(payload)


def run(
    circuit: Union[QuantumCircuit, Path, str],
    shots: int = 0,
    seed: int | None = None,
    simulator: StateVectorSimulator | None = None,
) -> SimulationResult:
    """Run a circuit and return the simulation result.

    Args:
        circuit: A QuantumCircuit, or a path to a JSON circuit file.
        shots: Number of measurement shots to sample (0 = state vector only, no sampling).
        seed: Random seed for reproducible sampling.
        simulator: Optional simulator instance; a new one is created if not provided.

    Returns:
        SimulationResult with final_state, probabilities, and (if shots > 0) samples and counts.
    """
    if not isinstance(circuit, QuantumCircuit):
        circuit = load_circuit(circuit)
    sim = simulator or StateVectorSimulator()
    return sim.run(circuit, shots=shots, seed=seed)


def run_xeb(
    circuit: Union[QuantumCircuit, Path, str],
    shots: int,
    seed: int | None = None,
    simulator: StateVectorSimulator | None = None,
) -> LinearXEBResult:
    """Run a linear XEB experiment: simulate the circuit, sample measurements, compute fidelity.

    Args:
        circuit: A QuantumCircuit, or a path to a JSON circuit file.
        shots: Number of measurement shots (must be > 0).
        seed: Random seed for reproducibility.
        simulator: Optional simulator instance.

    Returns:
        LinearXEBResult with fidelity, samples, and sample_probabilities.
    """
    if not isinstance(circuit, QuantumCircuit):
        circuit = load_circuit(circuit)
    return run_linear_xeb_experiment(
        circuit,
        simulator=simulator,
        shots=shots,
        seed=seed,
    )


__all__ = [
    "load_circuit",
    "run",
    "run_xeb",
    "random_circuit",
]
