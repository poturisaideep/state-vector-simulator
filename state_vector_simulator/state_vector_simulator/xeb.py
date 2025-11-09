"""Cross-entropy benchmarking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .circuit import QuantumCircuit
from .simulator import StateVectorSimulator


@dataclass
class LinearXEBResult:
    """Summary of a linear XEB experiment."""

    circuit: QuantumCircuit
    fidelity: float
    shots: int
    samples: List[str]
    ideal_probabilities: List[float]
    sample_probabilities: Dict[str, float]


def compute_linear_xeb_fidelity(
    ideal_probabilities: Iterable[float],
    samples: Iterable[str],
) -> float:
    samples = list(samples)
    if not samples:
        raise ValueError("XEB fidelity requires at least one sample")

    num_qubits = len(samples[0])
    dimension = 1 << num_qubits

    probs = list(ideal_probabilities)
    if len(probs) != dimension:
        raise ValueError("Ideal probability vector size does not match bitstring width")

    prob_lookup = {format(index, f"0{num_qubits}b"): float(p) for index, p in enumerate(probs)}
    average_p = sum(prob_lookup[bitstring] for bitstring in samples) / len(samples)
    fidelity = dimension * average_p - 1.0
    return fidelity


def run_linear_xeb_experiment(
    circuit: QuantumCircuit,
    simulator: Optional[StateVectorSimulator] = None,
    *,
    shots: int,
    seed: int | None = None,
) -> LinearXEBResult:
    if shots <= 0:
        raise ValueError("XEB experiment requires shots > 0")
    simulator = simulator or StateVectorSimulator()
    result = simulator.run(circuit, shots=shots, seed=seed)
    if result.samples is None:
        raise RuntimeError("Simulator did not return measurement samples")
    fidelity = compute_linear_xeb_fidelity(result.probabilities, result.samples)
    sample_probabilities = _relative_frequencies(result.samples)
    return LinearXEBResult(
        circuit=circuit,
        fidelity=fidelity,
        shots=shots,
        samples=result.samples,
        ideal_probabilities=result.probabilities,
        sample_probabilities=sample_probabilities,
    )


def _relative_frequencies(samples: Iterable[str]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total = 0
    for bitstring in samples:
        counts[bitstring] = counts.get(bitstring, 0) + 1
        total += 1
    return {bitstring: count / total for bitstring, count in counts.items()}

