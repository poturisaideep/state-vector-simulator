"""State vector simulation backend implemented with pure Python numerics."""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .circuit import Gate, QuantumCircuit


@dataclass
class SimulationResult:
    """Results of a simulation run."""

    circuit: QuantumCircuit
    final_state: List[complex]
    probabilities: List[float]
    shots: Optional[int] = None
    samples: Optional[List[str]] = None
    counts: Optional[Dict[str, int]] = None

    def bitstring_probabilities(self) -> Dict[str, float]:
        """Return probabilities keyed by little-endian bitstrings."""
        probs: Dict[str, float] = {}
        for index, probability in enumerate(self.probabilities):
            bitstring = format(index, f"0{self.circuit.num_qubits}b")
            probs[bitstring] = probability
        return probs


class StateVectorSimulator:
    """Simple state vector simulator supporting a small gate set."""

    def run(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> SimulationResult:
        state = self._initial_state(circuit.num_qubits)
        for gate in circuit.gates:
            state = self._apply_gate(state, gate, circuit.num_qubits)

        probabilities = [abs(amplitude) ** 2 for amplitude in state]
        result_shots = shots if shots and shots > 0 else None
        samples: Optional[List[str]] = None
        counts: Optional[Dict[str, int]] = None
        if result_shots:
            rng = random.Random(seed)
            samples = self._sample_measurements(probabilities, circuit.num_qubits, result_shots, rng)
            counts = self._counts_from_samples(samples)

        return SimulationResult(
            circuit=circuit,
            final_state=state,
            probabilities=probabilities,
            shots=result_shots,
            samples=samples,
            counts=counts,
        )

    # -------------------------------------------------------------- Internals
    def _initial_state(self, num_qubits: int) -> List[complex]:
        size = 1 << num_qubits
        state = [0j] * size
        state[0] = 1.0 + 0j
        return state

    def _apply_gate(self, state: List[complex], gate: Gate, num_qubits: int) -> List[complex]:
        name = gate.name
        if name in {"x", "y", "z", "h", "s", "t"}:
            matrix = _named_single_qubit_matrix(name)
            return _apply_single_qubit(state, matrix, gate.targets[0])
        if name in {"rx", "ry", "rz"}:
            theta = float(gate.params.get("theta", 0.0))
            matrix = _rotation_matrix(name, theta)
            return _apply_single_qubit(state, matrix, gate.targets[0])
        if name == "cx":
            if len(gate.controls) != 1 or len(gate.targets) != 1:
                raise ValueError("cx gate requires one control and one target")
            control = gate.controls[0]
            target = gate.targets[0]
            return _apply_cnot(state, control, target)
        raise ValueError(f"Unsupported gate: {gate.name}")

    def _sample_measurements(
        self,
        probabilities: List[float],
        num_qubits: int,
        shots: int,
        rng: random.Random,
    ) -> List[str]:
        outcomes = [format(index, f"0{num_qubits}b") for index in range(len(probabilities))]
        samples = rng.choices(outcomes, weights=probabilities, k=shots)
        return samples

    def _counts_from_samples(self, samples: Iterable[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for measurement in samples:
            counts[measurement] = counts.get(measurement, 0) + 1
        return counts


# ---------------------------------------------------------------- Gate helpers
Matrix2x2 = Tuple[Tuple[complex, complex], Tuple[complex, complex]]


def _named_single_qubit_matrix(name: str) -> Matrix2x2:
    if name == "x":
        return ((0.0 + 0j, 1.0 + 0j), (1.0 + 0j, 0.0 + 0j))
    if name == "y":
        return ((0.0 + 0j, -1j), (1j, 0.0 + 0j))
    if name == "z":
        return ((1.0 + 0j, 0.0 + 0j), (0.0 + 0j, -1.0 + 0j))
    if name == "h":
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        return (
            (inv_sqrt2 + 0j, inv_sqrt2 + 0j),
            (inv_sqrt2 + 0j, (-inv_sqrt2) + 0j),
        )
    if name == "s":
        return ((1.0 + 0j, 0.0 + 0j), (0.0 + 0j, 0.0 + 1j))
    if name == "t":
        phase = cmath.exp(1j * math.pi / 4.0)
        return ((1.0 + 0j, 0.0 + 0j), (0.0 + 0j, phase))
    raise ValueError(f"Unknown single-qubit gate: {name}")


def _rotation_matrix(name: str, theta: float) -> Matrix2x2:
    half_theta = theta / 2.0
    if name == "rx":
        cos = math.cos(half_theta)
        sin = math.sin(half_theta)
        return (
            (cos + 0j, -1j * sin),
            (-1j * sin, cos + 0j),
        )
    if name == "ry":
        cos = math.cos(half_theta)
        sin = math.sin(half_theta)
        return (
            (cos + 0j, -sin + 0j),
            (sin + 0j, cos + 0j),
        )
    if name == "rz":
        phase_pos = cmath.exp(1j * half_theta)
        phase_neg = cmath.exp(-1j * half_theta)
        return (
            (phase_neg, 0.0 + 0j),
            (0.0 + 0j, phase_pos),
        )
    raise ValueError(f"Unknown rotation gate: {name}")


def _apply_single_qubit(state: List[complex], matrix: Matrix2x2, target: int) -> List[complex]:
    size = len(state)
    stride = 1 << target
    period = stride << 1
    new_state = state[:]
    for start in range(0, size, period):
        for offset in range(stride):
            idx0 = start + offset
            idx1 = idx0 + stride
            a0 = state[idx0]
            a1 = state[idx1]
            new_state[idx0] = matrix[0][0] * a0 + matrix[0][1] * a1
            new_state[idx1] = matrix[1][0] * a0 + matrix[1][1] * a1
    return new_state


def _apply_cnot(state: List[complex], control: int, target: int) -> List[complex]:
    if control == target:
        raise ValueError("Control and target qubit must differ for cx gate")
    size = len(state)
    control_mask = 1 << control
    target_mask = 1 << target
    new_state = state[:]
    for index in range(size):
        if index & control_mask:
            flipped = index ^ target_mask
            if index < flipped:
                new_state[index], new_state[flipped] = state[flipped], state[index]
            else:
                new_state[index] = state[flipped]
    return new_state

