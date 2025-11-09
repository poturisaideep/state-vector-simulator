"""Utility for comparing the state vector simulator against Cirq.

This script generates random quantum circuits, executes them with both
the in-tree state vector simulator and Cirq's reference simulator, and
reports discrepancies in amplitudes, probabilities, and XEB fidelity.

Usage (after installing Cirq):

    python tools/cirq_comparison.py --max-qubits 3 --depths 3 5
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import cirq
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "Cirq is required for this comparison script. Install with `pip install cirq`."
    ) from exc

from state_vector_simulator import (
    QuantumCircuit,
    StateVectorSimulator,
    compute_linear_xeb_fidelity,
)


DEFAULT_SINGLE_QUBIT_GATES = ("h", "rx", "ry", "rz", "s", "t")
GLOBAL_PHASE_TOLERANCE = 1e-12


@dataclass
class ComparisonMetrics:
    num_qubits: int
    depth: int
    circuit_index: int
    amplitude_error: float
    probability_error: float
    xeb_error: float | None


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    simulator = StateVectorSimulator()

    metrics: List[ComparisonMetrics] = []

    for num_qubits in range(args.min_qubits, args.max_qubits + 1):
        for depth in args.depths:
            for circuit_index in range(args.circuits_per_config):
                circuit_seed = rng.randint(0, 2**31 - 1)
                circuit_rng = random.Random(circuit_seed)
                circuit = _random_circuit(
                    num_qubits=num_qubits,
                    depth=depth,
                    single_qubit_gates=args.single_qubit_gates,
                    rng=circuit_rng,
                )
                metric = _compare_with_cirq(
                    circuit,
                    simulator=simulator,
                    shots=args.shots,
                    seed=circuit_seed,
                    depth=depth,
                    circuit_index=circuit_index,
                )
                metrics.append(metric)
                _print_metric(metric)

    summary = _compute_summary(metrics)
    _print_summary(summary)

    max_amp_err = summary["max_amplitude_error"]
    max_prob_err = summary["max_probability_error"]
    amp_ok = max_amp_err <= args.amplitude_tolerance
    prob_ok = max_prob_err <= args.probability_tolerance
    xeb_ok = True
    if args.shots > 0:
        max_xeb_err = summary["max_xeb_error"]
        xeb_ok = max_xeb_err <= args.xeb_tolerance

    return 0 if amp_ok and prob_ok and xeb_ok else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare simulator outputs with Cirq")
    parser.add_argument("--min-qubits", type=int, default=1, help="Minimum number of qubits to test")
    parser.add_argument("--max-qubits", type=int, default=3, help="Maximum number of qubits to test")
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[3],
        help="List of circuit depths to evaluate",
    )
    parser.add_argument(
        "--circuits-per-config",
        type=int,
        default=3,
        help="Number of random circuits per (qubits, depth) combo",
    )
    parser.add_argument(
        "--single-qubit-gates",
        type=str,
        nargs="*",
        default=DEFAULT_SINGLE_QUBIT_GATES,
        help="Pool of single-qubit gates to sample from",
    )
    parser.add_argument("--shots", type=int, default=512, help="Number of samples for XEB comparison")
    parser.add_argument("--seed", type=int, default=1234, help="Master random seed")
    parser.add_argument(
        "--amplitude-tolerance",
        type=float,
        default=1e-6,
        help="Maximum tolerated amplitude sup-norm difference",
    )
    parser.add_argument(
        "--probability-tolerance",
        type=float,
        default=1e-6,
        help="Maximum tolerated probability sup-norm difference",
    )
    parser.add_argument(
        "--xeb-tolerance",
        type=float,
        default=1e-3,
        help="Maximum tolerated linear XEB fidelity difference",
    )
    return parser


def _random_circuit(
    *,
    num_qubits: int,
    depth: int,
    single_qubit_gates: Iterable[str],
    rng: random.Random,
) -> QuantumCircuit:
    if num_qubits <= 0:
        raise ValueError("Number of qubits must be positive")
    if depth <= 0:
        raise ValueError("Depth must be positive")
    gate_pool = list(single_qubit_gates)
    if not gate_pool:
        raise ValueError("Single-qubit gate set must not be empty")

    circuit = QuantumCircuit(num_qubits)
    for _layer in range(depth):
        for qubit in range(num_qubits):
            gate_name = rng.choice(gate_pool)
            if gate_name in {"rx", "ry", "rz"}:
                theta = rng.uniform(0.0, 2.0 * math.pi)
                circuit.add_gate(gate_name, [qubit], params={"theta": theta})
            else:
                circuit.add_gate(gate_name, [qubit])

        if num_qubits > 1:
            control, target = rng.sample(range(num_qubits), 2)
            circuit.cx(control, target)

    return circuit


def _compare_with_cirq(
    circuit: QuantumCircuit,
    *,
    simulator: StateVectorSimulator,
    shots: int,
    seed: int,
    depth: int,
    circuit_index: int,
) -> ComparisonMetrics:
    ours = simulator.run(circuit, shots=shots, seed=seed)
    cirq_circuit, qubits = _circuit_to_cirq(circuit)
    cirq_sim = cirq.Simulator()
    cirq_result = cirq_sim.simulate(cirq_circuit)
    cirq_state = _to_little_endian_state(cirq_result.final_state_vector, circuit.num_qubits)

    aligned_cirq_state = _align_global_phase(ours.final_state, cirq_state)
    amplitude_error = _max_difference(ours.final_state, aligned_cirq_state)

    cirq_probabilities = [abs(amplitude) ** 2 for amplitude in aligned_cirq_state]
    probability_error = _max_difference(ours.probabilities, cirq_probabilities)

    xeb_error = None
    if shots > 0:
        sample_rng = random.Random(seed)
        samples = _sample_bitstrings(sample_rng, cirq_probabilities, circuit.num_qubits, shots)
        xeb_ours = compute_linear_xeb_fidelity(ours.probabilities, samples)
        xeb_cirq = compute_linear_xeb_fidelity(cirq_probabilities, samples)
        xeb_error = abs(xeb_ours - xeb_cirq)

    return ComparisonMetrics(
        num_qubits=circuit.num_qubits,
        depth=depth,
        circuit_index=circuit_index,
        amplitude_error=amplitude_error,
        probability_error=probability_error,
        xeb_error=xeb_error,
    )


def _circuit_to_cirq(circuit: QuantumCircuit) -> Tuple[cirq.Circuit, Tuple[cirq.Qid, ...]]:
    qubits = tuple(cirq.LineQubit.range(circuit.num_qubits))
    operations: List[cirq.Operation] = []
    for gate in circuit.gates:
        if len(gate.targets) == 1 and not gate.controls:
            target = qubits[gate.targets[0]]
            operations.append(_single_qubit_operation(gate.name, target, gate.params))
        elif gate.name == "cx" and len(gate.controls) == 1 and len(gate.targets) == 1:
            control = qubits[gate.controls[0]]
            target = qubits[gate.targets[0]]
            operations.append(cirq.CNOT(control, target))
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported gate for Cirq conversion: {gate}")
    return cirq.Circuit(operations), qubits


def _single_qubit_operation(name: str, qubit: cirq.Qid, params: Dict[str, float]) -> cirq.Operation:
    if name == "x":
        return cirq.X(qubit)
    if name == "y":
        return cirq.Y(qubit)
    if name == "z":
        return cirq.Z(qubit)
    if name == "h":
        return cirq.H(qubit)
    if name == "s":
        return cirq.S(qubit)
    if name == "t":
        return cirq.T(qubit)
    if name in {"rx", "ry", "rz"}:
        theta = float(params.get("theta", 0.0))
        if name == "rx":
            return cirq.rx(theta)(qubit)
        if name == "ry":
            return cirq.ry(theta)(qubit)
        return cirq.rz(theta)(qubit)
    raise ValueError(f"Unsupported single-qubit gate: {name}")


def _align_global_phase(
    reference: Sequence[complex],
    target: Sequence[complex],
) -> List[complex]:
    phase = 1 + 0j
    for ref, tgt in zip(reference, target):
        if abs(tgt) > GLOBAL_PHASE_TOLERANCE:
            phase = ref / tgt
            break
    return [tgt * phase for tgt in target]


def _max_difference(
    reference: Sequence[float | complex],
    candidate: Sequence[float | complex],
) -> float:
    return max(abs(r - c) for r, c in zip(reference, candidate))


def _to_little_endian_state(state: Sequence[complex], num_qubits: int) -> List[complex]:
    """Reorder Cirq's big-endian state vector into little-endian ordering."""
    size = 1 << num_qubits
    if len(state) != size:
        raise ValueError("State vector size does not match qubit count")
    reordered = [0j] * size
    for index, amplitude in enumerate(state):
        little_index = _reverse_bits(index, num_qubits)
        reordered[little_index] = amplitude
    return reordered


def _reverse_bits(value: int, width: int) -> int:
    reversed_value = 0
    for _ in range(width):
        reversed_value = (reversed_value << 1) | (value & 1)
        value >>= 1
    return reversed_value


def _sample_bitstrings(
    rng: random.Random,
    probabilities: Sequence[float],
    num_qubits: int,
    shots: int,
) -> List[str]:
    outcomes = [format(index, f"0{num_qubits}b") for index in range(len(probabilities))]
    return rng.choices(outcomes, weights=probabilities, k=shots)


def _print_metric(metric: ComparisonMetrics) -> None:
    summary = (
        f"[qubits={metric.num_qubits} depth={metric.depth} circuit={metric.circuit_index}] "
        f"amp_err={metric.amplitude_error:.3e} "
        f"prob_err={metric.probability_error:.3e}"
    )
    if metric.xeb_error is not None:
        summary += f" xeb_err={metric.xeb_error:.3e}"
    print(summary)


def _compute_summary(metrics: List[ComparisonMetrics]) -> Dict[str, float]:
    max_amp = max(metric.amplitude_error for metric in metrics) if metrics else 0.0
    max_prob = max(metric.probability_error for metric in metrics) if metrics else 0.0
    max_xeb = max(
        (metric.xeb_error for metric in metrics if metric.xeb_error is not None),
        default=0.0,
    )
    return {
        "max_amplitude_error": max_amp,
        "max_probability_error": max_prob,
        "max_xeb_error": max_xeb,
    }


def _print_summary(summary: Dict[str, float]) -> None:
    print("=== Comparison Summary ===")
    print(f"Max amplitude error   : {summary['max_amplitude_error']:.6e}")
    print(f"Max probability error : {summary['max_probability_error']:.6e}")
    print(f"Max XEB error         : {summary['max_xeb_error']:.6e}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

