"""Command line interface for the state vector simulator."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

from .circuit import QuantumCircuit
from .simulator import StateVectorSimulator
from .xeb import run_linear_xeb_experiment


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "simulate":
        return _cmd_simulate(args)
    if args.command == "random-circuit":
        return _cmd_random_circuit(args)

    parser.print_help()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="State vector simulator CLI")
    subparsers = parser.add_subparsers(dest="command")

    simulate_parser = subparsers.add_parser("simulate", help="Simulate a circuit from JSON")
    simulate_parser.add_argument("--circuit", type=Path, required=True, help="Path to circuit JSON file")
    simulate_parser.add_argument("--shots", type=int, default=0, help="Number of measurement shots to sample")
    simulate_parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")

    rand_parser = subparsers.add_parser("random-circuit", help="Generate a random circuit and run XEB")
    rand_parser.add_argument("--qubits", type=int, required=True, help="Number of qubits")
    rand_parser.add_argument("--depth", type=int, required=True, help="Circuit depth (layers)")
    rand_parser.add_argument("--shots", type=int, required=True, help="Number of measurement shots")
    rand_parser.add_argument("--seed", type=int, default=None, help="Random seed for circuit generation and sampling")
    rand_parser.add_argument(
        "--single-qubit-gates",
        type=str,
        nargs="*",
        default=["h", "rx", "ry", "rz", "s", "t"],
        help="Set of single-qubit gates to sample from",
    )

    return parser


def _cmd_simulate(args: argparse.Namespace) -> int:
    circuit = _load_circuit(args.circuit)
    simulator = StateVectorSimulator()
    result = simulator.run(circuit, shots=args.shots, seed=args.seed)
    payload = {
        "num_qubits": circuit.num_qubits,
        "shots": result.shots,
        "final_state_real": [amp.real for amp in result.final_state],
        "final_state_imag": [amp.imag for amp in result.final_state],
        "probabilities": result.probabilities,
        "counts": result.counts,
        "samples": result.samples,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_random_circuit(args: argparse.Namespace) -> int:
    single_qubit_gates = args.single_qubit_gates
    circuit = _random_circuit(
        num_qubits=args.qubits,
        depth=args.depth,
        single_qubit_gates=single_qubit_gates,
        seed=args.seed,
    )
    xeb_result = run_linear_xeb_experiment(
        circuit,
        shots=args.shots,
        seed=args.seed,
    )
    payload = {
        "num_qubits": circuit.num_qubits,
        "depth": args.depth,
        "shots": args.shots,
        "fidelity": xeb_result.fidelity,
        "sample_probabilities": xeb_result.sample_probabilities,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _load_circuit(path: Path) -> QuantumCircuit:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return QuantumCircuit.from_dict(payload)


def _random_circuit(
    *,
    num_qubits: int,
    depth: int,
    single_qubit_gates: Iterable[str],
    seed: int | None,
) -> QuantumCircuit:
    if num_qubits <= 0:
        raise ValueError("Number of qubits must be positive")
    if depth <= 0:
        raise ValueError("Depth must be positive")
    single_qubit_gates = list(single_qubit_gates)
    if not single_qubit_gates:
        raise ValueError("Single-qubit gate set must not be empty")

    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)
    for _layer in range(depth):
        # Apply a random single-qubit gate to each qubit
        for qubit in range(num_qubits):
            gate_name = rng.choice(single_qubit_gates)
            if gate_name in {"rx", "ry", "rz"}:
                theta = float(rng.uniform(0.0, 2.0 * math.pi))
                circuit.add_gate(gate_name, [qubit], params={"theta": theta})
            else:
                circuit.add_gate(gate_name, [qubit])

        # Add a random CX pair
        if num_qubits > 1:
            control, target = rng.sample(range(num_qubits), 2)
            circuit.cx(control, target)

    return circuit


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

