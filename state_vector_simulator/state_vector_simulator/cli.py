"""Command line interface for the state vector simulator."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

from .circuit import QuantumCircuit
from .gates import (
    DEFAULT_MULTI_QUBIT_GATES,
    DEFAULT_SINGLE_QUBIT_GATES,
    DEFAULT_TWO_QUBIT_GATES,
)
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
        default=DEFAULT_SINGLE_QUBIT_GATES,
        help="Set of single-qubit gates to sample from",
    )
    rand_parser.add_argument(
        "--two-qubit-gates",
        type=str,
        nargs="*",
        default=DEFAULT_TWO_QUBIT_GATES,
        help="Set of two-qubit gates to sample from",
    )
    rand_parser.add_argument(
        "--multi-qubit-gates",
        type=str,
        nargs="*",
        default=DEFAULT_MULTI_QUBIT_GATES,
        help="Set of multi-qubit gates (three or more qubits) to sample from",
    )

    return parser


def _cmd_simulate(args: argparse.Namespace) -> int:
    circuit = _load_circuit(args.circuit)
    simulator = StateVectorSimulator()
    result = simulator.run(circuit, shots=args.shots, seed=args.seed)
    payload = {
        "num_qubits": circuit.num_qubits,
        "shots": result.shots,
        "final_state_real": result.final_state.real.tolist(),
        "final_state_imag": result.final_state.imag.tolist(),
        "probabilities": result.probabilities.tolist(),
        "counts": result.counts,
        "samples": result.samples,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_random_circuit(args: argparse.Namespace) -> int:
    single_qubit_gates = args.single_qubit_gates
    two_qubit_gates = args.two_qubit_gates
    multi_qubit_gates = args.multi_qubit_gates
    circuit = _random_circuit(
        num_qubits=args.qubits,
        depth=args.depth,
        single_qubit_gates=single_qubit_gates,
        two_qubit_gates=two_qubit_gates,
        multi_qubit_gates=multi_qubit_gates,
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
    two_qubit_gates: Iterable[str],
    multi_qubit_gates: Iterable[str],
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

    two_qubit_gates = list(two_qubit_gates)
    multi_qubit_gates = list(multi_qubit_gates)
    if num_qubits > 1 and not two_qubit_gates:
        raise ValueError("Two-qubit gate set must not be empty when num_qubits > 1")

    multi_specs: Dict[str, Tuple[int, int]] = {
        "ccx": (2, 1),
        "ccz": (2, 1),
        "cswap": (1, 2),
    }

    circuit = QuantumCircuit(num_qubits)

    for _layer in range(depth):
        for qubit in range(num_qubits):
            gate_name = rng.choice(single_qubit_gates)
            _apply_single_qubit_gate(circuit, gate_name, qubit, rng)

        if num_qubits > 1 and two_qubit_gates:
            gate_name = rng.choice(two_qubit_gates)
            q1, q2 = rng.sample(range(num_qubits), 2)
            _apply_two_qubit_gate(circuit, gate_name, q1, q2, rng)

        if num_qubits > 2 and multi_qubit_gates:
            gate_name = rng.choice(multi_qubit_gates)
            if gate_name not in multi_specs:
                raise ValueError(f"Unsupported multi-qubit gate in random circuit: {gate_name}")
            controls_required, targets_required = multi_specs[gate_name]
            total_required = controls_required + targets_required
            if num_qubits >= total_required:
                selected = rng.sample(range(num_qubits), total_required)
                controls = selected[:controls_required]
                targets = selected[controls_required:]
                _apply_multi_qubit_gate(circuit, gate_name, controls, targets, rng)

    return circuit


def _apply_single_qubit_gate(circuit: QuantumCircuit, gate_name: str, qubit: int, rng: random.Random) -> None:
    two_pi = 2.0 * math.pi
    if gate_name == "id":
        circuit.i(qubit)
    elif gate_name == "x":
        circuit.x(qubit)
    elif gate_name == "y":
        circuit.y(qubit)
    elif gate_name == "z":
        circuit.z(qubit)
    elif gate_name == "h":
        circuit.h(qubit)
    elif gate_name == "s":
        circuit.s(qubit)
    elif gate_name == "sdg":
        circuit.sdg(qubit)
    elif gate_name == "t":
        circuit.t(qubit)
    elif gate_name == "tdg":
        circuit.tdg(qubit)
    elif gate_name == "sx":
        circuit.sx(qubit)
    elif gate_name == "sxdg":
        circuit.sxdg(qubit)
    elif gate_name == "rx":
        circuit.rx(qubit, rng.uniform(0.0, two_pi))
    elif gate_name == "ry":
        circuit.ry(qubit, rng.uniform(0.0, two_pi))
    elif gate_name == "rz":
        circuit.rz(qubit, rng.uniform(0.0, two_pi))
    elif gate_name == "u1":
        circuit.u1(qubit, rng.uniform(0.0, two_pi))
    elif gate_name == "u2":
        circuit.u2(qubit, rng.uniform(0.0, two_pi), rng.uniform(0.0, two_pi))
    elif gate_name == "u3":
        circuit.u3(
            qubit,
            rng.uniform(0.0, two_pi),
            rng.uniform(0.0, two_pi),
            rng.uniform(0.0, two_pi),
        )
    else:
        raise ValueError(f"Unsupported single-qubit gate: {gate_name}")


def _apply_two_qubit_gate(
    circuit: QuantumCircuit,
    gate_name: str,
    q1: int,
    q2: int,
    rng: random.Random,
) -> None:
    two_pi = 2.0 * math.pi
    if gate_name == "cx":
        circuit.cx(q1, q2)
    elif gate_name == "cy":
        circuit.cy(q1, q2)
    elif gate_name == "cz":
        circuit.cz(q1, q2)
    elif gate_name == "cp":
        circuit.cp(q1, q2, rng.uniform(0.0, two_pi))
    elif gate_name == "swap":
        circuit.swap(q1, q2)
    elif gate_name == "iswap":
        circuit.iswap(q1, q2)
    elif gate_name == "sqrtiswap":
        circuit.sqrt_iswap(q1, q2)
    elif gate_name == "rxx":
        circuit.rxx(q1, q2, rng.uniform(0.0, two_pi))
    elif gate_name == "ryy":
        circuit.ryy(q1, q2, rng.uniform(0.0, two_pi))
    elif gate_name == "rzz":
        circuit.rzz(q1, q2, rng.uniform(0.0, two_pi))
    elif gate_name == "csx":
        circuit.csx(q1, q2)
    else:
        raise ValueError(f"Unsupported two-qubit gate in random circuit: {gate_name}")


def _apply_multi_qubit_gate(
    circuit: QuantumCircuit,
    gate_name: str,
    controls: Sequence[int],
    targets: Sequence[int],
    rng: random.Random,
) -> None:
    if gate_name == "ccx":
        circuit.ccx(controls[0], controls[1], targets[0])
    elif gate_name == "ccz":
        circuit.ccz(controls[0], controls[1], targets[0])
    elif gate_name == "cswap":
        circuit.cswap(controls[0], targets[0], targets[1])
    else:
        raise ValueError(f"Unsupported multi-qubit gate in random circuit: {gate_name}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

