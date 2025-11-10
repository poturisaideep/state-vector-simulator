"""Sample circuits that can be executed with the state vector simulator.

Examples included:
  * Bell state (2 qubits)
  * Deutsch-Jozsa (2 qubits, constant or balanced oracle)
  * GHZ / Greenberger–Horne–Zeilinger (3 qubits)

Run with:

  python -m state_vector_simulator.examples.demo_circuits --example bell
  python -m state_vector_simulator.examples.demo_circuits --example deutsch-jozsa --oracle balanced --shots 1024
  python -m state_vector_simulator.examples.demo_circuits --example ghz --shots 1000 --seed 42
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict

import numpy as np

from state_vector_simulator.circuit import QuantumCircuit
from state_vector_simulator.simulator import StateVectorSimulator


def build_bell_circuit() -> QuantumCircuit:
    """Prepare the Bell state (|00> + |11>)/sqrt(2)."""
    circuit = QuantumCircuit(2)
    circuit.h(0).cx(0, 1)
    return circuit


def build_deutsch_jozsa_circuit(oracle: str) -> QuantumCircuit:
    """Construct a two-qubit Deutsch–Jozsa circuit for the requested oracle."""
    oracle = oracle.lower()
    if oracle not in {"constant", "balanced"}:
        raise ValueError("Deutsch–Jozsa oracle must be 'constant' or 'balanced'")

    circuit = QuantumCircuit(2)
    # Initialise |0>|1>
    circuit.x(1)
    # Create |+>|-> before the oracle.
    circuit.h(0).h(1)

    if oracle == "balanced":
        # Oracle implements f(x) = x => CNOT
        circuit.cx(0, 1)
    else:
        # Oracle implements f(x) = 0 => identity (no additional gates)
        pass

    # Interfere the first qubit to reveal the oracle type.
    circuit.h(0)
    return circuit


def build_ghz_circuit() -> QuantumCircuit:
    """Prepare a three-qubit GHZ state (|000> + |111>)/sqrt(2)."""
    circuit = QuantumCircuit(3)
    circuit.h(0).cx(0, 1).cx(1, 2)
    return circuit


def run_example(factory: Callable[[], QuantumCircuit], shots: int, seed: int | None) -> None:
    circuit = factory()
    simulator = StateVectorSimulator()
    result = simulator.run(circuit, shots=shots if shots > 0 else None, seed=seed)

    labels = [format(index, f"0{circuit.num_qubits}b") for index in range(2**circuit.num_qubits)]

    print(f"Number of qubits: {circuit.num_qubits}")
    print("Final state amplitudes:")
    print(np.array2string(result.final_state, precision=6, suppress_small=True))

    print("Measurement probabilities per computational basis state:")
    for label, probability in zip(labels, result.probabilities):
        if probability > 1e-6:
            print(f"  P(|{label}⟩) = {probability:.6f}")

    if result.counts:
        print(f"\nSampled {result.shots} shots (seed={seed}):")
        for bitstring, count in sorted(result.counts.items()):
            print(f"  {bitstring}: {count} times")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run example circuits on the state vector simulator.")
    parser.add_argument(
        "--example",
        type=str,
        required=True,
        choices=["bell", "deutsch-jozsa", "ghz"],
        help="Which example circuit to run.",
    )
    parser.add_argument(
        "--oracle",
        type=str,
        default="constant",
        help="Oracle type for the Deutsch–Jozsa example (constant or balanced).",
    )
    parser.add_argument("--shots", type=int, default=0, help="Number of measurement shots to sample.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    example_builders: Dict[str, Callable[[], QuantumCircuit]] = {
        "bell": build_bell_circuit,
        "ghz": build_ghz_circuit,
    }

    if args.example == "deutsch-jozsa":
        factory = lambda: build_deutsch_jozsa_circuit(args.oracle)  # noqa: E731
    else:
        factory = example_builders[args.example]

    run_example(factory, shots=args.shots, seed=args.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


