#!/usr/bin/env python3
"""Example: using the state-vector simulator as a library."""

from qomputing_simulator import (
    QuantumCircuit,
    load_circuit,
    random_circuit,
    run,
    run_xeb,
)

# 1) Build a circuit in code and run it (no shots = state vector only)
circuit = QuantumCircuit(2)
circuit.h(0).cx(0, 1)
result = run(circuit)
print("Bell circuit final state (real):", result.final_state.real.round(4).tolist())
print("Probabilities:", result.probabilities.round(4).tolist())

# 2) Run with measurement shots
result = run(circuit, shots=1000, seed=42)
print("Counts:", result.counts)

# 3) Load a circuit from JSON and run
# result = run("circuits/example.json", shots=512, seed=123)

# 4) Random circuit and XEB
qc = random_circuit(num_qubits=3, depth=5, seed=7)
xeb = run_xeb(qc, shots=1000, seed=7)
print("XEB fidelity:", round(xeb.fidelity, 4))
print("Sample probabilities:", xeb.sample_probabilities)
