#!/usr/bin/env python3
"""Example: using Qomputing Simulator as a library (backend API + simple run)."""

from qomputing_simulator import (
    QomputingSimulator,
    QuantumCircuit,
    load_circuit,
    random_circuit,
    run,
    run_xeb,
)

# --- Backend API (recommended) ---
backend = QomputingSimulator.get_backend("state_vector")

# Circuit with qubits + classical bits, explicit measure
qc = QuantumCircuit(4, 4)
qc.h(0).cx(0, 1)
qc.measure(range(3), range(3))

result = backend.run(qc, shots=1024, seed=42)
counts = result.get_counts()
print("Backend API – get_counts():", counts)
print("Backend API – get_statevector() shape:", result.get_statevector().shape)

# Bell state with backend (no classical bits)
qc_bell = QuantumCircuit(2)
qc_bell.h(0).cx(0, 1)
result_bell = backend.run(qc_bell, shots=1000, seed=42)
print("Bell counts:", result_bell.get_counts())

# --- Simple API (run, no backend) ---
circuit = QuantumCircuit(2)
circuit.h(0).cx(0, 1)
result = run(circuit, shots=1000, seed=42)
print("Simple run() – counts:", result.counts)

# Random circuit + XEB
qc_rand = random_circuit(num_qubits=3, depth=5, seed=7)
xeb = run_xeb(qc_rand, shots=1000, seed=7)
print("XEB fidelity:", round(xeb.fidelity, 4))

# Optional: load from JSON
# circuit = load_circuit("circuits/example.json")
# result = run(circuit, shots=512)
