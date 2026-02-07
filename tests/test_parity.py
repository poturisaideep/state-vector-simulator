import math
import numpy as np
import cirq
import pytest

from qomputing_simulator import QuantumCircuit, StateVectorSimulator


def _align_global_phase(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    phase = 1 + 0j
    for ref, cand in zip(reference, candidate):
        if abs(cand) > 1e-12:
            phase = ref / cand
            break
    return candidate * phase


def _to_little_endian(state: np.ndarray, num_qubits: int) -> np.ndarray:
    reordered = np.zeros_like(state)
    for index, amplitude in enumerate(state):
        little_index = int("{:0{width}b}".format(index, width=num_qubits)[::-1], 2)
        reordered[little_index] = amplitude
    return reordered


def _run_and_compare(circuit: QuantumCircuit) -> float:
    ours = StateVectorSimulator().run(circuit).final_state
    q = cirq.LineQubit.range(circuit.num_qubits)
    ops = []
    for gate in circuit.gates:
        name = gate.name
        t = gate.targets
        c = gate.controls
        params = gate.params
        if name == "h":
            ops.append(cirq.H(q[t[0]]))
        elif name == "x":
            ops.append(cirq.X(q[t[0]]))
        elif name == "y":
            ops.append(cirq.Y(q[t[0]]))
        elif name == "z":
            ops.append(cirq.Z(q[t[0]]))
        elif name == "s":
            ops.append(cirq.S(q[t[0]]))
        elif name == "sdg":
            ops.append((cirq.S ** -1)(q[t[0]]))
        elif name == "t":
            ops.append(cirq.T(q[t[0]]))
        elif name == "tdg":
            ops.append((cirq.T ** -1)(q[t[0]]))
        elif name == "sx":
            ops.append((cirq.X ** 0.5)(q[t[0]]))
        elif name == "sxdg":
            ops.append((cirq.X ** -0.5)(q[t[0]]))
        elif name == "rx":
            ops.append(cirq.rx(params["theta"])(q[t[0]]))
        elif name == "ry":
            ops.append(cirq.ry(params["theta"])(q[t[0]]))
        elif name == "rz":
            ops.append(cirq.rz(params["theta"])(q[t[0]]))
        elif name == "cx":
            ops.append(cirq.CNOT(q[c[0]], q[t[0]]))
        elif name == "cz":
            ops.append(cirq.CZ(q[c[0]], q[t[0]]))
        elif name == "swap":
            ops.append(cirq.SWAP(q[t[0]], q[t[1]]))
        elif name == "rxx":
            ops.append(cirq.XXPowGate(exponent=params["theta"] / np.pi)(q[t[0]], q[t[1]]))
        elif name == "ccx":
            ops.append(cirq.CCX(q[c[0]], q[c[1]], q[t[0]]))
        elif name == "ccz":
            ops.append(cirq.CCZ(q[c[0]], q[c[1]], q[t[0]]))
        elif name == "cswap":
            ops.append(cirq.CSWAP(q[c[0]], q[t[0]], q[t[1]]))
        else:
            raise ValueError(f"Unsupported gate {name} in parity test")
    cirq_state = cirq.Simulator(dtype=np.complex128).simulate(cirq.Circuit(ops), qubit_order=q).final_state_vector
    cirq_state = _to_little_endian(cirq_state, circuit.num_qubits)
    cirq_state = _align_global_phase(ours, cirq_state)
    return float(np.max(np.abs(ours - cirq_state)))


@pytest.mark.parametrize("gate_name", ["h", "sx", "sxdg", "sdg", "t", "tdg"])
def test_single_qubit_gates(gate_name):
    qc = QuantumCircuit(1)
    qc.ry(0, 0.73).rz(0, -0.42)
    getattr(qc, gate_name)(0)
    assert _run_and_compare(qc) < 1e-7


@pytest.mark.parametrize("gate_name", ["rx", "ry", "rz"])
def test_single_qubit_rotations(gate_name):
    angle = 0.37 + {"rx": 0.1, "ry": 0.2, "rz": 0.3}[gate_name]
    qc = QuantumCircuit(1)
    qc.h(0)
    getattr(qc, gate_name)(0, angle)
    qc.h(0)
    assert _run_and_compare(qc) < 1e-7


@pytest.mark.parametrize("gate_name", ["cx", "cz", "swap", "rxx"])
def test_two_qubit_gates(gate_name):
    qc = QuantumCircuit(2)
    qc.h(0).ry(1, 0.56)
    if gate_name == "rxx":
        qc.rxx(0, 1, 0.42)
    elif gate_name == "swap":
        qc.swap(0, 1)
    else:
        getattr(qc, gate_name)(0, 1)
    qc.rx(0, 0.11).rz(1, -0.33)
    assert _run_and_compare(qc) < 1e-7


@pytest.mark.parametrize("gate_name", ["ccx", "ccz", "cswap"])
def test_multi_qubit_gates(gate_name):
    qc = QuantumCircuit(3)
    qc.h(0).h(1).rx(2, 0.25)
    if gate_name == "cswap":
        qc.cswap(0, 1, 2)
    else:
        getattr(qc, gate_name)(0, 1, 2)
    qc.ry(0, 0.1).rz(1, -0.2)
    assert _run_and_compare(qc) < 1e-7

