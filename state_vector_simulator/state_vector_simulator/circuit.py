"""Circuit primitives for the state vector simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _ensure_tuple(values: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class Gate:
    """Immutable gate description."""

    name: str
    targets: Tuple[int, ...]
    controls: Tuple[int, ...] = field(default_factory=tuple)
    params: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.lower()
        object.__setattr__(self, "name", name)

        if not self.targets:
            raise ValueError("Gate must have at least one target qubit")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError(f"Duplicate target qubits in gate {self.name}")
        if any(t < 0 for t in self.targets):
            raise ValueError("Target qubit indices must be non-negative")

        if len(set(self.controls)) != len(self.controls):
            raise ValueError(f"Duplicate control qubits in gate {self.name}")
        if any(c < 0 for c in self.controls):
            raise ValueError("Control qubit indices must be non-negative")

        overlap = set(self.targets) & set(self.controls)
        if overlap:
            raise ValueError(f"Control and target sets overlap for gate {self.name}: {overlap}")


class QuantumCircuit:
    """Simple representation of a quantum circuit."""

    def __init__(self, num_qubits: int) -> None:
        if num_qubits <= 0:
            raise ValueError("Circuit must have at least one qubit")
        self.num_qubits = int(num_qubits)
        self._gates: List[Gate] = []

    @property
    def gates(self) -> Tuple[Gate, ...]:
        return tuple(self._gates)

    def add_gate(
        self,
        name: str,
        targets: Sequence[int],
        *,
        controls: Sequence[int] | None = None,
        params: Mapping[str, float] | None = None,
    ) -> "QuantumCircuit":
        gate = Gate(
            name=name,
            targets=_ensure_tuple(targets),
            controls=_ensure_tuple(controls or ()),
            params=dict(params or {}),
        )
        self._validate_gate_qubits(gate)
        self._gates.append(gate)
        return self

    # Convenience builders -------------------------------------------------
    def x(self, target: int) -> "QuantumCircuit":
        return self.add_gate("x", [target])

    def y(self, target: int) -> "QuantumCircuit":
        return self.add_gate("y", [target])

    def z(self, target: int) -> "QuantumCircuit":
        return self.add_gate("z", [target])

    def h(self, target: int) -> "QuantumCircuit":
        return self.add_gate("h", [target])

    def s(self, target: int) -> "QuantumCircuit":
        return self.add_gate("s", [target])

    def t(self, target: int) -> "QuantumCircuit":
        return self.add_gate("t", [target])

    def rx(self, target: int, theta: float) -> "QuantumCircuit":
        return self.add_gate("rx", [target], params={"theta": float(theta)})

    def ry(self, target: int, theta: float) -> "QuantumCircuit":
        return self.add_gate("ry", [target], params={"theta": float(theta)})

    def rz(self, target: int, theta: float) -> "QuantumCircuit":
        return self.add_gate("rz", [target], params={"theta": float(theta)})

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        return self.add_gate("cx", [target], controls=[control])

    # Serialisation --------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "num_qubits": self.num_qubits,
            "gates": [
                {
                    "name": gate.name,
                    "targets": list(gate.targets),
                    **(
                        {"controls": list(gate.controls)}
                        if gate.controls
                        else {}
                    ),
                    **({"params": dict(gate.params)} if gate.params else {}),
                }
                for gate in self._gates
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "QuantumCircuit":
        num_qubits = int(payload["num_qubits"])
        circuit = cls(num_qubits)
        for gate_payload in payload.get("gates", []):
            if not isinstance(gate_payload, Mapping):
                raise TypeError("Gate payload must be a mapping")
            circuit.add_gate(
                str(gate_payload["name"]),
                gate_payload.get("targets", ()),
                controls=gate_payload.get("controls"),
                params=gate_payload.get("params"),
            )
        return circuit

    # Internal helpers -----------------------------------------------------
    def _validate_gate_qubits(self, gate: Gate) -> None:
        for index in gate.targets + gate.controls:
            if index < 0 or index >= self.num_qubits:
                raise ValueError(
                    f"Qubit index {index} for gate {gate.name} "
                    f"is out of range for circuit with {self.num_qubits} qubits",
                )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"QuantumCircuit(num_qubits={self.num_qubits}, gates={self._gates!r})"

