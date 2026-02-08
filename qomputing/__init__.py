"""Qomputing Simulator package."""

from .backend import Backend, QomputingSimulator, Result
from .circuit import Gate, QuantumCircuit
from .engine import SimulationResult, StateVectorSimulator
from .run import load_circuit, random_circuit, run, run_xeb
from .xeb import LinearXEBResult, compute_linear_xeb_fidelity, run_linear_xeb_experiment

__all__ = [
    "Gate",
    "QuantumCircuit",
    "StateVectorSimulator",
    "SimulationResult",
    "LinearXEBResult",
    "compute_linear_xeb_fidelity",
    "run_linear_xeb_experiment",
    # Backend API (Qomputing style)
    "QomputingSimulator",
    "Backend",
    "Result",
    # Library API
    "load_circuit",
    "run",
    "run_xeb",
    "random_circuit",
]

