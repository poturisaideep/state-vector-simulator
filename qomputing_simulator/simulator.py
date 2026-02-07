"""Backward-compatible shim exposing the simulation engine."""

from .engine import SimulationResult, StateVectorSimulator

__all__ = ["SimulationResult", "StateVectorSimulator"]

