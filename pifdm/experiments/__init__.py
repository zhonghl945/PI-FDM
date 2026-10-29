"""Experiment APIs with lazy ML imports so configuration needs no PyTorch."""

from .contracts import ALL_METHODS as METHODS, PHYSICS_METHODS
from .registry import CASES, get_case

__all__ = ["CASES", "METHODS", "PHYSICS_METHODS", "get_case",
           "build_experiment", "build_physics_loss", "build_boundary_loss"]


def __getattr__(name):
    if name == "build_experiment":
        from .runner import build_experiment
        return build_experiment
    if name == "build_physics_loss":
        from .physics import build_physics_loss
        return build_physics_loss
    if name == "build_boundary_loss":
        from .physics import build_boundary_loss
        return build_boundary_loss
    raise AttributeError(name)
