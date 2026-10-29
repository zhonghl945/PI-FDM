"""Core APIs with lazy ML imports so data generation does not require PyTorch."""

__all__ = ["FrozenConditioning", "GaussianDiffusion", "OperatorObjective", "GridOperatorObjective",
           "PhysicsSchedule", "Trainer"]


def __getattr__(name):
    if name in ("FrozenConditioning", "GaussianDiffusion"):
        from .diffusion import FrozenConditioning, GaussianDiffusion
        return {"FrozenConditioning": FrozenConditioning, "GaussianDiffusion": GaussianDiffusion}[name]
    if name in ("PhysicsSchedule", "Trainer"):
        from .training import PhysicsSchedule, Trainer
        return {"PhysicsSchedule": PhysicsSchedule, "Trainer": Trainer}[name]
    if name in ("OperatorObjective", "GridOperatorObjective"):
        from .objectives import GridOperatorObjective, OperatorObjective
        return {"OperatorObjective": OperatorObjective,
                "GridOperatorObjective": GridOperatorObjective}[name]
    raise AttributeError(name)
