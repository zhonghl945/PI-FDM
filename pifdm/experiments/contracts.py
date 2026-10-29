"""Method-specific configuration contracts.

The case-study files deliberately expose plain Python dictionaries.  These
checks make a misspelled or misplaced option fail before a long cluster job
starts instead of being silently ignored.
"""

from __future__ import annotations

from collections.abc import Mapping

from .naming import CFDM_METHODS, PIFDM_METHODS


DIFFUSION_METHODS = (
    "fdm_unet", "fdm_vit", *CFDM_METHODS, *PIFDM_METHODS, "diffusionpde",
)
POINT_OPERATOR_METHODS = ("deeponet", "pideeponet")
GRID_OPERATOR_METHODS = ("fno", "pino")
OPERATOR_METHODS = (*POINT_OPERATOR_METHODS, *GRID_OPERATOR_METHODS)
GRID_PHYSICS_METHODS = ("pino", "diffusionpde")
PHYSICS_METHODS = (*PIFDM_METHODS, "pideeponet", *GRID_PHYSICS_METHODS)
ALL_METHODS = (*DIFFUSION_METHODS, *OPERATOR_METHODS)

_EMA_KEYS = {"ema_decay", "ema_update_every"}


def validate_experiment_config(config: Mapping) -> None:
    """Reject method options that have no meaning for the selected method."""
    method = config.get("method")
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {ALL_METHODS}")

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError(f"{method} requires a training dictionary")
    if method in DIFFUSION_METHODS:
        if not isinstance(config.get("diffusion"), Mapping):
            raise ValueError(f"{method} requires diffusion settings")
    else:
        if "diffusion" in config:
            raise ValueError(f"{method} is an operator model and must not define diffusion settings")
        unused_ema = sorted(_EMA_KEYS.intersection(training))
        if unused_ema:
            raise ValueError(f"{method} does not use diffusion EMA settings: {unused_ema}")
    if "epochs" in training:
        raise ValueError(f"{method} is step-based: use training.steps, not training.epochs")
    if int(training.get("steps", 0)) <= 0:
        raise ValueError(f"{method} requires training.steps > 0")
    if method in PHYSICS_METHODS:
        physics_training = config.get("physics_training")
        if not isinstance(physics_training, Mapping):
            raise ValueError(f"{method} requires its own physics_training dictionary")
        if float(physics_training.get("lambda_phy", 0.0)) <= 0.0:
            raise ValueError(f"{method} requires physics_training.lambda_phy > 0")
        point_sampling = physics_training.get("point_sampling", "random")
        if point_sampling not in ("random", "grid", "grid_subsample"):
            raise ValueError(
                "physics_training.point_sampling must be 'random', 'grid', or 'grid_subsample'"
            )
        boundary_sampling = physics_training.get("boundary_point_sampling", point_sampling)
        if boundary_sampling not in ("random", "grid"):
            raise ValueError(
                "physics_training.boundary_point_sampling must be 'random' or 'grid'"
            )
        if method in GRID_PHYSICS_METHODS and point_sampling != "grid":
            raise ValueError(f"{method} computes physics only on the saved grid")
        if point_sampling in ("random", "grid_subsample") and int(physics_training.get("collocation_points", 0)) <= 0:
            raise ValueError(
                f"{method} requires physics_training.collocation_points > 0 "
                "in sampled point_sampling mode"
            )
        huber_delta = float(physics_training.get("huber_delta", 1.0))
        if huber_delta <= 0.0:
            raise ValueError("physics_training.huber_delta must be positive")
        from .registry import get_case
        requires_robin_loss = get_case(config.get("case")).boundary_slots == "operator"
        if requires_robin_loss and "lambda_bc" not in physics_training:
            raise ValueError(f"{method} requires physics_training.lambda_bc for Robin boundaries")
        if "lambda_bc" in physics_training:
            if float(physics_training["lambda_bc"]) <= 0.0:
                raise ValueError(f"{method} requires physics_training.lambda_bc > 0")
            if boundary_sampling == "random" and int(physics_training.get("boundary_points", 0)) <= 0:
                raise ValueError(
                    f"{method} requires physics_training.boundary_points > 0 "
                    "in random point_sampling mode"
                )
        required_schedule = {"start_step", "warmup_steps"}
        missing_schedule = sorted(required_schedule.difference(physics_training))
        if missing_schedule:
            raise ValueError(f"{method} is missing physics schedule settings: {missing_schedule}")
        wrong_schedule = {"start_epoch", "warmup_epochs"}
        misplaced_schedule = sorted(wrong_schedule.intersection(physics_training))
        if misplaced_schedule:
            raise ValueError(f"{method} uses the wrong physics schedule unit: {misplaced_schedule}")
    elif "physics_training" in config:
        raise ValueError(f"{method} must not define physics_training settings")
