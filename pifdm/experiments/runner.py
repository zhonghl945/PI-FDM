"""Reference assembly for a single configured experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pifdm.core import (FrozenConditioning, GaussianDiffusion, GridOperatorObjective, OperatorObjective,
                        PhysicsSchedule, Trainer)
from pifdm.experiments.registry import get_case
from pifdm.experiments.contracts import (DIFFUSION_METHODS, GRID_OPERATOR_METHODS,
                                          OPERATOR_METHODS, POINT_OPERATOR_METHODS,
                                          validate_experiment_config)
from pifdm.models.methods import build_method


@dataclass
class Experiment:
    objective: torch.nn.Module
    trainer: Trainer


def build_experiment(config: dict[str, Any], *, physics_loss_fn=None, boundary_loss_fn=None,
                     scheduler_updates: int | None = None, grid_coordinates=None,
                     material_condition=None, material_function=None) -> Experiment:
    """Build a model/trainer from a plain dictionary.

    This intentionally does not generate data or start a long-running job;
    callers can load a dataset, set seeds and choose their accelerator first.
    """

    validate_experiment_config(config)
    case = get_case(config["case"])
    shape = tuple(config["grid_shape"])
    method = config["method"]
    left_robin_conditioning = config["case"] == "radiative_2d"
    model_method = {
        "c_fdm_unet": "pifdm_unet",
        "c_fdm_vit": "pifdm_vit",
        "diffusionpde": "fdm_unet",
    }.get(method, method)
    # Initial E and Robin g are both frozen conditions.  Their participation
    # in reconstruction is case-configurable because g is not a boundary
    # value of E.
    reconstruction = config.get("reconstruction_loss", {})
    reconstruct_initial = bool(reconstruction.get("include_initial_condition", True))
    reconstruct_boundaries = bool(
        reconstruction.get("include_robin_boundary", case.boundary_slots != "operator")
    )
    # Diffusion backbones normally receive [noisy/frozen field, binary
    # condition mask, material].  The optional condition-channel ablation
    # removes only those extra channels; FrozenConditioning below still
    # enforces the known field values during noising and sampling.
    if method in POINT_OPERATOR_METHODS:
        in_channels = config.get("in_channels", 1)
    elif method in DIFFUSION_METHODS:
        use_condition_channels = bool(config.get("use_condition_channels", True))
        in_channels = config.get("in_channels", 1) + int(left_robin_conditioning)
        in_channels += int(use_condition_channels)
        if use_condition_channels and material_condition is not None:
            in_channels += int(material_condition.shape[1])
    else:
        # FNO/PINO retain their known-condition-plus-mask contract.  The
        # radiative_2d layout adds one clean gL channel, not packed boundaries.
        in_channels = config.get("in_channels", 1) + 1 + int(left_robin_conditioning)
        if material_condition is not None:
            in_channels += int(material_condition.shape[1])
    model = build_method(model_method, input_shape=shape, coord_dim=case.dimension + 1,
                         in_channels=in_channels, grid_coordinates=grid_coordinates,
                         material_function=material_function,
                         **config.get("model", {}))
    if method in GRID_OPERATOR_METHODS:
        objective = GridOperatorObjective(
            model, solution_shape=shape, physics_loss_fn=physics_loss_fn,
            boundary_loss_fn=boundary_loss_fn,
            reconstruct_boundaries=reconstruct_boundaries,
            reconstruct_initial=reconstruct_initial,
            material_condition=material_condition,
            left_robin_conditioning=left_robin_conditioning,
        )
    elif method in OPERATOR_METHODS:
        objective = OperatorObjective(
            model, solution_shape=shape, physics_loss_fn=physics_loss_fn,
            boundary_loss_fn=boundary_loss_fn,
            reconstruct_boundaries=reconstruct_boundaries,
            reconstruct_initial=reconstruct_initial,
            left_robin_conditioning=left_robin_conditioning,
        )
    else:
        diffusion = config.get("diffusion", {})
        objective = GaussianDiffusion(
            model, solution_shape=shape,
            timesteps=diffusion.get("timesteps", 1000), beta_schedule=diffusion.get("beta_schedule", "cosine"),
            sampling_timesteps=diffusion.get("sampling_timesteps"), ddim_eta=diffusion.get("eta", 0.0),
            # ``radiative_2d`` stores the Robin right-hand side separately;
            # only the t=0 E slice belongs to the field mask.  All other
            # cases retain the historical spatial-boundary mask.
            conditioning=FrozenConditioning(
                case.dimension,
                freeze_spatial_boundaries=not left_robin_conditioning,
            ),
            physics_loss_fn=physics_loss_fn, boundary_loss_fn=boundary_loss_fn,
            reconstruct_boundaries=reconstruct_boundaries,
            reconstruct_initial=reconstruct_initial,
            material_condition=material_condition,
            use_condition_channels=use_condition_channels,
            left_robin_conditioning=left_robin_conditioning,
        )
    training = config.get("training", {})
    optimizer = torch.optim.AdamW(
        objective.parameters(), lr=training.get("learning_rate", 1e-4),
        betas=tuple(training.get("adam_betas", (0.9, 0.99))), weight_decay=training.get("weight_decay", 0.01),
    )
    accumulation = max(1, int(training.get("gradient_accumulation_steps", 1)))
    duration = int(training.get("steps", 1))
    updates = max(1, int(scheduler_updates or ((duration + accumulation - 1) // accumulation)))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=float(training.get("minimum_learning_rate", 0.0))
    ) if training.get("scheduler", "cosine") == "cosine" else None
    schedule_cfg = config.get("physics_training", {})
    schedule = PhysicsSchedule(weight=schedule_cfg.get("lambda_phy", 0.0),
                               start_step=schedule_cfg.get("start_step", 0),
                               warmup_steps=schedule_cfg.get("warmup_steps", 0))
    boundary_schedule = PhysicsSchedule(
        weight=schedule_cfg.get("lambda_bc", 0.0),
        start_step=schedule_cfg.get("start_step", 0),
        warmup_steps=schedule_cfg.get("warmup_steps", 0),
    )
    return Experiment(objective=objective, trainer=Trainer(
        objective, optimizer, schedule=schedule, boundary_schedule=boundary_schedule,
        lr_scheduler=lr_scheduler,
        grad_clip=training.get("gradient_clip", 1.0),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 1),
        mixed_precision=training.get("mixed_precision", False),
        ema_decay=training.get("ema_decay"), ema_update_every=training.get("ema_update_every", 10),
    ))
