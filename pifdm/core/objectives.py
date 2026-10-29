"""Training objectives that are not diffusion models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Tuple

import torch
from torch import nn

from .diffusion import FrozenConditioning


class OperatorObjective(nn.Module):
    """Supervised/physics-informed objective for DeepONet variants."""

    def __init__(self, model: nn.Module, *, solution_shape: Tuple[int, ...], physics_loss_fn=None,
                 boundary_loss_fn=None, reconstruct_boundaries: bool = True,
                 reconstruct_initial: bool = True, left_robin_conditioning: bool = False):
        super().__init__()
        self.model = model
        self.solution_shape = tuple(solution_shape)
        self.physics_loss_fn = physics_loss_fn
        self.boundary_loss_fn = boundary_loss_fn
        self.reconstruct_boundaries = bool(reconstruct_boundaries)
        self.reconstruct_initial = bool(reconstruct_initial)
        self.left_robin_conditioning = bool(left_robin_conditioning)
        self.conditioning = FrozenConditioning(
            len(self.solution_shape) - 1,
            freeze_spatial_boundaries=not self.left_robin_conditioning,
        )

    def _with_channel(self, field: torch.Tensor) -> torch.Tensor:
        return field.unsqueeze(1) if field.ndim == len(self.solution_shape) + 1 else field

    @staticmethod
    def _unpack_condition(condition):
        if isinstance(condition, Mapping):
            field = condition.get("field")
            if field is None:
                raise ValueError("condition mapping must contain a 'field' tensor")
            return field, condition.get("g_left")
        return condition, None

    def _raw_prediction(self, condition: torch.Tensor, g_left=None) -> torch.Tensor:
        condition = self._with_channel(condition)
        if self.left_robin_conditioning:
            prediction = self.model(condition, None, g_left=g_left)
        else:
            prediction = self.model(condition, None)
        return prediction.reshape_as(condition)

    def predict(self, condition: torch.Tensor) -> torch.Tensor:
        condition, g_left = self._unpack_condition(condition)
        condition = self._with_channel(condition)
        prediction = self._raw_prediction(condition, g_left)
        # The packed grid carries E(t=0) and Robin g as clean conditions.
        # Preserve both on the returned grid, while forward() below keeps the
        # raw E prediction for physics and boundary-residual losses.
        return torch.where(self.conditioning.mask(condition), condition, prediction)

    def forward(self, target: torch.Tensor, *, lambda_phy: float = 0.0,
                lambda_bc: float = 0.0, **kwargs):
        target, g_left = self._unpack_condition(target)
        target = self._with_channel(target)
        prediction = self._raw_prediction(target, g_left)
        squared_error = (prediction - target).square()
        if self.reconstruct_boundaries and self.reconstruct_initial:
            reconstruction = squared_error.mean()
        else:
            supervised = self.conditioning.reconstruction_mask(
                target, include_initial=self.reconstruct_initial,
                include_boundaries=self.reconstruct_boundaries,
            )
            reconstruction = squared_error.masked_select(supervised).mean()
        physics = prediction.new_zeros(())
        if lambda_phy:
            if self.physics_loss_fn is None:
                raise RuntimeError("lambda_phy > 0 but no physics_loss_fn was provided")
            physics = self.physics_loss_fn(
                prediction=prediction, model=self.model, model_input=target,
                timestep=None, g_left=g_left, condition_field=target, **kwargs
            ) * float(lambda_phy)
        boundary = prediction.new_zeros(())
        if lambda_bc:
            if self.boundary_loss_fn is None:
                raise RuntimeError("lambda_bc > 0 but no boundary_loss_fn was provided")
            boundary = self.boundary_loss_fn(
                prediction=prediction, model=self.model, model_input=target,
                timestep=None, g_left=g_left, condition_field=target, **kwargs,
            ) * float(lambda_bc)
        return {
            "total_loss": reconstruction + physics + boundary,
            "L_recon": reconstruction.detach(), "L_phy": physics.detach(),
            "L_bc": boundary.detach(),
        }


class GridOperatorObjective(nn.Module):
    """Leak-free FNO/PINO objective over a packed regular space-time grid."""

    def __init__(self, model: nn.Module, *, solution_shape: Tuple[int, ...],
                 physics_loss_fn=None, boundary_loss_fn=None,
                 reconstruct_boundaries: bool = True,
                 reconstruct_initial: bool = True,
                 material_condition: torch.Tensor | None = None,
                 condition_fill: float = -1.0,
                 left_robin_conditioning: bool = False):
        super().__init__()
        self.model = model
        self.solution_shape = tuple(solution_shape)
        self.physics_loss_fn = physics_loss_fn
        self.boundary_loss_fn = boundary_loss_fn
        self.reconstruct_boundaries = bool(reconstruct_boundaries)
        self.reconstruct_initial = bool(reconstruct_initial)
        self.left_robin_conditioning = bool(left_robin_conditioning)
        self.conditioning = FrozenConditioning(
            len(self.solution_shape) - 1,
            freeze_spatial_boundaries=not self.left_robin_conditioning,
        )
        self.condition_fill = float(condition_fill)
        if material_condition is None:
            self.register_buffer("material_condition", None, persistent=False)
        else:
            material = torch.as_tensor(material_condition).detach().float()
            expected = (1, 1, *self.solution_shape)
            if tuple(material.shape) != expected:
                raise ValueError(
                    f"material_condition must have shape {expected}, got {tuple(material.shape)}"
                )
            self.register_buffer("material_condition", material, persistent=False)

    def _with_channel(self, field: torch.Tensor) -> torch.Tensor:
        return field.unsqueeze(1) if field.ndim == len(self.solution_shape) + 1 else field

    def _left_robin_channel(self, g_left: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if g_left is None:
            raise ValueError("radiative_2d grid operators require the clean g_left condition")
        if g_left.ndim == 4 and g_left.shape[1] == 1:
            g_left = g_left[:, 0]
        expected = (condition.shape[0], condition.shape[2], condition.shape[3])
        if g_left.ndim != 3 or tuple(g_left.shape) != expected:
            raise ValueError(
                f"g_left must have shape {expected} for condition {tuple(condition.shape)}, "
                f"got {tuple(g_left.shape)}"
            )
        return g_left.to(condition).unsqueeze(-1).expand(
            condition.shape[0], condition.shape[2], condition.shape[3], condition.shape[4]
        ).unsqueeze(1)

    def _model_input(self, condition: torch.Tensor, g_left=None) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.conditioning.mask(condition)
        known = torch.where(mask, condition, torch.full_like(condition, self.condition_fill))
        channels = [known]
        if self.left_robin_conditioning:
            channels.append(self._left_robin_channel(g_left, condition))
        channels.append(mask.to(condition.dtype))
        if self.material_condition is not None:
            material = self.material_condition.to(condition).expand(
                condition.shape[0], -1, *self.material_condition.shape[2:]
            )
            channels.append(material)
        return torch.cat(channels, dim=1), mask

    def _raw_prediction(self, condition: torch.Tensor, g_left=None):
        condition = self._with_channel(condition)
        model_input, mask = self._model_input(condition, g_left)
        prediction = self.model(model_input, None)
        if prediction.numel() != condition.numel():
            raise ValueError(
                f"model output {tuple(prediction.shape)} cannot represent {tuple(condition.shape)}"
            )
        return prediction.reshape_as(condition), model_input, mask

    def predict(self, condition: torch.Tensor) -> torch.Tensor:
        condition, g_left = OperatorObjective._unpack_condition(condition)
        condition = self._with_channel(condition)
        prediction, _, mask = self._raw_prediction(condition, g_left)
        return torch.where(mask, condition, prediction)

    def forward(self, target: torch.Tensor, *, lambda_phy: float = 0.0,
                lambda_bc: float = 0.0, **kwargs):
        target, g_left = OperatorObjective._unpack_condition(target)
        target = self._with_channel(target)
        prediction, model_input, _ = self._raw_prediction(target, g_left)
        squared_error = (prediction - target).square()
        if self.reconstruct_boundaries and self.reconstruct_initial:
            reconstruction = squared_error.mean()
        else:
            supervised = self.conditioning.reconstruction_mask(
                target, include_initial=self.reconstruct_initial,
                include_boundaries=self.reconstruct_boundaries,
            )
            reconstruction = squared_error.masked_select(supervised).mean()
        physics = prediction.new_zeros(())
        if lambda_phy:
            if self.physics_loss_fn is None:
                raise RuntimeError("lambda_phy > 0 but no grid physics loss was provided")
            physics = self.physics_loss_fn(
                prediction=prediction, model_input=model_input, model=self.model,
                condition_field=target, g_left=g_left, **kwargs
            ) * float(lambda_phy)
        boundary = prediction.new_zeros(())
        if lambda_bc:
            if self.boundary_loss_fn is None:
                raise RuntimeError("lambda_bc > 0 but no grid boundary loss was provided")
            boundary = self.boundary_loss_fn(
                prediction=prediction, model_input=model_input, model=self.model,
                condition_field=target, g_left=g_left, **kwargs
            ) * float(lambda_bc)
        return {
            "total_loss": reconstruction + physics + boundary,
            "L_recon": reconstruction.detach(), "L_phy": physics.detach(),
            "L_bc": boundary.detach(),
        }
