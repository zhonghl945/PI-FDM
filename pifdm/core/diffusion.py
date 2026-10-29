"""Diffusion utilities with explicit frozen-condition semantics.

The original implementations duplicated this wrapper for every case and
method.  This version keeps conditioning in one place and exposes the same
mask for training, sampling and diagnostics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def extract(values: torch.Tensor, timestep: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Gather a schedule at ``timestep`` and broadcast it to ``shape``."""

    out = values.to(timestep.device).gather(0, timestep.long())
    return out.reshape(timestep.shape[0], *((1,) * (len(shape) - 1))).to(dtype=values.dtype)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0, 0.999)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000.0 / timesteps
    return torch.linspace(scale * 1e-4, scale * 2e-2, timesteps, dtype=torch.float64).clamp(max=0.999)


@dataclass(frozen=True)
class FrozenConditioning:
    """Description of values that are known and must not be noised.

    For a 1D space-time field, the conventional mask freezes ``t=0`` and the
    two spatial boundaries.  For 2D space-time fields it additionally freezes
    the two y boundaries.
    """

    spatial_dims: int = 1
    freeze_spatial_boundaries: bool = True

    def _shape_reference(self, values: torch.Tensor) -> torch.Tensor:
        # A field has batch + channel + time + spatial axes.  Raw datasets
        # omit the channel axis.
        if values.ndim == self.spatial_dims + 3:  # already has channel dim
            return values
        if values.ndim == self.spatial_dims + 2:
            return values.unsqueeze(1)
        raise ValueError(
            f"expected {self.spatial_dims + 2} or {self.spatial_dims + 3} dims, got {values.shape}"
        )

    def initial_mask(self, values: torch.Tensor) -> torch.Tensor:
        x = self._shape_reference(values)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, :, 0, ...] = True
        return mask

    def boundary_mask(self, values: torch.Tensor) -> torch.Tensor:
        x = self._shape_reference(values)
        mask = torch.zeros_like(x, dtype=torch.bool)
        if not self.freeze_spatial_boundaries:
            return mask
        mask[:, :, :, 0] = True
        mask[:, :, :, -1] = True
        if self.spatial_dims == 2:
            mask[:, :, :, 0, :] = True
            mask[:, :, :, -1, :] = True
        return mask

    def mask(self, values: torch.Tensor) -> torch.Tensor:
        return self.initial_mask(values) | self.boundary_mask(values)

    def reconstruction_mask(self, values: torch.Tensor, *, include_initial: bool = True,
                            include_boundaries: bool = True) -> torch.Tensor:
        """Return the field slots included in the supervised loss."""
        reference = self._shape_reference(values)
        supervised = torch.ones_like(reference, dtype=torch.bool)
        if not include_initial:
            # The initial condition applies to E in the interior.  A Robin
            # slot at t=0 is still a boundary datum when boundaries are
            # explicitly supervised.
            supervised &= ~(self.initial_mask(reference) & ~self.boundary_mask(reference))
        if not include_boundaries:
            supervised &= ~self.boundary_mask(reference)
        return supervised

    def apply(self, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        return torch.where(self.mask(clean), clean if clean.ndim == noisy.ndim else clean.unsqueeze(1), noisy)


class GaussianDiffusion(nn.Module):
    """Minimal DDPM forward process shared by FDM, C-FDM and PI-FDM.

    ``predict_x0`` is intentional: all methods compare the network's decoded
    field to the clean field.  Physics is an optional extra term and therefore
    setting ``lambda_phy=0`` gives the *same* C-FDM objective exactly.
    """

    def __init__(self, model: nn.Module, *, solution_shape: Tuple[int, ...], timesteps: int = 1000,
                 beta_schedule: str = "cosine", sampling_timesteps: int | None = None, ddim_eta: float = 0.0,
                 conditioning: Optional[FrozenConditioning] = None, physics_loss_fn=None,
                 boundary_loss_fn=None, reconstruct_boundaries: bool = True,
                 reconstruct_initial: bool = True,
                 material_condition: torch.Tensor | None = None,
                 use_condition_channels: bool = True,
                 left_robin_conditioning: bool = False):
        super().__init__()
        self.model = model
        self.solution_shape = tuple(solution_shape)
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = int(sampling_timesteps or timesteps)
        self.ddim_eta = float(ddim_eta)
        self.conditioning = conditioning or FrozenConditioning(len(solution_shape) - 1)
        self.physics_loss_fn = physics_loss_fn
        self.boundary_loss_fn = boundary_loss_fn
        self.reconstruct_boundaries = bool(reconstruct_boundaries)
        self.reconstruct_initial = bool(reconstruct_initial)
        # The field is always frozen at known slots.  This flag only controls
        # whether the mask/material are exposed as extra network channels.
        self.use_condition_channels = bool(use_condition_channels)
        self.left_robin_conditioning = bool(left_robin_conditioning)
        if material_condition is None:
            self.register_buffer("material_condition", None, persistent=False)
        else:
            material_condition = torch.as_tensor(material_condition).detach().float()
            expected_shape = (1, 1, *self.solution_shape)
            if tuple(material_condition.shape) != expected_shape:
                raise ValueError(
                    f"material_condition must have shape {expected_shape}, "
                    f"got {tuple(material_condition.shape)}"
                )
            self.register_buffer("material_condition", material_condition)
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.num_timesteps).float()
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(self.num_timesteps).float()
        else:
            raise ValueError("beta_schedule must be 'cosine' or 'linear'")
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        for name, value in {
            "betas": betas,
            "alpha_bar": alpha_bar,
            "sqrt_alpha_bar": alpha_bar.sqrt(),
            "sqrt_one_minus_alpha_bar": (1 - alpha_bar).sqrt(),
            "posterior_variance": betas * (1 - alpha_prev) / (1 - alpha_bar),
            "posterior_mean_coef1": betas * alpha_prev.sqrt() / (1 - alpha_bar),
            "posterior_mean_coef2": (1 - alpha_prev) * alphas.sqrt() / (1 - alpha_bar),
        }.items():
            self.register_buffer(name, value)

    def q_sample(self, x_start: torch.Tensor, timestep: torch.Tensor, noise: Optional[torch.Tensor] = None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return (extract(self.sqrt_alpha_bar, timestep, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alpha_bar, timestep, x_start.shape) * noise), noise

    def conditioned_noisy_input(self, x_start: torch.Tensor, timestep: torch.Tensor,
                                noise: Optional[torch.Tensor] = None):
        noisy, noise = self.q_sample(x_start, timestep, noise)
        return self.conditioning.apply(noisy, x_start), noise

    def _left_robin_channel(self, g_left: torch.Tensor | None,
                            field: torch.Tensor) -> torch.Tensor | None:
        """Broadcast saved ``gL(t,y)`` over x without interpolating it."""
        if not self.left_robin_conditioning:
            return None
        if g_left is None:
            raise ValueError("radiative_2d diffusion requires the clean g_left condition")
        if g_left.ndim == 4 and g_left.shape[1] == 1:
            g_left = g_left[:, 0]
        expected = (field.shape[0], field.shape[2], field.shape[3])
        if g_left.ndim != 3 or tuple(g_left.shape) != expected:
            raise ValueError(
                f"g_left must have shape {expected} for field {tuple(field.shape)}, "
                f"got {tuple(g_left.shape)}"
            )
        return g_left.to(device=field.device, dtype=field.dtype).unsqueeze(-1).expand(
            field.shape[0], field.shape[2], field.shape[3], field.shape[4]
        ).unsqueeze(1)

    def _material_input(self, field: torch.Tensor) -> torch.Tensor | None:
        if self.material_condition is None:
            return None
        material = self.material_condition.to(device=field.device, dtype=field.dtype)
        if material.shape[2:] != field.shape[2:]:
            raise ValueError(
                f"material condition shape {tuple(material.shape[2:])} does not match "
                f"field shape {tuple(field.shape[2:])}"
            )
        return material.expand(field.shape[0], -1, *material.shape[2:])

    def model_input(self, field: torch.Tensor, mask: Optional[torch.Tensor] = None,
                    material: Optional[torch.Tensor] = None,
                    g_left: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Build the network input while keeping frozen-condition semantics."""
        mask = self.conditioning.mask(field) if mask is None else mask
        if mask.shape != field.shape:
            raise ValueError(f"condition mask {tuple(mask.shape)} does not match field {tuple(field.shape)}")
        channels = [field]
        left_channel = self._left_robin_channel(g_left, field)
        if left_channel is not None:
            channels.append(left_channel)
        if self.use_condition_channels:
            channels.append(mask.to(dtype=field.dtype))
            material = self._material_input(field) if material is None else material
            if material is not None:
                if material.ndim != field.ndim or material.shape[0] != field.shape[0]:
                    raise ValueError(
                        f"material condition must match field batch and dimensions, got "
                        f"{tuple(material.shape)} for field {tuple(field.shape)}"
                    )
                if material.shape[2:] != field.shape[2:]:
                    raise ValueError(
                        f"material condition shape {tuple(material.shape[2:])} does not match "
                        f"field shape {tuple(field.shape[2:])}"
                    )
                channels.append(material.to(dtype=field.dtype))
        return torch.cat(channels, dim=1)

    def forward(self, x_start: torch.Tensor, *, lambda_phy: float = 0.0,
                lambda_bc: float = 0.0, **kwargs):
        g_left = None
        if isinstance(x_start, Mapping):
            g_left = x_start.get("g_left")
            x_start = x_start.get("field")
            if x_start is None:
                raise ValueError("condition mapping must contain a 'field' tensor")
        if x_start.ndim == len(self.solution_shape) + 1:
            x_start = x_start.unsqueeze(1)
        timestep = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
        noisy_field, _ = self.conditioned_noisy_input(x_start, timestep)
        mask = self.conditioning.mask(x_start)
        model_input = self.model_input(noisy_field, mask, g_left=g_left)
        latent_grid = None
        if getattr(self.model, "continuous", False) and kwargs.get("coords") is None:
            latent_grid = self.model.encode(model_input, timestep)
            pred = self.model.decode(latent_grid, output_shape=self.solution_shape)
        else:
            pred = self.model(model_input, timestep, **kwargs)
        if pred.ndim == x_start.ndim - 1:
            pred = pred.unsqueeze(1)
        if pred.shape != x_start.shape:
            if pred.numel() != x_start.numel():
                raise ValueError(f"model output {tuple(pred.shape)} cannot represent field {tuple(x_start.shape)}")
            pred = pred.reshape_as(x_start)
        snr = (extract(self.alpha_bar, timestep, x_start.shape)
               / extract(1 - self.alpha_bar, timestep, x_start.shape).clamp_min(1e-8)).clamp(max=5.0)
        squared_error = snr * (pred - x_start).square()
        if self.reconstruct_boundaries and self.reconstruct_initial:
            recon = squared_error.mean()
        else:
            supervised = self.conditioning.reconstruction_mask(
                x_start, include_initial=self.reconstruct_initial,
                include_boundaries=self.reconstruct_boundaries,
            )
            recon = squared_error.masked_select(supervised).mean()
        # Physics terms are supplied by a case-specific trainer.  Keeping the
        # zero branch exact makes C-FDM a meaningful ablation of PI-FDM.
        physics = pred.new_zeros(())
        if lambda_phy:
            # Physics-informed continuous models need access to their decoder
            # and the current latent representation, not a detached grid
            # prediction.  The callback receives both and owns collocation.
            physics = self.physics_loss(
                prediction=pred, model=self.model, model_input=model_input,
                timestep=timestep, latent_grid=latent_grid,
                condition_field=x_start, g_left=g_left, **kwargs,
            ) * float(lambda_phy)
        boundary = pred.new_zeros(())
        if lambda_bc:
            if self.boundary_loss_fn is None:
                raise RuntimeError("lambda_bc > 0 but no boundary_loss_fn was provided")
            boundary = self.boundary_loss_fn(
                prediction=pred, model=self.model, model_input=model_input,
                timestep=timestep, latent_grid=latent_grid,
                condition_field=x_start, g_left=g_left, **kwargs,
            ) * float(lambda_bc)
        return {
            "total_loss": recon + physics + boundary,
            "L_recon": recon.detach(), "L_phy": physics.detach(),
            "L_bc": boundary.detach(),
        }

    def physics_loss(self, prediction: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.physics_loss_fn is None:
            raise RuntimeError("lambda_phy > 0 but no physics_loss_fn was provided")
        return self.physics_loss_fn(prediction, **kwargs)

    def _prepare_condition(self, condition, g_left=None):
        if isinstance(condition, Mapping):
            if g_left is None:
                g_left = condition.get("g_left")
            condition = condition.get("field")
            if condition is None:
                raise ValueError("condition mapping must contain a 'field' tensor")
        if condition.ndim == len(self.solution_shape) + 1:
            condition = condition.unsqueeze(1)
        condition = condition.to(self.alpha_bar.device)
        if g_left is not None:
            g_left = g_left.to(device=condition.device, dtype=condition.dtype)
        return condition, self.conditioning.mask(condition), g_left

    def _model_x_start(self, image, timestep, mask, g_left=None):
        """Run a grid or continuous decoder and restore the field layout."""
        prediction = self.model(self.model_input(image, mask, g_left=g_left), timestep)
        if prediction.ndim == image.ndim - 1:
            prediction = prediction.unsqueeze(1)
        if prediction.numel() != image.numel():
            raise ValueError(f"model output {tuple(prediction.shape)} cannot represent field {tuple(image.shape)}")
        return prediction.reshape_as(image).clamp(-1, 1)

    def _apply_grid_guidance(self, x_start: torch.Tensor, condition: torch.Tensor,
                             mask: torch.Tensor, *, guidance_weight: float,
                             physics_loss_fn=None, boundary_loss_fn=None,
                             g_left: torch.Tensor | None = None) -> torch.Tensor:
        """One fixed-strength differentiable physics correction for sampling."""
        if guidance_weight == 0.0 or (physics_loss_fn is None and boundary_loss_fn is None):
            return x_start
        with torch.enable_grad():
            guided = x_start.detach().requires_grad_(True)
            loss = guided.new_zeros(())
            if physics_loss_fn is not None:
                loss = loss + physics_loss_fn(
                    prediction=guided, model_input=condition,
                    condition_field=condition, g_left=g_left,
                )
            if boundary_loss_fn is not None:
                loss = loss + boundary_loss_fn(
                    prediction=guided, model_input=condition,
                    condition_field=condition, g_left=g_left,
                )
            # Grid losses average over both points and batch items. Remove only
            # the batch averaging so guidance is independent of test batch size.
            gradient = torch.autograd.grad(
                loss * guided.shape[0], guided, allow_unused=False
            )[0]
            guided = guided - float(guidance_weight) * gradient
        lower = getattr(self.model, "output_floor", None)
        lower = -1.0 if lower is None else max(-1.0, float(lower))
        return torch.where(mask, condition, guided.detach()).clamp(lower, 1.0)

    @torch.no_grad()
    def ddpm_sample(self, condition: torch.Tensor, *, guidance_weight: float = 0.0,
                    guidance_physics_loss_fn=None, guidance_boundary_loss_fn=None,
                    g_left: torch.Tensor | None = None) -> torch.Tensor:
        """Generate a conditional field with the full DDPM reverse process."""
        condition, mask, g_left = self._prepare_condition(condition, g_left)
        image = torch.randn_like(condition)
        for index in reversed(range(self.num_timesteps)):
            timestep = torch.full((condition.shape[0],), index, device=condition.device, dtype=torch.long)
            image = torch.where(mask, condition, image)
            x_start = self._model_x_start(image, timestep, mask, g_left)
            x_start = self._apply_grid_guidance(
                x_start, condition, mask, guidance_weight=guidance_weight,
                physics_loss_fn=guidance_physics_loss_fn,
                boundary_loss_fn=guidance_boundary_loss_fn, g_left=g_left,
            )
            x_start = torch.where(mask, condition, x_start)
            mean = (extract(self.posterior_mean_coef1, timestep, image.shape) * x_start
                    + extract(self.posterior_mean_coef2, timestep, image.shape) * image)
            if index:
                std = extract(self.posterior_variance, timestep, image.shape).clamp_min(1e-20).sqrt()
                image = mean + std * torch.randn_like(image)
            else:
                image = mean
        return torch.where(mask, condition, image)

    @torch.no_grad()
    def ddim_sample(self, condition: torch.Tensor, *, sampling_timesteps: int | None = None,
                    eta: float | None = None, guidance_weight: float = 0.0,
                    guidance_physics_loss_fn=None, guidance_boundary_loss_fn=None,
                    g_left: torch.Tensor | None = None) -> torch.Tensor:
        """Conditional DDIM sampling with configurable stochasticity."""
        condition, mask, g_left = self._prepare_condition(condition, g_left)
        steps = int(sampling_timesteps or self.sampling_timesteps)
        if not 1 <= steps <= self.num_timesteps:
            raise ValueError("sampling_timesteps must be between 1 and diffusion timesteps")
        eta = self.ddim_eta if eta is None else float(eta)
        times = torch.linspace(-1, self.num_timesteps - 1, steps + 1, device=condition.device).long().flip(0)
        image = torch.randn_like(condition)
        for current, following in zip(times[:-1], times[1:]):
            image = torch.where(mask, condition, image)
            timestep = torch.full((condition.shape[0],), int(current), device=condition.device, dtype=torch.long)
            x_start = self._model_x_start(image, timestep, mask, g_left)
            x_start = self._apply_grid_guidance(
                x_start, condition, mask, guidance_weight=guidance_weight,
                physics_loss_fn=guidance_physics_loss_fn,
                boundary_loss_fn=guidance_boundary_loss_fn, g_left=g_left,
            )
            x_start = torch.where(mask, condition, x_start)
            following_index = int(following)
            if following_index < 0:
                image = x_start
                continue
            alpha = self.alpha_bar[current]
            alpha_next = self.alpha_bar[following_index]
            predicted_noise = (image - alpha.sqrt() * x_start) / (1 - alpha).sqrt().clamp_min(1e-8)
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).clamp_min(0).sqrt()
            coefficient = (1 - alpha_next - sigma.square()).clamp_min(0).sqrt()
            image = alpha_next.sqrt() * x_start + coefficient * predicted_noise + sigma * torch.randn_like(image)
        return torch.where(mask, condition, image)

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, *, sampler: str = "auto", sampling_timesteps: int | None = None,
               eta: float | None = None, guidance_weight: float = 0.0,
               guidance_physics_loss_fn=None, guidance_boundary_loss_fn=None,
               g_left: torch.Tensor | None = None) -> torch.Tensor:
        steps = int(sampling_timesteps or self.sampling_timesteps)
        if sampler == "auto":
            sampler = "ddim" if steps < self.num_timesteps else "ddpm"
        if sampler == "ddpm":
            return self.ddpm_sample(
                condition, guidance_weight=guidance_weight,
                guidance_physics_loss_fn=guidance_physics_loss_fn,
                guidance_boundary_loss_fn=guidance_boundary_loss_fn, g_left=g_left,
            )
        if sampler == "ddim":
            return self.ddim_sample(
                condition, sampling_timesteps=steps, eta=eta,
                guidance_weight=guidance_weight,
                guidance_physics_loss_fn=guidance_physics_loss_fn,
                guidance_boundary_loss_fn=guidance_boundary_loss_fn, g_left=g_left,
            )
        raise ValueError("sampler must be auto, ddpm or ddim")

    predict = sample
