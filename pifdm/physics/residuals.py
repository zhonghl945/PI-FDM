"""Differentiable PDE residuals.

All functions take a callable ``field(coords)`` where ``coords`` is normalized
and ordered as ``(t, x)`` or ``(t, y, x)``.  A full Jacobian is intentionally
not formed: ``autograd.grad(..., ones_like(field))`` is valid here because each
query is independent in the decoder and is substantially cheaper.
"""

from __future__ import annotations

import torch
from torch import autograd


def _first_derivative(value: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    return autograd.grad(value, coords, torch.ones_like(value), create_graph=True, retain_graph=True)[0]


def _material_at(material, coords: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
    value = material(coords) if callable(material) else material
    value = torch.as_tensor(value, dtype=energy.dtype, device=energy.device)
    if value.ndim == 0:
        value = value.reshape(1, 1, 1)
    elif value.ndim == 1:
        value = value.reshape(1, -1, 1)
    elif value.ndim == 2:
        value = value.unsqueeze(-1)
    while value.ndim < energy.ndim:
        value = value.unsqueeze(0)
    return value


def _radiative_diffusivity(energy: torch.Tensor, e_x: torch.Tensor,
                            e_y: torch.Tensor | None, material: torch.Tensor,
                            *, use_flux_limiter: bool, limiter_norm: str,
                            eps: float) -> torch.Tensor:
    safe = energy.clamp_min(eps)
    if not use_flux_limiter:
        return safe.pow(0.75) / (3.0 * material.pow(3)).clamp_min(eps)
    if limiter_norm == "l2_smooth":
        gradient_norm = torch.sqrt(
            e_x.square() + (e_y.square() if e_y is not None else 0.0) + eps
        )
    elif limiter_norm == "l1":
        gradient_norm = e_x.abs() + (e_y.abs() if e_y is not None else 0.0)
    else:
        raise ValueError("limiter_norm must be 'l1' or 'l2_smooth'")
    denominator = 3.0 * material.pow(3) * safe.pow(0.25) + gradient_norm
    return safe / denominator.clamp_min(eps)


def linear_diffusion_residual(field, coords: torch.Tensor, *, dt_scale: float, dx_scale: float,
                              diffusivity: float = 1.0, source=None) -> torch.Tensor:
    """Residual of ``u_t - D u_xx - f = 0`` at arbitrary collocation points."""

    u = field(coords)
    grad = _first_derivative(u, coords)
    u_t = grad[..., 0:1] * dt_scale
    u_x = grad[..., 1:2] * dx_scale
    grad_x = _first_derivative(u_x, coords)
    u_xx = grad_x[..., 1:2] * dx_scale
    residual = u_t - diffusivity * u_xx
    if source is not None:
        residual = residual - (source(coords) if callable(source) else
                               torch.as_tensor(source, dtype=residual.dtype, device=residual.device))
    return residual


def radiative_diffusion_residual(field, coords: torch.Tensor, *, dt_scale: float, dx_scale: float,
                                 dy_scale: float | None = None, material: torch.Tensor | float = 1.0,
                                 source=None, use_flux_limiter: bool = True,
                                 limiter_norm: str = "l2_smooth", eps: float = 1e-8) -> torch.Tensor:
    """Residual of ``E_t + div(-D grad E) - S = 0`` in 1D or 2D space.

    ``coords`` is ordered ``(t, x)`` in 1D and ``(t, y, x)`` in 2D.  ``material`` is the opacity/material coefficient ``z``.  It is kept as an
    input rather than reconstructed from a hard-coded mask, so training and
    evaluation use exactly the same heterogeneous field.  ``source`` may be a
    callable of physical coordinates or a tensor already aligned to queries.
    """

    energy = field(coords)
    grad = _first_derivative(energy, coords)
    e_t = grad[..., 0:1] * dt_scale
    if coords.shape[-1] == 2:
        e_x = grad[..., 1:2] * dx_scale
        e_y = None
    elif coords.shape[-1] == 3:
        if dy_scale is None:
            raise ValueError("dy_scale is required for 2D radiative residuals")
        e_y = grad[..., 1:2] * dy_scale
        e_x = grad[..., 2:3] * dx_scale
    else:
        raise ValueError("radiative residual expects (t,x) or (t,y,x) coordinates")
    mat = _material_at(material, coords, energy)
    diffusivity = _radiative_diffusivity(
        energy, e_x, e_y, mat, use_flux_limiter=use_flux_limiter,
        limiter_norm=limiter_norm, eps=eps,
    )
    flux_x = -diffusivity * e_x
    flux_grad = _first_derivative(flux_x, coords)
    div_flux = flux_grad[..., -1:] * dx_scale
    if e_y is not None:
        flux_y = -diffusivity * e_y
        flux_y_grad = _first_derivative(flux_y, coords)
        div_flux = div_flux + flux_y_grad[..., 1:2] * dy_scale
    residual = e_t + div_flux
    if source is not None:
        if callable(source):
            residual = residual - source(coords)
        else:
            residual = residual - torch.as_tensor(source, dtype=residual.dtype, device=residual.device)
    return residual


def radiative_robin_residual(field, coords: torch.Tensor, outward_normal: torch.Tensor,
                             boundary_rhs: torch.Tensor, *, dx_scale: float,
                             dy_scale: float | None = None,
                             material: torch.Tensor | float = 1.0,
                             use_flux_limiter: bool = True,
                             limiter_norm: str = "l2_smooth",
                             eps: float = 1e-8) -> torch.Tensor:
    """Residual of ``0.5 E + D grad(E).n = g`` on a radiative boundary."""
    energy = field(coords)
    gradient = _first_derivative(energy, coords)
    if coords.shape[-1] == 2:
        e_y = None
        e_x = gradient[..., 1:2] * dx_scale
        if outward_normal.shape[-1] != 1:
            raise ValueError("1D outward_normal must contain one spatial component")
        normal_derivative = e_x * outward_normal[..., 0:1]
    elif coords.shape[-1] == 3:
        if dy_scale is None or outward_normal.shape[-1] != 2:
            raise ValueError("2D Robin residual requires dy_scale and a (y,x) normal")
        e_y = gradient[..., 1:2] * dy_scale
        e_x = gradient[..., 2:3] * dx_scale
        normal_derivative = (
            e_y * outward_normal[..., 0:1] + e_x * outward_normal[..., 1:2]
        )
    else:
        raise ValueError("radiative Robin residual expects (t,x) or (t,y,x) coordinates")
    mat = _material_at(material, coords, energy)
    diffusivity = _radiative_diffusivity(
        energy, e_x, e_y, mat, use_flux_limiter=use_flux_limiter,
        limiter_norm=limiter_norm, eps=eps,
    )
    rhs = torch.as_tensor(boundary_rhs, dtype=energy.dtype, device=energy.device)
    return 0.5 * energy + diffusivity * normal_derivative - rhs
