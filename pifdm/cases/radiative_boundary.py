"""Nonlinear Robin-face closure shared by radiative finite-volume solvers."""

from __future__ import annotations

import numpy as np


def _radiative_diffusivity(
    energy, normal_gradient, tangential_gradient, material, *,
    use_flux_limiter: bool, smooth_limiter: bool,
):
    safe = np.maximum(np.asarray(energy, dtype=float), 1e-12)
    material = np.asarray(material, dtype=float)
    if use_flux_limiter:
        if smooth_limiter:
            gradient_norm = np.sqrt(
                np.asarray(normal_gradient) ** 2
                + np.asarray(tangential_gradient) ** 2
                + 1e-15
            )
        else:
            gradient_norm = (
                np.abs(np.asarray(normal_gradient))
                + np.abs(np.asarray(tangential_gradient))
            )
        return safe / (
            3.0 * material**3 * safe**0.25 + gradient_norm
        ).clip(min=1e-15)
    return safe**0.75 / (3.0 * material**3).clip(min=1e-15)


def solve_radiative_robin_face(
    cell_energy, spacing: float, boundary_rhs, material, *,
    tangential_gradient=0.0, use_flux_limiter: bool = True,
    smooth_limiter: bool = True, iterations: int = 40,
):
    """Solve a face state and diffusivity consistent with the Robin operator.

    A cell-centred ghost construction represents the boundary value by the
    average of the adjacent cell and ghost values and the outward derivative
    by their centred difference.  Eliminating the ghost gives

    ``normal_gradient = 2 * (face_energy - cell_energy) / spacing``.

    Both approximations are second-order at the geometric boundary.  The
    nonlinear scalar equation

    ``0.5 * face_energy + D(face_energy, grad E) * normal_gradient = rhs``

    is solved by bracketed bisection, keeping the face diffusivity positive.
    """
    spacing = np.asarray(spacing, dtype=float)
    if np.any(spacing <= 0.0):
        raise ValueError("boundary spacing must be positive")
    cell, spacing, rhs, mat, tangent = np.broadcast_arrays(
        np.asarray(cell_energy, dtype=float),
        spacing,
        np.asarray(boundary_rhs, dtype=float),
        np.asarray(material, dtype=float),
        np.asarray(tangential_gradient, dtype=float),
    )
    if np.any(cell < 0.0) or np.any(rhs < 0.0) or np.any(mat <= 0.0):
        raise ValueError("radiative Robin closure requires nonnegative energy/rhs and positive material")

    def residual(face):
        normal = 2.0 * (face - cell) / spacing
        diffusion = _radiative_diffusivity(
            face, normal, tangent, mat,
            use_flux_limiter=use_flux_limiter,
            smooth_limiter=smooth_limiter,
        )
        return 0.5 * face + diffusion * normal - rhs

    lower = np.zeros_like(cell)
    upper = np.maximum.reduce((cell, 2.0 * rhs, np.full_like(cell, 1e-6)))
    upper_residual = residual(upper)
    for _ in range(64):
        unresolved = upper_residual < 0.0
        if not np.any(unresolved):
            break
        upper = np.where(unresolved, 2.0 * upper + 1.0, upper)
        upper_residual = residual(upper)
    if np.any(upper_residual < 0.0):
        raise RuntimeError("failed to bracket nonlinear radiative Robin face state")

    for _ in range(max(1, int(iterations))):
        midpoint = 0.5 * (lower + upper)
        below = residual(midpoint) <= 0.0
        lower = np.where(below, midpoint, lower)
        upper = np.where(below, upper, midpoint)
    face = 0.5 * (lower + upper)
    normal = 2.0 * (face - cell) / spacing
    diffusion = _radiative_diffusivity(
        face, normal, tangent, mat,
        use_flux_limiter=use_flux_limiter,
        smooth_limiter=smooth_limiter,
    )
    return face, diffusion


def solve_radiative_robin_side(
    cell_energy, normal_spacing: float, tangent_spacing: float, boundary_rhs, material, *,
    initial_tangential_gradient=None, use_flux_limiter: bool = True,
    smooth_limiter: bool = True,
):
    """Close one 2D boundary side using a second-order tangential estimate.

    The caller extrapolates the tangential cell gradient to the geometric
    boundary.  Solving each face once keeps this closure inexpensive inside
    the outer nonlinear Picard iteration.
    """
    cell = np.asarray(cell_energy, dtype=float)
    if cell.ndim != 1:
        raise ValueError("a 2D boundary side must be represented by one-dimensional arrays")
    if tangent_spacing <= 0.0:
        raise ValueError("tangential spacing must be positive")
    if initial_tangential_gradient is None:
        tangent = np.zeros_like(cell)
    else:
        tangent = np.asarray(initial_tangential_gradient, dtype=float)
    return solve_radiative_robin_face(
        cell, normal_spacing, boundary_rhs, material,
        tangential_gradient=tangent,
        use_flux_limiter=use_flux_limiter,
        smooth_limiter=smooth_limiter,
    )
