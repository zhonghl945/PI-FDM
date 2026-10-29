"""Implicit finite-volume solver for heterogeneous 2D radiative diffusion."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .base import CaseAdapter, CaseData, register_case, residual_statistics
from .acceleration import (independent_seeds, njit, parallel_samples,
                           report_solver_progress)
from .radiative_boundary import solve_radiative_robin_face


def build_material(x: np.ndarray, y: np.ndarray, spec: dict) -> np.ndarray:
    xx, yy = np.meshgrid(x, y)
    material = np.full_like(xx, float(spec.get("value", spec.get("background", 1.0))), dtype=float)
    kind = spec.get("type", "constant")
    if kind not in ("constant", "rectangles"):
        raise ValueError(f"radiative_2d generation does not support material type {kind!r}")
    for region in spec.get("regions", []):
        x0, x1 = region["x"]
        y0, y1 = region["y"]
        mask = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        material[mask] = float(region["value"])
    return material


def build_source(t: np.ndarray, x: np.ndarray, y: np.ndarray, spec: dict) -> np.ndarray:
    kind = spec.get("type", "constant")
    if kind == "constant":
        return np.full((len(t), len(y), len(x)), float(spec.get("value", 0.0)))
    raise ValueError(f"radiative_2d generation does not support source type {kind!r}")


def _sample_parameter(value, rng: np.random.Generator, name: str) -> float:
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        return float(values)
    if values.shape != (2,) or values[1] < values[0]:
        raise ValueError(f"{name} must be a scalar or [minimum, maximum]")
    return float(rng.uniform(values[0], values[1]))


def build_initial_energy(x: np.ndarray, y: np.ndarray, spec: dict,
                         rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    """Build one constant or randomly parameterized Gaussian initial-energy field."""
    kind = spec.get("type", "constant")
    background = float(spec.get("background", spec.get("value", 1e-3)))
    if background < 0.0:
        raise ValueError("initial energy background must be non-negative")
    if kind == "constant":
        return np.full((len(y), len(x)), background), {
            "amplitude": 0.0, "sigma": 0.0, "center_x": np.nan, "center_y": np.nan,
        }
    if kind != "gaussian_hotspot":
        raise ValueError(f"unknown radiative_2d initial condition type {kind!r}")

    amplitude = _sample_parameter(spec.get("amplitude", [50.0, 150.0]), rng, "amplitude")
    sigma = _sample_parameter(spec.get("sigma", [0.04, 0.09]), rng, "sigma")
    center_x = _sample_parameter(spec.get("center_x", [0.03, 0.15]), rng, "center_x")
    center_y = _sample_parameter(spec.get("center_y", [0.03, 0.15]), rng, "center_y")
    if amplitude < 0.0 or sigma <= 0.0:
        raise ValueError("Gaussian amplitude must be non-negative and sigma must be positive")
    xx, yy = np.meshgrid(x, y)
    radius_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2
    energy = background + amplitude * np.exp(-0.5 * radius_squared / sigma**2)
    return energy, {
        "amplitude": amplitude, "sigma": sigma, "center_x": center_x, "center_y": center_y,
    }


def evaluate_left_boundary(
    t: np.ndarray,
    y: np.ndarray,
    *,
    amplitude: float,
    center: float,
    radius: float,
    mode: str = "sustained",
    rise: float | None = None,
    width: float | None = None,
    pulse_center: float | None = None,
    pulse_width: float | None = None,
) -> np.ndarray:
    """Evaluate the same left Robin boundary family used by data generation.

    The explicit parameters make it possible for a benchmark to choose fixed,
    unseen parameter combinations without duplicating the training formula.
    ``generate_left_boundary`` below remains the stochastic training wrapper.
    """
    if amplitude < 0.0 or radius <= 0.0:
        raise ValueError("left boundary amplitude must be non-negative and radius positive")
    duration = max(float(t[-1] - t[0]), np.finfo(float).eps)
    tau = (t - t[0]) / duration
    y_span = max(float(y[-1] - y[0]), np.finfo(float).eps)
    eta = (y - y[0]) / y_span
    spatial = np.exp(-0.5 * ((eta - center) / radius) ** 2)
    if mode == "sustained":
        if rise is None or width is None or width <= 0.0:
            raise ValueError("sustained boundary requires positive rise and width")
        temporal = 0.5 * (1.0 + np.tanh((tau - rise) / width))
    elif mode == "broad_pulse":
        if pulse_center is None or pulse_width is None or pulse_width <= 0.0:
            raise ValueError("broad_pulse boundary requires positive pulse_center and pulse_width")
        temporal = np.exp(-0.5 * ((tau - pulse_center) / pulse_width) ** 2)
    elif mode == "constant":
        temporal = np.ones_like(tau)
    else:
        raise ValueError(f"unknown 2D boundary mode {mode!r}")
    return np.maximum(amplitude * temporal[:, None] * spatial[None, :], 1e-8)


def generate_left_boundary(t: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                           mode: str = "sustained") -> np.ndarray:
    """Generate one stochastic left Robin boundary for the training dataset."""
    amplitude = rng.uniform(100.0, 300.0)
    center, radius = rng.uniform(0.3, 0.7), rng.uniform(0.2, 0.4)
    if mode == "sustained":
        return evaluate_left_boundary(
            t, y, amplitude=amplitude, center=center, radius=radius, mode=mode,
            rise=rng.uniform(0.05, 0.2), width=rng.uniform(0.05, 0.15),
        )
    if mode == "broad_pulse":
        return evaluate_left_boundary(
            t, y, amplitude=amplitude, center=center, radius=radius, mode=mode,
            pulse_center=rng.uniform(0.1, 0.3), pulse_width=rng.uniform(0.3, 0.6),
        )
    return evaluate_left_boundary(
        t, y, amplitude=amplitude, center=center, radius=radius, mode=mode,
    )


def generate_legacy_left_boundary(t: np.ndarray, sample_count: int | np.ndarray,
                                  rng: np.random.Generator,
                                  mode: str = "sustained") -> np.ndarray:
    """Reproduce the archived boundary sampler used by the original 2D code.

    The archived implementation sampled the spatial profile on ``linspace(0, 1, ny)``
    rather than on cell centres and interpreted the temporal width in physical time.
    Keep this path opt-in so the initial-condition case retains its current generator.
    """
    duration = float(t[-1] - t[0])
    amplitude = rng.uniform(100.0, 300.0)
    center = rng.uniform(0.3, 0.7)
    radius = rng.uniform(0.2, 0.4)
    rise = width = pulse_center = pulse_width = None
    if mode == "sustained":
        rise = rng.uniform(0.05 * duration, 0.2 * duration)
        width = rng.uniform(0.05, 0.15)
    elif mode == "broad_pulse":
        pulse_center = rng.uniform(0.1 * duration, 0.3 * duration)
        pulse_width = rng.uniform(0.3, 0.6)
    return evaluate_legacy_left_boundary(
        t, sample_count, amplitude=amplitude, center=center, radius=radius, mode=mode,
        rise=rise, width=width, pulse_center=pulse_center, pulse_width=pulse_width,
    )


def evaluate_legacy_left_boundary(
    t: np.ndarray, sample_count: int | np.ndarray, *, amplitude: float, center: float,
    radius: float, mode: str = "sustained", rise: float | None = None,
    width: float | None = None, pulse_center: float | None = None,
    pulse_width: float | None = None,
) -> np.ndarray:
    """Evaluate a fixed boundary with the archived physical-time convention."""
    t_array = np.linspace(0.0, float(t[-1] - t[0]), len(t))
    sample_array = np.asarray(sample_count)
    if sample_array.ndim == 0:
        y_array = np.linspace(0.0, 1.0, int(sample_array))
    else:
        y_array = np.asarray(sample_array, dtype=float)
        if y_array.ndim != 1 or y_array.size < 1 or not np.isfinite(y_array).all():
            raise ValueError("legacy boundary coordinates must be a finite 1D array")
    spatial = np.exp(-0.5 * ((y_array - center) / radius) ** 2)
    if mode == "sustained":
        if rise is None or width is None or width <= 0.0:
            raise ValueError("legacy sustained boundary requires rise and width")
        temporal = 0.5 * (1.0 + np.tanh((t_array - rise) / width))
    elif mode == "broad_pulse":
        if pulse_center is None or pulse_width is None or pulse_width <= 0.0:
            raise ValueError("legacy broad_pulse boundary requires pulse center and width")
        temporal = np.exp(-0.5 * ((t_array - pulse_center) / pulse_width) ** 2)
    else:
        temporal = np.ones_like(t_array)
    return np.maximum(amplitude * temporal[:, None] * spatial[None, :], 1e-8)


@njit
def rbgs_sor_solve(
    west: np.ndarray,
    east: np.ndarray,
    south: np.ndarray,
    north: np.ndarray,
    diagonal: np.ndarray,
    rhs: np.ndarray,
    initial: np.ndarray,
    *,
    tolerance: float = 1e-7,
    max_iterations: int = 2000,
    omega: float = 1.9,
) -> np.ndarray:
    """Solve the five-point system with red-black SOR."""
    # ``initial`` is already float64 at the Python boundary.  A plain copy is
    # also compatible with older Numba versions that reject
    # ``np.asarray(array, dtype=float)`` in nopython mode.
    solution = initial.copy()
    ny, nx = solution.shape
    for _ in range(max_iterations):
        max_change = 0.0
        for colour in (0, 1):
            for i in range(ny):
                for j in range((colour - i) & 1, nx, 2):
                    value = rhs[i, j]
                    if j > 0:
                        value += west[i, j] * solution[i, j - 1]
                    if j + 1 < nx:
                        value += east[i, j] * solution[i, j + 1]
                    if i > 0:
                        value += south[i, j] * solution[i - 1, j]
                    if i + 1 < ny:
                        value += north[i, j] * solution[i + 1, j]
                    candidate = (1.0 - omega) * solution[i, j] + omega * value / diagonal[i, j]
                    max_change = max(max_change, abs(candidate - solution[i, j]))
                    solution[i, j] = candidate
        if max_change < tolerance:
            return solution
    raise RuntimeError("radiative_2d SOR linear solve failed to converge")


@njit
def pcg_solve(
    west: np.ndarray,
    east: np.ndarray,
    south: np.ndarray,
    north: np.ndarray,
    diagonal: np.ndarray,
    rhs: np.ndarray,
    initial: np.ndarray,
    *,
    tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-12,
    max_iterations: int = 500,
) -> np.ndarray:
    """Solve the symmetric five-point system with Jacobi-preconditioned CG."""
    solution = initial.copy()
    residual = np.empty_like(solution)
    preconditioned = np.empty_like(solution)
    direction = np.empty_like(solution)
    product = np.empty_like(solution)
    ny, nx = solution.shape

    rhs_norm_squared = 0.0
    residual_norm_squared = 0.0
    rho = 0.0
    for i in range(ny):
        for j in range(nx):
            if diagonal[i, j] <= 0.0 or not np.isfinite(diagonal[i, j]):
                raise RuntimeError("radiative_2d PCG requires a positive finite diagonal")
            value = diagonal[i, j] * solution[i, j]
            if j > 0:
                value -= west[i, j] * solution[i, j - 1]
            if j + 1 < nx:
                value -= east[i, j] * solution[i, j + 1]
            if i > 0:
                value -= south[i, j] * solution[i - 1, j]
            if i + 1 < ny:
                value -= north[i, j] * solution[i + 1, j]
            residual[i, j] = rhs[i, j] - value
            preconditioned[i, j] = residual[i, j] / diagonal[i, j]
            direction[i, j] = preconditioned[i, j]
            rhs_norm_squared += rhs[i, j] * rhs[i, j]
            residual_norm_squared += residual[i, j] * residual[i, j]
            rho += residual[i, j] * preconditioned[i, j]

    residual_limit = max(absolute_tolerance, tolerance * np.sqrt(rhs_norm_squared))
    if np.sqrt(residual_norm_squared) <= residual_limit:
        return solution
    if rho <= 0.0 or not np.isfinite(rho):
        raise RuntimeError("radiative_2d PCG preconditioner breakdown")

    for _ in range(max_iterations):
        denominator = 0.0
        for i in range(ny):
            for j in range(nx):
                value = diagonal[i, j] * direction[i, j]
                if j > 0:
                    value -= west[i, j] * direction[i, j - 1]
                if j + 1 < nx:
                    value -= east[i, j] * direction[i, j + 1]
                if i > 0:
                    value -= south[i, j] * direction[i - 1, j]
                if i + 1 < ny:
                    value -= north[i, j] * direction[i + 1, j]
                product[i, j] = value
                denominator += direction[i, j] * value
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise RuntimeError("radiative_2d PCG matrix breakdown")

        alpha = rho / denominator
        residual_norm_squared = 0.0
        for i in range(ny):
            for j in range(nx):
                solution[i, j] += alpha * direction[i, j]
                residual[i, j] -= alpha * product[i, j]
                residual_norm_squared += residual[i, j] * residual[i, j]
        restart = False
        if np.sqrt(residual_norm_squared) <= residual_limit:
            # Verify the recursively updated residual before accepting.  If
            # roundoff has accumulated, restart CG from the true residual.
            residual_norm_squared = 0.0
            for i in range(ny):
                for j in range(nx):
                    value = diagonal[i, j] * solution[i, j]
                    if j > 0:
                        value -= west[i, j] * solution[i, j - 1]
                    if j + 1 < nx:
                        value -= east[i, j] * solution[i, j + 1]
                    if i > 0:
                        value -= south[i, j] * solution[i - 1, j]
                    if i + 1 < ny:
                        value -= north[i, j] * solution[i + 1, j]
                    residual[i, j] = rhs[i, j] - value
                    residual_norm_squared += residual[i, j] * residual[i, j]
            if np.sqrt(residual_norm_squared) <= residual_limit:
                return solution
            restart = True

        next_rho = 0.0
        for i in range(ny):
            for j in range(nx):
                preconditioned[i, j] = residual[i, j] / diagonal[i, j]
                next_rho += residual[i, j] * preconditioned[i, j]
        if next_rho <= 0.0 or not np.isfinite(next_rho):
            raise RuntimeError("radiative_2d PCG preconditioner breakdown")
        if restart:
            for i in range(ny):
                for j in range(nx):
                    direction[i, j] = preconditioned[i, j]
        else:
            beta = next_rho / rho
            for i in range(ny):
                for j in range(nx):
                    direction[i, j] = preconditioned[i, j] + beta * direction[i, j]
        rho = next_rho
    raise RuntimeError("radiative_2d PCG linear solve failed to converge")


def _cell_gradient(field: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    if min(field.shape) < 3:
        return np.gradient(field, dy, dx, edge_order=1)[1], np.gradient(field, dy, dx, edge_order=1)[0]
    gy, gx = np.gradient(field, dy, dx, edge_order=2)
    return gx, gy


def solve_radiative_step_2d(
    energy_old: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    material: np.ndarray,
    source: np.ndarray,
    boundary_left: np.ndarray,
    boundary_right: np.ndarray,
    boundary_bottom: np.ndarray,
    boundary_top: np.ndarray,
    *,
    use_flux_limiter: bool = True,
    smooth_limiter: bool = False,
    picard_tolerance: float = 1e-6,
    picard_iterations: int = 2000,
    linear_tolerance: float = 1e-7,
    linear_absolute_tolerance: float = 1e-12,
    linear_iterations: int = 2000,
    omega: float = 1.9,
    linear_solver: str = "sor",
    picard_relaxation: float = 1.0,
    legacy_boundary_extrapolation: bool = False,
) -> np.ndarray:
    """Advance one fully implicit Picard step with Marshak conditions."""
    energy_old = np.asarray(energy_old, dtype=float)
    material = np.asarray(material, dtype=float)
    if energy_old.ndim != 2 or energy_old.shape != material.shape or min(energy_old.shape) < 2:
        raise ValueError("energy and material must be matching 2D arrays with at least 2x2 cells")
    linear_solver = str(linear_solver).lower()
    if linear_solver not in ("sor", "pcg"):
        raise ValueError("linear_solver must be 'sor' or 'pcg'")
    if linear_tolerance <= 0.0 or linear_absolute_tolerance < 0.0 or linear_iterations < 1:
        raise ValueError("linear solver tolerances and iteration count must be valid")
    if not 0.0 < picard_relaxation <= 1.0:
        raise ValueError("picard_relaxation must be in (0, 1]")
    ny, nx = energy_old.shape
    iterate = energy_old.copy()
    for iteration in range(picard_iterations):
        safe = np.maximum(iterate, 1e-12)
        sigma = material**3 / safe**0.75
        if use_flux_limiter:
            gx, gy = _cell_gradient(iterate, dx, dy)
            grad_norm = np.sqrt(gx**2 + gy**2 + 1e-15) if smooth_limiter else np.abs(gx) + np.abs(gy)
            diffusion = 1.0 / (3.0 * sigma + grad_norm / safe)
        else:
            diffusion = 1.0 / (3.0 * sigma)

        diff_x = np.empty((ny, nx + 1))
        diff_y = np.empty((ny + 1, nx))
        diff_x[:, 1:-1] = 2.0 * diffusion[:, :-1] * diffusion[:, 1:] / (
            diffusion[:, :-1] + diffusion[:, 1:] + 1e-15
        )
        diff_y[1:-1, :] = 2.0 * diffusion[:-1, :] * diffusion[1:, :] / (
            diffusion[:-1, :] + diffusion[1:, :] + 1e-15
        )
        if legacy_boundary_extrapolation:
            # This is the archived closure: extrapolate the cell diffusion
            # coefficient to the geometric Robin face before assembling Marshak.
            diff_x[:, 0] = np.maximum(
                1.5 * diffusion[:, 0] - 0.5 * diffusion[:, 1], 1e-15
            )
            diff_x[:, -1] = np.maximum(
                1.5 * diffusion[:, -1] - 0.5 * diffusion[:, -2], 1e-15
            )
            diff_y[0, :] = np.maximum(
                1.5 * diffusion[0, :] - 0.5 * diffusion[1, :], 1e-15
            )
            diff_y[-1, :] = np.maximum(
                1.5 * diffusion[-1, :] - 0.5 * diffusion[-2, :], 1e-15
            )
        else:
            if use_flux_limiter:
                left_tangent = 1.5 * gy[:, 0] - 0.5 * gy[:, 1]
                right_tangent = 1.5 * gy[:, -1] - 0.5 * gy[:, -2]
                bottom_tangent = 1.5 * gx[0, :] - 0.5 * gx[1, :]
                top_tangent = 1.5 * gx[-1, :] - 0.5 * gx[-2, :]
            else:
                left_tangent = right_tangent = np.zeros(ny)
                bottom_tangent = top_tangent = np.zeros(nx)
            boundary_lengths = (ny, ny, nx, nx)
            _, boundary_diffusion = solve_radiative_robin_face(
                np.concatenate((
                    iterate[:, 0], iterate[:, -1], iterate[0, :], iterate[-1, :]
                )),
                np.concatenate((
                    np.full(ny, dx), np.full(ny, dx),
                    np.full(nx, dy), np.full(nx, dy),
                )),
                np.concatenate((
                    boundary_left, boundary_right, boundary_bottom, boundary_top
                )),
                np.concatenate((
                    material[:, 0], material[:, -1], material[0, :], material[-1, :]
                )),
                tangential_gradient=np.concatenate((
                    left_tangent, right_tangent, bottom_tangent, top_tangent
                )),
                use_flux_limiter=use_flux_limiter, smooth_limiter=smooth_limiter,
            )
            left_diff, right_diff, bottom_diff, top_diff = np.split(
                boundary_diffusion, np.cumsum(boundary_lengths)[:-1]
            )
            diff_x[:, 0], diff_x[:, -1] = left_diff, right_diff
            diff_y[0, :], diff_y[-1, :] = bottom_diff, top_diff

        west, east = dt * diff_x[:, :-1] / dx**2, dt * diff_x[:, 1:] / dx**2
        south, north = dt * diff_y[:-1, :] / dy**2, dt * diff_y[1:, :] / dy**2
        diagonal = 1.0 + west + east + south + north
        rhs = energy_old + dt * np.asarray(source, dtype=float)

        gamma = 2.0 * diff_x[:, 0] / dx
        diagonal[:, 0] -= west[:, 0]
        diagonal[:, 0] += dt / dx * 0.5 * gamma / (0.5 + gamma)
        rhs[:, 0] += dt / dx * gamma / (0.5 + gamma) * boundary_left
        west[:, 0] = 0.0
        gamma = 2.0 * diff_x[:, -1] / dx
        diagonal[:, -1] -= east[:, -1]
        diagonal[:, -1] += dt / dx * 0.5 * gamma / (0.5 + gamma)
        rhs[:, -1] += dt / dx * gamma / (0.5 + gamma) * boundary_right
        east[:, -1] = 0.0
        gamma = 2.0 * diff_y[0, :] / dy
        diagonal[0, :] -= south[0, :]
        diagonal[0, :] += dt / dy * 0.5 * gamma / (0.5 + gamma)
        rhs[0, :] += dt / dy * gamma / (0.5 + gamma) * boundary_bottom
        south[0, :] = 0.0
        gamma = 2.0 * diff_y[-1, :] / dy
        diagonal[-1, :] -= north[-1, :]
        diagonal[-1, :] += dt / dy * 0.5 * gamma / (0.5 + gamma)
        rhs[-1, :] += dt / dy * gamma / (0.5 + gamma) * boundary_top
        north[-1, :] = 0.0

        if linear_solver == "pcg":
            next_iterate = pcg_solve(
                west, east, south, north, diagonal, rhs, iterate,
                tolerance=linear_tolerance, absolute_tolerance=linear_absolute_tolerance,
                max_iterations=linear_iterations,
            )
        else:
            next_iterate = rbgs_sor_solve(
                west, east, south, north, diagonal, rhs, iterate,
                tolerance=linear_tolerance, max_iterations=linear_iterations, omega=omega,
            )
        # Under-relaxation is a numerical aid for difficult nonlinear fixed
        # points.  At the converged point it leaves the same implicit FV
        # equation unchanged; the default value preserves the old path.
        candidate = (
            next_iterate if picard_relaxation == 1.0
            else iterate + picard_relaxation * (next_iterate - iterate)
        )
        error = np.linalg.norm(candidate - iterate) / (np.linalg.norm(candidate) + 1e-15)
        if error < picard_tolerance:
            return candidate
        iterate = candidate
    raise RuntimeError("radiative_2d Picard iteration failed to converge")


def solve_radiative_2d(
    initial: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    material: np.ndarray,
    source: np.ndarray,
    boundaries: dict[str, np.ndarray],
    *,
    use_flux_limiter: bool = True,
    smooth_limiter: bool = False,
    save_t: np.ndarray | None = None,
    save_x: np.ndarray | None = None,
    save_y: np.ndarray | None = None,
    solver_options: dict | None = None,
) -> np.ndarray:
    t, x, y = np.asarray(t), np.asarray(x), np.asarray(y)
    save_t = np.arange(len(t)) if save_t is None else np.asarray(save_t, dtype=int)
    save_x = np.arange(len(x)) if save_x is None else np.asarray(save_x, dtype=int)
    save_y = np.arange(len(y)) if save_y is None else np.asarray(save_y, dtype=int)
    positions = {int(index): pos for pos, index in enumerate(save_t)}
    result = np.empty((len(save_t), len(save_y), len(save_x)))
    state = np.asarray(initial, dtype=float).copy()
    if 0 in positions:
        result[positions[0]] = state[np.ix_(save_y, save_x)]
    report_solver_progress(0, len(t) - 1)
    options = solver_options or {}
    for step in range(1, len(t)):
        try:
            state = solve_radiative_step_2d(
                state, x[1] - x[0], y[1] - y[0], t[step] - t[step - 1], material, source[step],
                boundaries["bL"][step], boundaries["bR"][step],
                boundaries["bB"][step], boundaries["bT"][step],
                use_flux_limiter=use_flux_limiter, smooth_limiter=smooth_limiter, **options,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"radiative_2d solve failed at time step {step}/{len(t) - 1}"
            ) from error
        if step in positions:
            result[positions[step]] = state[np.ix_(save_y, save_x)]
        report_solver_progress(step, len(t) - 1)
    return result


def _generate_radiative_2d_sample(context, seed):
    rng = np.random.default_rng(seed)
    t, x, y = context["t"], context["x"], context["y"]
    initial, initial_parameters = build_initial_energy(x, y, context["initial_condition"], rng)
    if context["boundary_mode"] == "zero":
        boundaries = {
            "bL": np.zeros((len(t), len(y))), "bR": np.zeros((len(t), len(y))),
            "bB": np.zeros((len(t), len(x))), "bT": np.zeros((len(t), len(x))),
        }
    else:
        left_boundary = (
            generate_legacy_left_boundary(
                t, len(y) if context.get("legacy_layout", False) else y,
                rng, context["boundary_mode"]
            )
            if context.get("legacy_boundary_formula", False)
            else generate_left_boundary(t, y, rng, context["boundary_mode"])
        )
        boundaries = {
            "bL": left_boundary,
            "bR": np.zeros((len(t), len(y))),
            "bB": np.zeros((len(t), len(x))),
            "bT": np.zeros((len(t), len(x))),
        }
    field = solve_radiative_2d(
        initial, t, x, y, context["material"], context["source"], boundaries,
        use_flux_limiter=context["limiter"], smooth_limiter=context["smooth"],
        save_t=context["save_t"], save_x=context["save_x"], save_y=context["save_y"],
        solver_options=context["solver_options"],
    )
    it, ix, iy = context["save_t"], context["save_x"], context["save_y"]
    if context.get("legacy_layout", False):
        saved_condition = {
            name: boundaries[name][np.ix_(it, iy if name in ("bL", "bR") else ix)]
            for name in ("bL", "bR", "bB", "bT")
        }
    else:
        # ``gL`` is a Robin right-hand side, not a boundary value of E.  Keep
        # it separate from the interior solution so no loader can accidentally
        # turn it into a supervised E boundary slot.
        saved_condition = {"gL": boundaries["bL"][np.ix_(it, iy)]}
    return field, saved_condition, initial[np.ix_(iy, ix)], initial_parameters


class Radiative2DCase(CaseAdapter):
    name, dimension = "radiative_2d", 2

    def validate(self, data: CaseData) -> None:
        """Validate the layout belonging to this adapter's concrete case."""
        if self.name == "radiative_initial_2d":
            # ``RadiativeInitial2DCase`` intentionally keeps the archived
            # packed-boundary file format.  Do not make its existing data
            # depend on the new radiative_2d gL contract.
            CaseAdapter.validate(self, data)
            samples, times, ny, nx = data.fields.shape
            expected = {
                "bL": (samples, times, ny), "bR": (samples, times, ny),
                "bB": (samples, times, nx), "bT": (samples, times, nx),
            }
            for name, shape in expected.items():
                if name not in data.boundaries:
                    raise ValueError(
                        f"radiative_initial_2d data must contain boundary dataset {name!r}"
                    )
                if tuple(data.boundaries[name].shape) != shape:
                    raise ValueError(
                        f"radiative_initial_2d {name} shape {data.boundaries[name].shape} "
                        f"does not match {shape}"
                    )
                if not np.all(np.isfinite(data.boundaries[name])):
                    raise ValueError(f"radiative_initial_2d {name} contains NaN or infinity")
            return

        """Validate the interior-plus-left-Robin layout used by this case."""
        super().validate(data)
        if "gL" not in data.boundaries:
            raise ValueError("radiative_2d data must contain the saved gL condition")
        expected = (data.fields.shape[0], data.fields.shape[1], data.fields.shape[2])
        if data.boundaries["gL"].shape != expected:
            raise ValueError(
                f"radiative_2d gL shape {data.boundaries['gL'].shape} does not match {expected}"
            )
        if not np.all(np.isfinite(data.boundaries["gL"])):
            raise ValueError("radiative_2d gL contains NaN or infinity")

    def generate(self, config, output_path):
        legacy_layout = self.name == "radiative_initial_2d"
        gen = config.get("generation", {})
        samples = int(gen.get("samples", 100))
        nx, ny, nt = int(gen.get("nx", 128)), int(gen.get("ny", 128)), int(gen.get("nt", 1200))
        nx_save, ny_save, nt_save = (int(gen.get("nx_save", 64)), int(gen.get("ny_save", 64)),
                                     int(gen.get("nt_save", 64)))
        if min(nx, ny, nt) < 2 or not (1 <= nx_save <= nx and 1 <= ny_save <= ny and 1 <= nt_save <= nt):
            raise ValueError("generation grid/save sizes are invalid")
        x0, x1 = config["domain"]["x"]
        y0, y1 = config["domain"]["y"]
        t0, t1 = config["domain"]["t"]
        dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
        x = np.linspace(x0 + dx / 2, x1 - dx / 2, nx)
        y = np.linspace(y0 + dy / 2, y1 - dy / 2, ny)
        t = np.linspace(t0, t1, nt)
        ix = np.linspace(0, nx - 1, nx_save, dtype=int)
        iy = np.linspace(0, ny - 1, ny_save, dtype=int)
        it = np.linspace(0, nt - 1, nt_save, dtype=int)
        physics = config.get("physics", {})
        material = build_material(x, y, physics.get("material", {}))
        source = build_source(t, x, y, physics.get("source", {}))
        initial_condition = gen.get("initial_condition", {
            "type": "constant", "value": float(gen.get("initial_value", 1e-3)),
        })
        limiter = bool(physics.get("use_flux_limiter", True))
        smooth = gen.get("limiter_norm", physics.get("limiter_norm", "l1")) == "l2_smooth"
        seed = int(gen.get("seed", config.get("seed", 42)))
        if gen.get("seed_mode") == "legacy_indexed":
            sample_seeds = [int(gen.get("sample_seed", seed)) + index for index in range(samples)]
        else:
            sample_seeds = independent_seeds(seed, samples)
        results = parallel_samples(
            _generate_radiative_2d_sample,
            {"t": t, "x": x, "y": y, "initial_condition": initial_condition, "material": material,
            "source": source, "boundary_mode": gen.get("boundary_mode", "sustained"),
             "legacy_boundary_formula": bool(gen.get("legacy_boundary_formula", False)),
             "legacy_layout": legacy_layout,
             "limiter": limiter, "smooth": smooth, "save_t": it, "save_x": ix,
             "save_y": iy, "solver_options": gen.get("solver", {})},
            sample_seeds, int(gen.get("workers", 1)),
            description=gen.get("progress_label", self.name),
            show_progress=bool(gen.get("show_progress", True)),
            progress_interval=float(gen.get("progress_interval", 30.0)),
        )
        fields = [field for field, _, _, _ in results]
        initial_fields = [initial for _, _, initial, _ in results]
        parameter_names = ("amplitude", "sigma", "center_x", "center_y")
        initial_parameters = np.asarray([
            [parameters[name] for name in parameter_names] for _, _, _, parameters in results
        ])
        # Keep the complete condition dictionaries until the layout-specific
        # write below.  The legacy case stores four boundary arrays, whereas
        # the new radiative_2d case stores only the saved Robin RHS ``gL``.
        saved_conditions = [condition for _, condition, _, _ in results]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as handle:
            handle["u"] = np.asarray(fields)
            if legacy_layout:
                for name in ("bL", "bR", "bB", "bT"):
                    handle[name] = np.asarray([condition[name] for condition in saved_conditions])
            else:
                handle["gL"] = np.asarray([condition["gL"] for condition in saved_conditions])
            handle["x"], handle["y"], handle["t"] = x[ix], y[iy], t[it]
            handle["initial"] = np.asarray(initial_fields)
            handle["initial_parameters"] = initial_parameters
            handle["initial_parameters"].attrs["columns"] = json.dumps(parameter_names)
            handle["material"] = material[np.ix_(iy, ix)]
            handle["source"] = source[np.ix_(it, iy, ix)]
            handle.attrs["case"] = self.name
            if not legacy_layout:
                handle.attrs["data_layout"] = "interior_plus_gL"
            handle.attrs["config_json"] = json.dumps(config, ensure_ascii=False)
        return output_path

    def load(self, path):
        with h5py.File(path, "r") as handle:
            metadata = dict(handle.attrs)
            if self.name == "radiative_initial_2d":
                boundary_names = ("bL", "bR", "bB", "bT")
            else:
                boundary_names = ("gL",)
            data = CaseData(
                handle["u"][:], {name: handle[name][:] for name in ("t", "y", "x")},
                {name: handle[name][:] for name in boundary_names},
                material=handle["material"][:] if "material" in handle else None,
                source=handle["source"][:] if "source" in handle else None,
                metadata=metadata,
            )
        self.validate(data)
        return data

    def evaluate(self, prediction, target, data):
        metrics = super().evaluate(prediction, target, data)
        if self.name != "radiative_2d":
            return metrics
        if data.material is not None and min(prediction.shape[1:]) >= 3:
            u_t = np.gradient(prediction, data.axes["t"], axis=1, edge_order=2)
            u_y = np.gradient(prediction, data.axes["y"], axis=2, edge_order=2)
            u_x = np.gradient(prediction, data.axes["x"], axis=3, edge_order=2)
            safe = np.maximum(prediction, 1e-8)
            material = data.material[None, None]
            grad_norm = np.abs(u_x) + np.abs(u_y)
            diffusion = safe / (3.0 * material**3 * safe**0.25 + grad_norm)
            div_flux = np.gradient(-diffusion * u_x, data.axes["x"], axis=3, edge_order=2)
            div_flux += np.gradient(-diffusion * u_y, data.axes["y"], axis=2, edge_order=2)
            source = 0.0 if data.source is None else np.asarray(data.source)
            if np.ndim(source) == 3:
                source = source[None]
            metrics.update(residual_statistics((u_t + div_flux - source)[:, 1:, 1:-1, 1:-1]))
        return metrics


register_case(Radiative2DCase())
