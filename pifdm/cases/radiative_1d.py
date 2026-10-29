"""Implicit finite-volume solver for one-dimensional radiative diffusion."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .base import CaseAdapter, CaseData, register_case, residual_statistics
from .acceleration import independent_seeds, parallel_samples, report_solver_progress
from .linear_1d import tdma_solve
from .radiative_boundary import solve_radiative_robin_face


def _face_material(x_faces: np.ndarray, spec: dict) -> np.ndarray:
    values = np.full_like(x_faces, float(spec.get("value", spec.get("background", 1.0))), dtype=float)
    for region in spec.get("regions", []):
        lo, hi = region["x"]
        values[(x_faces >= lo) & (x_faces <= hi)] = float(region["value"])
    return values


def _source_grid(t: np.ndarray, x: np.ndarray, spec: dict) -> np.ndarray:
    kind = spec.get("type", "constant")
    if kind == "constant":
        return np.full((len(t), len(x)), float(spec.get("value", 0.0)))
    raise ValueError(f"radiative_1d generation does not support source type {kind!r}")


def generate_left_boundary(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a non-negative incident Marshak flux."""
    duration = max(float(t[-1] - t[0]), np.finfo(float).eps)
    tau = (t - t[0]) / duration
    amplitude = rng.uniform(2.0, 10.0)
    mode = rng.choice(("multi_pulse", "step_hold", "ramp_decay", "random_fluctuation"))
    if mode == "multi_pulse":
        boundary = np.zeros_like(tau)
        for _ in range(int(rng.integers(1, 4))):
            boundary += rng.uniform(0.5, 1.5) * amplitude * np.exp(
                -0.5 * ((tau - rng.uniform(0.05, 0.9)) / rng.uniform(0.02, 0.06)) ** 2
            )
    elif mode == "step_hold":
        start = rng.uniform(0.0, 0.3)
        boundary = amplitude / (1.0 + np.exp(-rng.uniform(50.0, 200.0) * (tau - start)))
        boundary *= 1.0 + 0.3 * np.exp(-20.0 * np.maximum(0.0, tau - start))
    elif mode == "ramp_decay":
        peak = rng.uniform(0.1, 0.3)
        boundary = np.where(tau <= peak, amplitude * tau / peak,
                            amplitude * np.exp(-rng.uniform(1.0, 5.0) * (tau - peak)))
    else:
        noise = rng.standard_normal(len(t))
        width = max(1, int(0.05 * len(t)))
        noise = np.convolve(noise, np.ones(width) / width, mode="same")
        boundary = amplitude * (noise - noise.min()) / (np.ptp(noise) + 1e-12)
    return np.maximum(boundary, 1e-8)


def solve_radiative_step(
    energy_old: np.ndarray,
    dx: float,
    dt: float,
    material_faces: np.ndarray,
    source: np.ndarray,
    boundary_left: float,
    boundary_right: float,
    *,
    use_flux_limiter: bool = True,
    smooth_limiter: bool = True,
    tolerance: float = 1e-7,
    max_iterations: int = 2000,
) -> np.ndarray:
    """Advance one fully implicit step using Picard linearisation."""
    energy_old = np.asarray(energy_old, dtype=float)
    nx = len(energy_old)
    if nx < 2 or np.shape(material_faces) != (nx + 1,):
        raise ValueError("energy needs at least two cells and material_faces must have nx+1 entries")
    iterate = energy_old.copy()
    for iteration in range(max_iterations):
        face_energy = np.empty(nx + 1)
        face_energy[1:-1] = 0.5 * (iterate[:-1] + iterate[1:])
        face_energy[[0, -1]] = iterate[[0, -1]]
        safe_energy = np.maximum(face_energy, 1e-12)
        sigma = np.asarray(material_faces, dtype=float) ** 3 / safe_energy ** 0.75
        gradient = np.empty(nx + 1)
        gradient[1:-1] = np.diff(iterate) / dx
        gradient[0], gradient[-1] = gradient[1], gradient[-2]
        if use_flux_limiter:
            magnitude = np.sqrt(gradient**2 + 1e-15) if smooth_limiter else np.abs(gradient)
            diffusion = 1.0 / (3.0 * sigma + magnitude / safe_energy)
        else:
            diffusion = 1.0 / (3.0 * sigma)
        _, boundary_diffusion = solve_radiative_robin_face(
            iterate[[0, -1]], dx, np.asarray([boundary_left, boundary_right]),
            material_faces[[0, -1]],
            use_flux_limiter=use_flux_limiter, smooth_limiter=smooth_limiter,
        )
        diffusion[[0, -1]] = boundary_diffusion

        lower = np.zeros(nx)
        diagonal = np.ones(nx)
        upper = np.zeros(nx)
        rhs = energy_old + dt * np.asarray(source, dtype=float)
        conductance = dt * diffusion[1:-1] / dx**2
        diagonal[:-1] += conductance
        diagonal[1:] += conductance
        upper[:-1] = -conductance
        lower[1:] = -conductance

        gamma_left = 2.0 * diffusion[0] / dx
        gamma_right = 2.0 * diffusion[-1] / dx
        diagonal[0] += dt / dx * 0.5 * gamma_left / (0.5 + gamma_left)
        diagonal[-1] += dt / dx * 0.5 * gamma_right / (0.5 + gamma_right)
        rhs[0] += dt / dx * gamma_left / (0.5 + gamma_left) * boundary_left
        rhs[-1] += dt / dx * gamma_right / (0.5 + gamma_right) * boundary_right

        next_iterate = tdma_solve(lower, diagonal, upper, rhs)
        error = np.linalg.norm(next_iterate - iterate) / (np.linalg.norm(next_iterate) + 1e-15)
        if error < tolerance:
            return next_iterate
        iterate = next_iterate
    raise RuntimeError("radiative_1d Picard iteration failed to converge")


def solve_radiative_1d(
    initial: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    material_faces: np.ndarray,
    source: np.ndarray | None = None,
    boundary_left: np.ndarray | None = None,
    boundary_right: np.ndarray | None = None,
    *,
    use_flux_limiter: bool = True,
    smooth_limiter: bool = True,
    save_indices: np.ndarray | None = None,
) -> np.ndarray:
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    dx, dt = x[1] - x[0], t[1] - t[0]
    source = np.zeros((len(t), len(x))) if source is None else np.asarray(source, dtype=float)
    boundary_left = np.zeros(len(t)) if boundary_left is None else np.asarray(boundary_left, dtype=float)
    boundary_right = np.zeros(len(t)) if boundary_right is None else np.asarray(boundary_right, dtype=float)
    save_indices = np.arange(len(t)) if save_indices is None else np.asarray(save_indices, dtype=int)
    result = np.empty((len(save_indices), len(x)))
    positions = {int(index): pos for pos, index in enumerate(save_indices)}
    state = np.asarray(initial, dtype=float).copy()
    if 0 in positions:
        result[positions[0]] = state
    report_solver_progress(0, len(t) - 1)
    for step in range(1, len(t)):
        state = solve_radiative_step(
            state, dx, dt, material_faces, source[step], boundary_left[step], boundary_right[step],
            use_flux_limiter=use_flux_limiter, smooth_limiter=smooth_limiter,
        )
        if step in positions:
            result[positions[step]] = state
        report_solver_progress(step, len(t) - 1)
    return result


def _generate_radiative_1d_sample(context, seed):
    rng = np.random.default_rng(seed)
    left = generate_left_boundary(context["t"], rng)
    right = np.full(len(context["t"]), context["right_value"])
    solution = solve_radiative_1d(
        context["initial"], context["t"], context["x"], context["material_faces"],
        context["source"], left, right, use_flux_limiter=context["limiter"],
        smooth_limiter=context["smooth"], save_indices=context["save_t"],
    )
    return (solution[:, context["save_x"]], left[context["save_t"]],
            right[context["save_t"]])


class Radiative1DCase(CaseAdapter):
    name, dimension = "radiative_1d", 1

    def generate(self, config, output_path):
        gen = config.get("generation", {})
        samples, nx, nt = (int(gen.get(key, default)) for key, default in
                           (("samples", 100), ("nx", 512), ("nt", 5120)))
        nx_save, nt_save = int(gen.get("nx_save", 126)), int(gen.get("nt_save", 64))
        if nx < 2 or nt < 2 or not (1 <= nx_save <= nx and 1 <= nt_save <= nt):
            raise ValueError("generation requires nx,nt >= 2 and save sizes within the fine grid")
        x0, x1 = config["domain"]["x"]
        t0, t1 = config["domain"]["t"]
        dx = (x1 - x0) / nx
        x = np.linspace(x0 + dx / 2, x1 - dx / 2, nx)
        t = np.linspace(t0, t1, nt)
        save_x = np.linspace(0, nx - 1, nx_save, dtype=int)
        save_t = np.linspace(0, nt - 1, nt_save, dtype=int)
        physics = config.get("physics", {})
        material_faces = _face_material(np.linspace(x0, x1, nx + 1), physics.get("material", {}))
        material_cells = _face_material(x, physics.get("material", {}))
        source = _source_grid(t, x, physics.get("source", {}))
        initial = np.full(nx, float(gen.get("initial_value", 1e-3)))
        right_value = float(gen.get("right_boundary", 0.0))
        limiter = bool(physics.get("use_flux_limiter", True))
        smooth = physics.get("limiter_norm", "l2_smooth") == "l2_smooth"
        seed = int(gen.get("seed", config.get("seed", 42)))
        results = parallel_samples(
            _generate_radiative_1d_sample,
            {"t": t, "x": x, "initial": initial, "material_faces": material_faces,
             "source": source, "right_value": right_value, "limiter": limiter,
             "smooth": smooth, "save_t": save_t, "save_x": save_x},
            independent_seeds(seed, samples), int(gen.get("workers", 1)),
            description=gen.get("progress_label", "radiative_1d"),
            show_progress=bool(gen.get("show_progress", True)),
            progress_interval=float(gen.get("progress_interval", 30.0)),
        )
        fields, lefts, rights = (list(values) for values in zip(*results))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as handle:
            handle["u"] = np.asarray(fields)
            handle["bL"] = np.asarray(lefts)
            handle["bR"] = np.asarray(rights)
            handle["x"], handle["t"] = x[save_x], t[save_t]
            handle["material"] = material_cells[save_x]
            handle["source"] = source[np.ix_(save_t, save_x)]
            handle.attrs["case"] = self.name
            handle.attrs["config_json"] = json.dumps(config, ensure_ascii=False)
        return output_path

    def load(self, path):
        with h5py.File(path, "r") as handle:
            data = CaseData(
                handle["u"][:], {name: handle[name][:] for name in ("t", "x")},
                {name: handle[name][:] for name in ("bL", "bR")},
                material=handle["material"][:] if "material" in handle else None,
                source=handle["source"][:] if "source" in handle else None,
                metadata=dict(handle.attrs),
            )
        self.validate(data)
        return data

    def evaluate(self, prediction, target, data):
        metrics = super().evaluate(prediction, target, data)
        if data.material is not None and min(prediction.shape[1:]) >= 3:
            u_t = np.gradient(prediction, data.axes["t"], axis=1, edge_order=2)
            u_x = np.gradient(prediction, data.axes["x"], axis=2, edge_order=2)
            safe = np.maximum(prediction, 1e-8)
            material = data.material[None, None, :]
            diffusion = safe / (3.0 * material**3 * safe**0.25 + np.sqrt(u_x**2 + 1e-15))
            flux = -diffusion * u_x
            div_flux = np.gradient(flux, data.axes["x"], axis=2, edge_order=2)
            source = 0.0 if data.source is None else np.asarray(data.source)
            if np.ndim(source) == 2:
                source = source[None]
            metrics.update(residual_statistics((u_t + div_flux - source)[:, 1:, 1:-1]))
        return metrics


register_case(Radiative1DCase())
