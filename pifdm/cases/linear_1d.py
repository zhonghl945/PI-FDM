"""Cell-centred implicit FVM solver for u_t = D u_xx + f."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import h5py

from .base import CaseAdapter, CaseData, register_case, residual_statistics
from .acceleration import independent_seeds, njit, parallel_samples, report_solver_progress


@njit
def tdma_solve(lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = len(rhs)
    c_prime = np.zeros(n, dtype=np.float64)
    d_prime = np.zeros(n, dtype=np.float64)
    c_prime[0] = upper[0] / diagonal[0]
    d_prime[0] = rhs[0] / diagonal[0]
    for i in range(1, n):
        denom = diagonal[i] - lower[i] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom
    solution = np.empty(n, dtype=np.float64)
    solution[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    return solution


def solve_linear_step(u_old, dx, dt, source, b_left, b_right, diffusivity=1.0):
    nx = len(u_old)
    r = diffusivity * dt / dx**2
    lower = np.zeros(nx)
    diagonal = np.full(nx, 1.0 + 2.0 * r)
    upper = np.zeros(nx)
    rhs = u_old + dt * source
    lower[1:-1] = -r
    upper[1:-1] = -r
    diagonal[0] = 1.0 + 3.0 * r
    upper[0] = -r
    rhs[0] += 2.0 * r * b_left
    diagonal[-1] = 1.0 + r
    lower[-1] = -r
    rhs[-1] += r * dx * b_right
    return tdma_solve(lower, diagonal, upper, rhs)


def solve_linear(u0, t, x, source=None, b_left=None, b_right=None, diffusivity=1.0, save_indices=None):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or len(t) < 2:
        raise ValueError("linear FVM needs at least two time and cell-center coordinates")
    dx = (x[-1] - x[0]) / (len(x) - 1)
    dt = t[1] - t[0]
    source = np.zeros((len(t), len(x))) if source is None else np.asarray(source)
    b_left = np.zeros(len(t)) if b_left is None else np.asarray(b_left)
    b_right = np.zeros(len(t)) if b_right is None else np.asarray(b_right)
    save_indices = np.arange(len(t)) if save_indices is None else np.asarray(save_indices, dtype=int)
    result = np.empty((len(save_indices), len(x)), dtype=float)
    state = np.asarray(u0, dtype=float).copy()
    save_pos = {int(index): pos for pos, index in enumerate(save_indices)}
    if 0 in save_pos:
        result[save_pos[0]] = state
    report_solver_progress(0, len(t) - 1)
    for step in range(1, len(t)):
        state = solve_linear_step(state, dx, dt, source[step], b_left[step], b_right[step], diffusivity)
        if step in save_pos:
            result[save_pos[step]] = state
        report_solver_progress(step, len(t) - 1)
    return result


def _generate_linear_sample(context, seed):
    rng = np.random.default_rng(seed)
    t = context["t"]
    b_left = np.zeros(len(t))
    for _ in range(int(rng.integers(1, 4))):
        b_left += rng.uniform(0.5, 2.0) * np.exp(
            -0.5 * ((t - rng.uniform(0.1, 0.9 * context["t_end"])) / rng.uniform(0.02, 0.08)) ** 2
        )
    b_left *= np.tanh(t / max(0.02 * context["t_end"], 1e-12))
    b_right = np.zeros(len(t))
    solution = solve_linear(
        context["initial"], t, context["x"], source=context["source"],
        b_left=b_left, b_right=b_right, diffusivity=context["diffusivity"],
        save_indices=context["save_t"],
    )
    return solution[:, context["save_x"]], b_left[context["save_t"]], b_right[context["save_t"]]


class Linear1DCase(CaseAdapter):
    name, dimension = "linear_1d", 1

    def generate(self, config, output_path):
        gen = config.get("generation", {})
        samples = int(gen.get("samples", 100))
        nx = int(gen.get("nx", 512))
        nt = int(gen.get("nt", 5120))
        nx_save = int(gen.get("nx_save", 126))
        nt_save = int(gen.get("nt_save", 64))
        x_min, L = config["domain"]["x"]
        t_min, T = config["domain"]["t"]
        if nx < 2 or nt < 2 or not (1 <= nx_save <= nx and 1 <= nt_save <= nt):
            raise ValueError("generation requires nx,nt >= 2 and save sizes within the fine grid")
        dx = (L - x_min) / nx
        x = np.linspace(x_min + dx / 2, L - dx / 2, nx)
        t = np.linspace(t_min, T, nt)
        save_x = np.linspace(0, nx - 1, nx_save, dtype=int)
        save_t = np.linspace(0, nt - 1, nt_save, dtype=int)
        physics = config.get("physics", {})
        diffusivity = float(physics.get("diffusivity", 1.0))
        source_spec = physics.get("source", {"type": "constant", "value": 0.0})
        if source_spec.get("type", "constant") != "constant":
            raise ValueError("linear_1d generation currently supports only a constant source")
        source_value = float(source_spec.get("value", 0.0))
        source = np.full((nt, nx), source_value)
        seed = int(gen.get("seed", config.get("seed", 42)))
        results = parallel_samples(
            _generate_linear_sample,
            {"t": t, "x": x, "t_end": T, "source": source, "initial": np.zeros(nx),
             "diffusivity": diffusivity, "save_t": save_t, "save_x": save_x},
            independent_seeds(seed, samples), int(gen.get("workers", 1)),
            description=gen.get("progress_label", "linear_1d"),
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
            handle["x"] = x[save_x]
            handle["t"] = t[save_t]
            handle["source"] = source[np.ix_(save_t, save_x)]
            handle.attrs["case"] = self.name
            handle.attrs["diffusivity"] = diffusivity
            handle.attrs["config_json"] = json.dumps(config, ensure_ascii=False)
        return output_path

    def load(self, path):
        with h5py.File(path, "r") as handle:
            data = CaseData(
                handle["u"][:], {"t": handle["t"][:], "x": handle["x"][:]},
                {key: handle[key][:] for key in ("bL", "bR") if key in handle},
                source=handle["source"][:] if "source" in handle else None,
                metadata=dict(handle.attrs),
            )
        self.validate(data)
        return data

    def evaluate(self, prediction, target, data):
        metrics = super().evaluate(prediction, target, data)
        if len(data.axes["t"]) >= 3 and len(data.axes["x"]) >= 3:
            edge_t = 2 if len(data.axes["t"]) >= 3 else 1
            edge_x = 2 if len(data.axes["x"]) >= 3 else 1
            u_t = np.gradient(prediction, data.axes["t"], axis=1, edge_order=edge_t)
            u_x = np.gradient(prediction, data.axes["x"], axis=2, edge_order=edge_x)
            u_xx = np.gradient(u_x, data.axes["x"], axis=2, edge_order=edge_x)
            diffusivity = float((data.metadata or {}).get("diffusivity", 1.0))
            source = 0.0 if data.source is None else np.asarray(data.source)
            if np.ndim(source) == 2:
                source = source[None]
            metrics.update(residual_statistics((u_t - diffusivity * u_xx - source)[:, 1:, 1:-1]))
        return metrics


register_case(Linear1DCase())
