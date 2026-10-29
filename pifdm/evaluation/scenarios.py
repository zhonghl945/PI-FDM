"""Named out-of-distribution boundary scenarios independent of model code."""

from __future__ import annotations

import numpy as np

from pifdm.cases.base import CaseData
from pifdm.cases.linear_1d import solve_linear
from pifdm.cases.radiative_1d import _face_material, _source_grid, solve_radiative_1d


OOD_SCENARIOS = ("high_peak", "square_wave", "high_freq", "step_ramp", "double_gauss")


def boundary_scenario(name: str, t: np.ndarray, *, radiative: bool) -> np.ndarray:
    tau = (t - t[0]) / max(t[-1] - t[0], 1e-12)
    if radiative:
        if name == "high_peak": value = 9.0 * np.exp(-0.5 * ((tau - 0.2) / 0.03) ** 2)
        elif name == "square_wave": value = np.where((tau > 0.1) & (tau < 0.4), 7.0, 1.0)
        elif name == "high_freq": value = 4.0 * (1.0 + np.sin(20.0 * tau)) * np.exp(-2.0 * tau)
        elif name == "step_ramp": value = np.clip(9.0 * tau, 0.0, 9.5)
        elif name == "double_gauss": value = 5.0 * (np.exp(-((tau - 0.2) / 0.04) ** 2) + np.exp(-((tau - 0.6) / 0.04) ** 2))
        else: raise ValueError(f"unknown OOD scenario {name!r}; expected {OOD_SCENARIOS}")
    else:
        if name == "high_peak": value = 3.0 * np.exp(-0.5 * ((tau - 0.2) / 0.03) ** 2)
        elif name == "square_wave": value = np.where((tau > 0.1) & (tau < 0.4), 1.5, 0.0)
        elif name == "high_freq": value = 0.5 + 0.5 * np.sin(20.0 * tau)
        elif name == "step_ramp": value = np.clip(3.0 * tau, 0.0, 2.0)
        elif name == "double_gauss": value = 1.5 * (np.exp(-((tau - 0.2) / 0.04) ** 2) + np.exp(-((tau - 0.6) / 0.04) ** 2))
        else: raise ValueError(f"unknown OOD scenario {name!r}; expected {OOD_SCENARIOS}")
    value[0] = 1e-8 if radiative else 0.0
    return value


def build_ood_case(config: dict, name: str) -> CaseData:
    """Generate one 1D reference solution using the migrated finite-volume solver."""
    case = config["case"]
    if case not in ("linear_1d", "radiative_1d"):
        raise ValueError("named OOD boundary scenarios currently apply to the two 1D cases")
    generation = config.get("generation", {})
    nx, nt = int(generation.get("nx", 512)), int(generation.get("nt", 5120))
    packed_shape = config.get("grid_shape")
    if packed_shape is not None:
        nt_save, nx_save = int(packed_shape[0]), int(packed_shape[1]) - 2
    else:
        nx_save, nt_save = int(generation.get("nx_save", 126)), int(generation.get("nt_save", 64))
    x0, x1 = config["domain"]["x"]
    t0, t1 = config["domain"]["t"]
    dx = (x1 - x0) / nx
    x = np.linspace(x0 + dx / 2, x1 - dx / 2, nx)
    t = np.linspace(t0, t1, nt)
    ix = np.linspace(0, nx - 1, nx_save, dtype=int)
    it = np.linspace(0, nt - 1, nt_save, dtype=int)
    physics = config.get("physics", {})
    radiative = case == "radiative_1d"
    left = boundary_scenario(name, t, radiative=radiative)
    right = np.full(nt, float(generation.get("right_boundary", 0.0)))
    if radiative:
        material_faces = _face_material(np.linspace(x0, x1, nx + 1), physics.get("material", {}))
        material = _face_material(x, physics.get("material", {}))
        source = _source_grid(t, x, physics.get("source", {}))
        initial = np.full(nx, float(generation.get("initial_value", 1e-3)))
        field = solve_radiative_1d(
            initial, t, x, material_faces, source, left, right,
            use_flux_limiter=physics.get("use_flux_limiter", True),
            smooth_limiter=physics.get("limiter_norm", "l2_smooth") == "l2_smooth", save_indices=it,
        )[:, ix]
        material = material[ix]
    else:
        source = np.full((nt, nx), float(physics.get("source", {}).get("value", 0.0)))
        field = solve_linear(np.zeros(nx), t, x, source, left, right,
                             diffusivity=float(physics.get("diffusivity", 1.0)), save_indices=it)[:, ix]
        material = None
    return CaseData(
        fields=field[None], axes={"t": t[it], "x": x[ix]},
        boundaries={"bL": left[it][None], "bR": right[it][None]},
        material=material, source=source[np.ix_(it, ix)],
        metadata={"case": case, "scenario": name, "diffusivity": float(physics.get("diffusivity", 1.0))},
    )
