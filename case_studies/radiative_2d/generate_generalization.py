"""Generate direct finite-volume spatial-superresolution references for radiative_2d."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_studies.radiative_2d.generate import CASE_CONFIG
from pifdm.cases.acceleration import parallel_samples, report_solver_progress
from pifdm.cases.radiative_2d import (
    build_material,
    build_source,
    evaluate_legacy_left_boundary,
    evaluate_left_boundary,
    solve_radiative_step_2d,
)


CASE = "radiative_2d"
DOMAIN = CASE_CONFIG["domain"]
GENERATION_SETTINGS = CASE_CONFIG["generation"]
SOLVER_TIME_STEPS = int(GENERATION_SETTINGS["nt"])
TRAINING_SAVE_COUNT = int(GENERATION_SETTINGS["nt_save"])
TRAINING_TIME_INDICES = np.linspace(
    0, SOLVER_TIME_STEPS - 1, TRAINING_SAVE_COUNT, dtype=int,
)
# Keep the initial slice in this benchmark.  It is a frozen known condition in
# the model, and retaining it makes the reference tensor have the same time
# layout as the regenerated radiative_2d training data.
TARGET_TIME_INDICES = TRAINING_TIME_INDICES
TARGET_TIMES = np.linspace(DOMAIN["t"][0], DOMAIN["t"][1], SOLVER_TIME_STEPS)[TARGET_TIME_INDICES]
BENCHMARK_VERSION = 3

REFERENCE_GRIDS = {
    "training_grid": {
        "label": "training 64x64",
        "solver_spatial_shape": [
            int(GENERATION_SETTINGS["ny"]), int(GENERATION_SETTINGS["nx"]),
        ],
        "saved_spatial_shape": [
            int(GENERATION_SETTINGS["ny_save"]), int(GENERATION_SETTINGS["nx_save"]),
        ],
    },
    "superresolution_200": {
        "label": "spatial super-resolution 200x200",
        "solver_spatial_shape": [200, 200],
        "saved_spatial_shape": [200, 200],
    },
    "superresolution_400": {
        "label": "spatial super-resolution 400x400",
        "solver_spatial_shape": [400, 400],
        "saved_spatial_shape": [400, 400],
    },
}

# These cases keep the exact sustained left-boundary function family and stay
# strictly inside the training ranges. Only the parameter combinations differ.
INTERPOLATION_CASES = (
    {"name": "interpolation_01", "description": "moderate central sustained inflow", "boundary":
     {"mode": "sustained", "amplitude": 140.0, "center": 0.36, "radius": 0.24,
      "rise": 0.08, "width": 0.07}},
    {"name": "interpolation_02", "description": "broad lower sustained inflow", "boundary":
     {"mode": "sustained", "amplitude": 175.0, "center": 0.43, "radius": 0.32,
      "rise": 0.12, "width": 0.10}},
    {"name": "interpolation_03", "description": "narrow upper sustained inflow", "boundary":
     {"mode": "sustained", "amplitude": 210.0, "center": 0.57, "radius": 0.26,
      "rise": 0.16, "width": 0.12}},
    {"name": "interpolation_04", "description": "strong broad central inflow", "boundary":
     {"mode": "sustained", "amplitude": 245.0, "center": 0.64, "radius": 0.37,
      "rise": 0.10, "width": 0.08}},
    {"name": "interpolation_05", "description": "upper moderate inflow", "boundary":
     {"mode": "sustained", "amplitude": 280.0, "center": 0.49, "radius": 0.29,
      "rise": 0.06, "width": 0.14}},
    {"name": "interpolation_06", "description": "lower strong inflow", "boundary":
     {"mode": "sustained", "amplitude": 225.0, "center": 0.33, "radius": 0.35,
      "rise": 0.18, "width": 0.09}},
)

REFERENCE_SETTINGS = {
    "case": CASE,
    "domain": DOMAIN,
    "solver_time_steps": SOLVER_TIME_STEPS,
    "grids": REFERENCE_GRIDS,
    "workers": 1,
    "show_progress": True,
    "progress_interval": 30.0,
    "physics": {
        **CASE_CONFIG["physics"],
    },
    "solver": GENERATION_SETTINGS["solver"],
    "boundary_mode": GENERATION_SETTINGS["boundary_mode"],
    "initial_value": float(GENERATION_SETTINGS["initial_value"]),
}
OUTPUT = CASE_DIR / "data" / "superresolution_reference_6cases_16times_64_200_400.h5"
INITIAL_CONDITION = {"type": "constant", "value": REFERENCE_SETTINGS["initial_value"]}


def build_generalization_initial(x: np.ndarray, y: np.ndarray, spec: dict) -> np.ndarray:
    """Return the constant initial field used by the radiative_2d dataset."""
    if spec.get("type", "constant") != "constant":
        raise ValueError("radiative_2d superresolution cases require a constant initial field")
    value = float(spec.get("value", REFERENCE_SETTINGS["initial_value"]))
    return np.full((len(y), len(x)), value, dtype=float)


def build_generalization_boundaries(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                                    spec: dict) -> dict[str, np.ndarray]:
    """Evaluate fixed boundary parameters with the training data's implementation."""
    mode = str(spec.get("mode", REFERENCE_SETTINGS["boundary_mode"]))
    if mode != REFERENCE_SETTINGS["boundary_mode"]:
        raise ValueError(
            f"benchmark boundary mode {mode!r} does not match training mode "
            f"{REFERENCE_SETTINGS['boundary_mode']!r}"
        )
    boundary_kwargs = {
        "amplitude": float(spec["amplitude"]), "center": float(spec["center"]),
        "radius": float(spec["radius"]), "mode": mode,
        "rise": (None if "rise" not in spec else float(spec["rise"])),
        "width": (None if "width" not in spec else float(spec["width"])),
        "pulse_center": (None if "pulse_center" not in spec else float(spec["pulse_center"])),
        "pulse_width": (None if "pulse_width" not in spec else float(spec["pulse_width"])),
    }
    if GENERATION_SETTINGS.get("legacy_boundary_formula", False):
        # Evaluate at the actual finite-volume cell centres.  The saved gL
        # values can then be selected by index without a spatial interpolation.
        left = evaluate_legacy_left_boundary(t, y, **boundary_kwargs)
    else:
        left = evaluate_left_boundary(t, y, **boundary_kwargs)
    return {"gL": left}


def build_reference_grid(grid_name: str):
    grid = REFERENCE_GRIDS[grid_name]
    ny, nx = (int(value) for value in grid["solver_spatial_shape"])
    saved_ny, saved_nx = (int(value) for value in grid["saved_spatial_shape"])
    nt = int(SOLVER_TIME_STEPS)
    x0, x1 = DOMAIN["x"]
    y0, y1 = DOMAIN["y"]
    t0, t1 = DOMAIN["t"]
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    x = np.linspace(x0 + dx / 2.0, x1 - dx / 2.0, nx)
    y = np.linspace(y0 + dy / 2.0, y1 - dy / 2.0, ny)
    t = np.linspace(t0, t1, nt)
    saved_x_indices = np.linspace(0, nx - 1, saved_nx, dtype=int)
    saved_y_indices = np.linspace(0, ny - 1, saved_ny, dtype=int)
    return t, y, x, saved_y_indices, saved_x_indices


def _solve_reference_case(context: dict, scenario_index: int) -> dict:
    scenario = INTERPOLATION_CASES[int(scenario_index)]
    t, y, x = context["t"], context["y"], context["x"]
    initial = build_generalization_initial(x, y, INITIAL_CONDITION)
    boundaries = build_generalization_boundaries(t, x, y, scenario["boundary"])
    state = initial.copy()
    positions = {int(step): index for index, step in enumerate(TARGET_TIME_INDICES)}
    saved_y, saved_x = context["saved_y"], context["saved_x"]
    fields = np.empty((len(TARGET_TIME_INDICES), len(saved_y), len(saved_x)), dtype=np.float32)
    saved_g_left = np.empty(
        (len(TARGET_TIME_INDICES), len(saved_y)), dtype=np.float32
    )
    initial_position = positions.get(0)
    if initial_position is None:
        raise RuntimeError("the super-resolution reference must contain t=0")
    fields[initial_position] = initial[np.ix_(saved_y, saved_x)]
    saved_g_left[initial_position] = boundaries["gL"][0, saved_y]
    material = context["material"]
    use_flux_limiter = bool(REFERENCE_SETTINGS["physics"].get("use_flux_limiter", True))
    smooth_limiter = GENERATION_SETTINGS.get(
        "limiter_norm", REFERENCE_SETTINGS["physics"].get("limiter_norm", "l1")
    ) == "l2_smooth"
    for step in range(1, len(t)):
        step_args = (
            state, x[1] - x[0], y[1] - y[0], t[step] - t[step - 1],
            material, context["source"][step], boundaries["gL"][step],
            np.zeros(len(y), dtype=float), np.zeros(len(x), dtype=float),
            np.zeros(len(x), dtype=float),
        )
        solver = dict(REFERENCE_SETTINGS["solver"])
        try:
            state = solve_radiative_step_2d(
                *step_args, use_flux_limiter=use_flux_limiter,
                smooth_limiter=smooth_limiter, **solver,
            )
        except RuntimeError as error:
            # A finer spatial grid can make the nonlinear Picard map more
            # oscillatory even though the underlying implicit FV problem is
            # unchanged. Retry this time step from the same old state with
            # fixed-point damping; successful steps use the normal solver.
            retry_solver = dict(solver)
            retry_solver["picard_relaxation"] = 0.5
            retry_solver["picard_iterations"] = max(
                int(retry_solver.get("picard_iterations", 2000)), 5000
            )
            try:
                state = solve_radiative_step_2d(
                    *step_args, use_flux_limiter=use_flux_limiter,
                    smooth_limiter=smooth_limiter, **retry_solver,
                )
            except RuntimeError:
                raise error
        report_solver_progress(step, len(t) - 1, phase="time-step")
        if step in positions:
            position = positions[step]
            fields[position] = state[np.ix_(saved_y, saved_x)]
            saved_g_left[position] = boundaries["gL"][step, saved_y]
    return {
        "field": fields,
        "initial": initial[np.ix_(saved_y, saved_x)].astype(np.float32),
        "g_left": saved_g_left,
    }


def _read_strings(dataset: h5py.Dataset) -> list[str]:
    values = dataset.asstr()[:] if hasattr(dataset, "asstr") else dataset[:]
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _text_attribute(value) -> str:
    """Normalize scalar HDF5 text attributes across h5py 2.x and 3.x."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _workers() -> int:
    return max(1, int(os.environ.get("PIFDM_GENERALIZATION_WORKERS", "1")))


def generate_reference(output: str | Path = OUTPUT) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("scenario_name", data=np.asarray(
            [case["name"] for case in INTERPOLATION_CASES], dtype=h5py.string_dtype("utf-8")))
        handle["t"] = TARGET_TIMES
        handle.attrs["case"] = CASE
        handle.attrs["benchmark"] = "radiative_2d_spatial_superresolution"
        handle.attrs["benchmark_version"] = BENCHMARK_VERSION
        handle.attrs["interpolation_cases_json"] = json.dumps(INTERPOLATION_CASES)
        groups = handle.create_group("grids")
        for grid_name, grid_spec in REFERENCE_GRIDS.items():
            t, y, x, saved_y_indices, saved_x_indices = build_reference_grid(grid_name)
            saved_y, saved_x = y[saved_y_indices], x[saved_x_indices]
            context = {"t": t, "y": y, "x": x, "saved_y": saved_y_indices,
                       "saved_x": saved_x_indices,
                       "material": build_material(x, y, REFERENCE_SETTINGS["physics"]["material"]),
                       "source": build_source(t, x, y, REFERENCE_SETTINGS["physics"].get("source", {}))}
            results = parallel_samples(
                _solve_reference_case, context, list(range(len(INTERPOLATION_CASES))),
                workers=_workers(), description=f"{CASE} {grid_name} reference",
                show_progress=True, progress_interval=30.0,
            )
            group = groups.create_group(grid_name)
            group["u"] = np.asarray([result["field"] for result in results])
            group["initial"] = np.asarray([result["initial"] for result in results])
            group["gL"] = np.asarray([result["g_left"] for result in results])
            group["t"], group["y"], group["x"] = TARGET_TIMES, saved_y, saved_x
            group["material"] = context["material"][np.ix_(saved_y_indices, saved_x_indices)]
            group["source"] = context["source"][np.ix_(TARGET_TIME_INDICES, saved_y_indices, saved_x_indices)]
            group.attrs["label"] = grid_spec["label"]
            group.attrs["solver_spatial_shape"] = json.dumps(grid_spec["solver_spatial_shape"])
            group.attrs["saved_spatial_shape"] = json.dumps(grid_spec["saved_spatial_shape"])
    temporary.replace(output)
    return output


def check_reference_file(path: str | Path = OUTPUT) -> tuple[bool, str]:
    path = Path(path)
    if not path.is_file():
        return False, "reference file does not exist"
    try:
        with h5py.File(path, "r") as handle:
            required_root = ("scenario_name", "t", "grids")
            missing_root = [name for name in required_root if name not in handle]
            if missing_root:
                return False, f"missing root datasets or groups: {missing_root}"
            if int(handle.attrs.get("benchmark_version", -1)) != BENCHMARK_VERSION:
                return False, "benchmark version is missing or stale"
            if (
                _text_attribute(handle.attrs.get("case", "")) != CASE
                or _text_attribute(handle.attrs.get("benchmark", ""))
                != "radiative_2d_spatial_superresolution"
            ):
                return False, "reference marker does not match"
            if _read_strings(handle["scenario_name"]) != [case["name"] for case in INTERPOLATION_CASES]:
                return False, "scenario names do not match"
            root_times = np.asarray(handle["t"][:])
            if root_times.shape != TARGET_TIMES.shape or not np.allclose(
                root_times, TARGET_TIMES, rtol=0.0, atol=1e-14,
            ):
                return False, "reference times do not match"
            for name, spec in REFERENCE_GRIDS.items():
                if name not in handle["grids"]:
                    return False, f"missing grid {name}"
                group = handle["grids"][name]
                ny, nx = spec["saved_spatial_shape"]
                required_fields = ("u", "initial", "gL", "t", "y", "x",
                                   "material", "source")
                missing_fields = [field for field in required_fields if field not in group]
                if missing_fields:
                    return False, f"{name} is missing datasets: {missing_fields}"
                expected_shapes = {
                    "u": (len(INTERPOLATION_CASES), len(TARGET_TIMES), ny, nx),
                    "initial": (len(INTERPOLATION_CASES), ny, nx),
                    "gL": (len(INTERPOLATION_CASES), len(TARGET_TIMES), ny),
                    "t": (len(TARGET_TIMES),), "y": (ny,), "x": (nx,),
                    "material": (ny, nx),
                    "source": (len(TARGET_TIMES), ny, nx),
                }
                for field, expected_shape in expected_shapes.items():
                    if tuple(group[field].shape) != expected_shape:
                        return False, f"{name}/{field} shape mismatch"
                _, full_y, full_x, saved_y_indices, saved_x_indices = build_reference_grid(name)
                expected_y = full_y[saved_y_indices]
                expected_x = full_x[saved_x_indices]
                for field, expected_values in (
                    ("t", TARGET_TIMES), ("y", expected_y), ("x", expected_x),
                ):
                    values = np.asarray(group[field][:])
                    if not np.allclose(values, expected_values, rtol=0.0, atol=1e-14):
                        return False, f"{name}/{field} coordinates do not match the current benchmark"
                numeric_fields = tuple(field for field in required_fields if field not in ("t", "y", "x"))
                if not all(np.isfinite(group[field][:]).all() for field in numeric_fields):
                    return False, f"{name} contains non-finite values"
                if not np.allclose(
                    group["material"][:],
                    build_material(expected_x, expected_y, REFERENCE_SETTINGS["physics"]["material"]),
                    rtol=0.0, atol=1e-12,
                ):
                    return False, f"{name}/material does not match the current benchmark"
                full_t, full_y, full_x, saved_y_indices, saved_x_indices = build_reference_grid(name)
                expected_source = build_source(
                    full_t, full_x, full_y,
                    REFERENCE_SETTINGS["physics"].get("source", {}),
                )[np.ix_(TARGET_TIME_INDICES, saved_y_indices, saved_x_indices)]
                if not np.allclose(group["source"][:], expected_source, rtol=0.0, atol=1e-12):
                    return False, f"{name}/source does not match the current benchmark"
                for case_index, scenario in enumerate(INTERPOLATION_CASES):
                    expected_boundary = build_generalization_boundaries(
                        full_t, full_x, full_y, scenario["boundary"]
                    )["gL"][np.ix_(TARGET_TIME_INDICES, saved_y_indices)]
                    # ``gL`` is intentionally stored as float32 together
                    # with the direct-solver field.  Validate against the
                    # exact value that the current generator would write,
                    # rather than comparing that value to a float64 formula
                    # with an unattainable 1e-12 tolerance.
                    stored_boundary = np.asarray(group["gL"][case_index])
                    expected_stored = expected_boundary.astype(
                        stored_boundary.dtype, copy=False
                    )
                    if not np.array_equal(stored_boundary, expected_stored):
                        return False, f"{name}/gL does not match the current benchmark"
    except (OSError, KeyError, TypeError, ValueError) as error:
        return False, str(error)
    return True, "reference file is complete and current"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        valid, reason = check_reference_file()
        print(json.dumps({"valid": valid, "path": str(OUTPUT.resolve()), "reason": reason}), flush=True)
        return 0 if valid else 1
    generate_reference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
