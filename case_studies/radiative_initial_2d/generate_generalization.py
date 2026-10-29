"""Build the dedicated spatial super-resolution benchmark.

This entry point is intentionally separate from generate.py and does not
participate in training or the existing ID evaluation workflow. Each saved
spatial resolution is produced by a finite-volume solve on that resolution;
reference fields are never obtained by interpolation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import argparse

import h5py
import numpy as np


CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pifdm.cases.acceleration import parallel_samples, report_solver_progress
from pifdm.cases.radiative_2d import (
    build_material,
    build_initial_energy,
    solve_radiative_step_2d,
)


SOLVER_TIME_STEPS = 5120
SOLVER_TIMES = np.linspace(0.0, 1.5, SOLVER_TIME_STEPS, dtype=float)
TRAINING_TIME_INDICES = np.linspace(0, SOLVER_TIME_STEPS - 1, 32, dtype=int)
TRAINING_TARGET_TIMES = SOLVER_TIMES[TRAINING_TIME_INDICES[1:]]
# The initial condition at t=0 is stored separately.  The benchmark keeps the
# same 31 positive-time samples as training and refines only the spatial axes.
TARGET_TIME_INDICES = TRAINING_TIME_INDICES[1:]
TARGET_TIMES = SOLVER_TIMES[TARGET_TIME_INDICES]
BENCHMARK_VERSION = 9

# The 62-point grid reproduces the training data coordinates by solving on the
# original 256-cell grid and selecting the same saved cells. The two
# super-resolution references are solved directly at their reported sizes.
REFERENCE_GRIDS = {
    "training_grid": {
        "label": "training 62x62",
        "solver_spatial_shape": [256, 256],
        "saved_spatial_shape": [62, 62],
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

# Keep the exact training function family and use fixed parameter combinations
# strictly inside the training ranges.  These are interpolation cases, not
# parameter-distribution extrapolation cases.
INTERPOLATION_CASES = (
    {
        "name": "interpolation_01",
        "description": "interior training-range Gaussian parameters, case 1",
        "initial_condition": {
            "type": "gaussian_hotspot", "background": 1e-3,
            "amplitude": 175.0, "sigma": 0.045, "center_x": 0.025, "center_y": 0.035,
        },
    },
    {
        "name": "interpolation_02",
        "description": "interior training-range Gaussian parameters, case 2",
        "initial_condition": {
            "type": "gaussian_hotspot", "background": 1e-3,
            "amplitude": 195.0, "sigma": 0.055, "center_x": 0.055, "center_y": 0.085,
        },
    },
    {
        "name": "interpolation_03",
        "description": "interior training-range Gaussian parameters, case 3",
        "initial_condition": {
            "type": "gaussian_hotspot", "background": 1e-3,
            "amplitude": 215.0, "sigma": 0.065, "center_x": 0.085, "center_y": 0.045,
        },
    },
    {
        "name": "interpolation_04",
        "description": "interior training-range Gaussian parameters, case 4",
        "initial_condition": {
            "type": "gaussian_hotspot", "background": 1e-3,
            "amplitude": 235.0, "sigma": 0.075, "center_x": 0.115, "center_y": 0.105,
        },
    },
    {
        "name": "interpolation_05",
        "description": "interior training-range Gaussian parameters, case 5",
        "initial_condition": {
            "type": "gaussian_hotspot", "background": 1e-3,
            "amplitude": 255.0, "sigma": 0.085, "center_x": 0.135, "center_y": 0.135,
        },
    },
    {
        "name": "interpolation_06",
        "description": "interior training-range Gaussian parameters, case 6",
        "initial_condition": {
            "type": "gaussian_hotspot", "background": 1e-3,
            "amplitude": 280.0, "sigma": 0.050, "center_x": 0.035, "center_y": 0.125,
        },
    },
)

REFERENCE_SETTINGS = {
    "case": "radiative_initial_2d",
    "domain": {"t": [0.0, 1.5], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    "solver_time_steps": SOLVER_TIME_STEPS,
    "grids": REFERENCE_GRIDS,
    "workers": 1,
    "show_progress": True,
    "progress_interval": 30.0,
    "solver": {
        "linear_solver": "pcg", "picard_tolerance": 1e-6,
        "linear_tolerance": 1e-6, "linear_absolute_tolerance": 1e-12,
        "picard_iterations": 2000, "linear_iterations": 500,
    },
    "physics": {
        "use_flux_limiter": True,
        "limiter_norm": "l2_smooth",
        "material": {
            "type": "rectangles", "background": 1.0,
            "regions": [
                {"x": [0.15, 0.35], "y": [0.15, 0.35], "value": 2.5},
                {"x": [0.60, 0.80], "y": [0.60, 0.80], "value": 5.0},
            ],
        },
    },
}

OUTPUT = CASE_DIR / "data" / "superresolution_reference_6cases_31times_62_200_400.h5"


def build_generalization_initial(x: np.ndarray, y: np.ndarray, spec: dict) -> np.ndarray:
    """Evaluate the canonical training Gaussian on arbitrary coordinates."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("initial-condition coordinates must be one-dimensional")
    if spec.get("type") != "gaussian_hotspot":
        raise ValueError("interpolation cases must use the canonical gaussian_hotspot form")
    initial, _ = build_initial_energy(x, y, spec, np.random.default_rng(0))
    if not np.isfinite(initial).all() or np.min(initial) < 0.0:
        raise ValueError("invalid values in interpolation initial condition")
    return initial


def build_reference_grid(grid_name: str, settings: dict = REFERENCE_SETTINGS):
    """Return one solver grid, its direct saved cells, and target steps."""
    domain = settings["domain"]
    if grid_name not in settings["grids"]:
        raise KeyError(f"unknown reference grid {grid_name!r}")
    grid = settings["grids"][grid_name]
    ny, nx = (int(value) for value in grid["solver_spatial_shape"])
    saved_ny, saved_nx = (int(value) for value in grid["saved_spatial_shape"])
    nt = int(settings["solver_time_steps"])
    if min(nx, ny, saved_nx, saved_ny) < 2 or saved_nx > nx or saved_ny > ny:
        raise ValueError(f"invalid solver/save shape for {grid_name}")
    x0, x1 = map(float, domain["x"])
    y0, y1 = map(float, domain["y"])
    t0, t1 = map(float, domain["t"])
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    x = np.linspace(x0 + dx / 2.0, x1 - dx / 2.0, nx)
    y = np.linspace(y0 + dy / 2.0, y1 - dy / 2.0, ny)
    time = np.linspace(t0, t1, nt)
    saved_x_indices = np.linspace(0, nx - 1, saved_nx, dtype=int)
    saved_y_indices = np.linspace(0, ny - 1, saved_ny, dtype=int)
    if len(np.unique(saved_x_indices)) != saved_nx or len(np.unique(saved_y_indices)) != saved_ny:
        raise ValueError(f"saved cells are not unique for {grid_name}")
    saved_x = x[saved_x_indices]
    saved_y = y[saved_y_indices]
    target_indices = TARGET_TIME_INDICES.copy()
    return (
        time, y, x, saved_y, saved_x,
        saved_y_indices, saved_x_indices, target_indices,
    )


def _solve_reference_case(context: dict, scenario_index: int):
    scenario = context["cases"][int(scenario_index)]
    time, y, x = context["time"], context["y"], context["x"]
    initial = build_generalization_initial(x, y, scenario["initial_condition"])
    state = initial.copy()
    target_indices = context["target_indices"]
    target_positions = {int(step): position for position, step in enumerate(target_indices)}
    fields = np.empty(
        (len(target_indices), len(context["saved_y"]), len(context["saved_x"])),
        dtype=np.float32,
    )
    zero_x = np.zeros(len(x))
    zero_y = np.zeros(len(y))
    zero_source = np.zeros_like(state)
    material = context["material"]
    physics = context["settings"]["physics"]
    solver = context["settings"]["solver"]

    for step in range(1, len(time)):
        try:
            state = solve_radiative_step_2d(
                state, x[1] - x[0], y[1] - y[0], time[step] - time[step - 1],
                material, zero_source, zero_y, zero_y, zero_x, zero_x,
                use_flux_limiter=bool(physics["use_flux_limiter"]),
                smooth_limiter=physics["limiter_norm"] == "l2_smooth",
                **solver,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"{context['grid_name']} case {scenario['name']!r} failed "
                f"at time step {step}/{len(time) - 1}"
            ) from error
        if step in target_positions:
            fields[target_positions[step]] = state[np.ix_(
                context["saved_y_indices"], context["saved_x_indices"]
            )]
        report_solver_progress(step, len(time) - 1, phase="time-step")

    return {
        "name": scenario["name"],
        "description": scenario["description"],
        "field": fields,
        "initial": initial[np.ix_(context["saved_y_indices"], context["saved_x_indices"])].astype(np.float32),
    }


def _workers(settings: dict) -> int:
    value = os.environ.get("PIFDM_GENERALIZATION_WORKERS", settings["workers"])
    workers = int(value)
    if workers < 1:
        raise ValueError("PIFDM_GENERALIZATION_WORKERS must be at least 1")
    return workers


def generate_generalization_reference(output: str | Path = OUTPUT,
                                      settings: dict = REFERENCE_SETTINGS) -> Path:
    """Generate all direct-solver reference grids in one self-contained file."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset(
            "scenario_name",
            data=np.asarray(
                [scenario["name"] for scenario in INTERPOLATION_CASES],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        handle["t"] = TARGET_TIMES
        handle.attrs["case"] = settings["case"]
        handle.attrs["benchmark"] = "radiative_initial_2d_spatial_superresolution"
        handle.attrs["config_json"] = json.dumps(settings, ensure_ascii=False)
        handle.attrs["interpolation_cases_json"] = json.dumps(INTERPOLATION_CASES, ensure_ascii=False)
        grids_group = handle.create_group("grids")
        for grid_name, grid_spec in settings["grids"].items():
            (time, y, x, saved_y, saved_x, saved_y_indices,
             saved_x_indices, target_indices) = build_reference_grid(grid_name, settings)
            material = build_material(x, y, settings["physics"]["material"])
            context = {
                "settings": settings, "cases": INTERPOLATION_CASES,
                "grid_name": grid_name, "time": time, "y": y, "x": x,
                "saved_y": saved_y, "saved_x": saved_x,
                "saved_y_indices": saved_y_indices, "saved_x_indices": saved_x_indices,
                "target_indices": target_indices, "material": material,
            }
            results = parallel_samples(
                _solve_reference_case, context, list(range(len(INTERPOLATION_CASES))),
                workers=_workers(settings),
                description=f"radiative_initial_2d {grid_name} reference",
                show_progress=bool(settings["show_progress"]),
                progress_interval=float(settings["progress_interval"]),
            )
            group = grids_group.create_group(grid_name)
            group["u"] = np.asarray([result["field"] for result in results])
            group["initial"] = np.asarray([result["initial"] for result in results])
            zero_side_y = np.zeros(
                (len(results), len(TARGET_TIMES), len(saved_y)), dtype=np.float32,
            )
            zero_side_x = np.zeros(
                (len(results), len(TARGET_TIMES), len(saved_x)), dtype=np.float32,
            )
            group["bL"] = zero_side_y
            group["bR"] = zero_side_y.copy()
            group["bB"] = zero_side_x
            group["bT"] = zero_side_x.copy()
            group["t"] = time[target_indices]
            group["y"] = saved_y
            group["x"] = saved_x
            group["material"] = material[np.ix_(saved_y_indices, saved_x_indices)]
            group["source"] = np.zeros(
                (len(TARGET_TIMES), len(saved_y), len(saved_x)), dtype=np.float32,
            )
            group.attrs["label"] = str(grid_spec["label"])
            group.attrs["solver_spatial_shape"] = json.dumps(grid_spec["solver_spatial_shape"])
            group.attrs["saved_spatial_shape"] = json.dumps(grid_spec["saved_spatial_shape"])
            print(json.dumps({
                "grid": grid_name, "solver_spatial_shape": grid_spec["solver_spatial_shape"],
                "saved_shape": [len(results), len(TARGET_TIMES), len(saved_y), len(saved_x)],
            }), flush=True)
        # Written last: its presence marks a complete file for --check.
        handle.attrs["benchmark_version"] = BENCHMARK_VERSION
    temporary.replace(output)
    print(json.dumps({
        "output": str(output.resolve()), "cases": len(INTERPOLATION_CASES),
        "times": TARGET_TIMES.tolist(),
        "grids": {name: spec["saved_spatial_shape"] for name, spec in settings["grids"].items()},
    }, ensure_ascii=False), flush=True)
    return output


def _dataset_is_finite(dataset: h5py.Dataset) -> bool:
    """Validate large fields one leading-axis slice at a time."""
    if dataset.ndim == 0:
        return bool(np.isfinite(dataset[()]))
    return all(np.isfinite(dataset[index]).all() for index in range(dataset.shape[0]))


def read_string_dataset(dataset: h5py.Dataset) -> list[str]:
    """Read UTF-8 HDF5 strings with both h5py 2.x and 3.x."""
    values = dataset.asstr()[:] if hasattr(dataset, "asstr") else dataset[:]
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def check_reference_file(path: str | Path = OUTPUT) -> tuple[bool, str]:
    """Check whether a complete, current reference file can be reused."""
    path = Path(path)
    if not path.is_file():
        return False, "reference file does not exist"
    expected_names = [scenario["name"] for scenario in INTERPOLATION_CASES]
    try:
        with h5py.File(path, "r") as handle:
            required = ("scenario_name", "t", "grids")
            missing = [name for name in required if name not in handle]
            if missing:
                return False, f"missing datasets: {missing}"
            if int(handle.attrs.get("benchmark_version", -1)) != BENCHMARK_VERSION:
                return False, "benchmark version is missing or stale"
            if handle.attrs.get("case") != REFERENCE_SETTINGS["case"]:
                return False, "reference case does not match the benchmark"
            if handle.attrs.get("benchmark") != "radiative_initial_2d_spatial_superresolution":
                return False, "reference benchmark marker does not match"
            if not np.allclose(handle["t"][:], TARGET_TIMES, rtol=0.0, atol=1e-14):
                return False, "saved times do not match the benchmark"
            if tuple(handle["scenario_name"].shape) != (len(expected_names),):
                return False, "scenario_name has an incompatible shape"
            names = read_string_dataset(handle["scenario_name"])
            if names != expected_names:
                return False, "scenario names do not match the benchmark"
            cases = json.loads(handle.attrs["interpolation_cases_json"])
            if cases != list(INTERPOLATION_CASES):
                return False, "scenario definitions do not match the benchmark"
            if not np.all(np.diff(handle["t"][:]) > 0.0):
                return False, "reference times are not strictly increasing"
            for grid_name, grid_spec in REFERENCE_SETTINGS["grids"].items():
                if grid_name not in handle["grids"]:
                    return False, f"missing reference grid {grid_name}"
                group = handle["grids"][grid_name]
                required_grid = ("u", "initial", "bL", "bR", "bB", "bT", "t", "y", "x", "material", "source")
                missing_grid = [name for name in required_grid if name not in group]
                if missing_grid:
                    return False, f"{grid_name} is missing datasets: {missing_grid}"
                ny, nx = (int(value) for value in grid_spec["saved_spatial_shape"])
                field_shape = (len(expected_names), len(TARGET_TIMES), ny, nx)
                expected_shapes = {
                    "u": field_shape, "initial": (len(expected_names), ny, nx),
                    "bL": field_shape[:2] + (ny,), "bR": field_shape[:2] + (ny,),
                    "bB": field_shape[:2] + (nx,), "bT": field_shape[:2] + (nx,),
                    "t": (len(TARGET_TIMES),), "y": (ny,), "x": (nx,),
                    "material": (ny, nx), "source": (len(TARGET_TIMES), ny, nx),
                }
                for name, shape in expected_shapes.items():
                    if tuple(group[name].shape) != shape:
                        return False, f"{grid_name}/{name} has shape {tuple(group[name].shape)}, expected {shape}"
                if not np.allclose(group["t"][:], TARGET_TIMES, rtol=0.0, atol=1e-14):
                    return False, f"{grid_name} times do not match the benchmark"
                if not all(_dataset_is_finite(group[name]) for name in ("u", "initial", "t", "y", "x")):
                    return False, f"{grid_name} contains non-finite values"
                if not (np.all(np.diff(group["y"][:]) > 0.0)
                        and np.all(np.diff(group["x"][:]) > 0.0)):
                    return False, f"{grid_name} coordinates are not strictly increasing"
    except (OSError, KeyError, TypeError, ValueError) as error:
        return False, f"reference file cannot be validated: {error}"
    return True, "reference file is complete and current"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate the existing reference without solving")
    arguments = parser.parse_args(argv)
    if arguments.check:
        valid, reason = check_reference_file()
        print(json.dumps({"valid": valid, "path": str(OUTPUT.resolve()), "reason": reason}), flush=True)
        return 0 if valid else 1
    generate_generalization_reference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
