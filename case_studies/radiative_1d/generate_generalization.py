"""Generate direct-solver spatial super-resolution references for radiative_1d.

This benchmark is intentionally independent from ``generate.py``.  It keeps
the training time snapshots and boundary/material convention, while solving
each requested spatial grid directly instead of interpolating a reference.
"""

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

from case_studies.radiative_1d.generate import CASE_CONFIG
from pifdm.cases.acceleration import parallel_samples
from pifdm.cases.radiative_1d import _face_material, _source_grid, solve_radiative_1d


OUTPUT = CASE_DIR / "data" / "spatial_superresolution_reference.h5"
BENCHMARK_VERSION = 1


def _save_indices(size: int, count: int) -> np.ndarray:
    return np.linspace(0, size - 1, count, dtype=int)


def _boundary_from_spec(t: np.ndarray, spec: dict) -> np.ndarray:
    """Build a deterministic boundary from the same four training families."""
    duration = max(float(t[-1] - t[0]), np.finfo(float).eps)
    tau = (t - t[0]) / duration
    mode = spec["mode"]
    amplitude = float(spec["amplitude"])
    if mode == "multi_pulse":
        result = np.zeros_like(tau)
        for center, width, scale in spec["pulses"]:
            result += scale * amplitude * np.exp(
                -0.5 * ((tau - center) / width) ** 2
            )
    elif mode == "step_hold":
        start, sharpness = spec["start"], spec["sharpness"]
        result = amplitude / (1.0 + np.exp(-sharpness * (tau - start)))
        result *= 1.0 + 0.3 * np.exp(-20.0 * np.maximum(0.0, tau - start))
    elif mode == "ramp_decay":
        peak, decay = spec["peak"], spec["decay"]
        result = np.where(
            tau <= peak, amplitude * tau / peak,
            amplitude * np.exp(-decay * (tau - peak)),
        )
    elif mode == "random_fluctuation":
        rng = np.random.default_rng(int(spec["seed"]))
        noise = rng.standard_normal(len(t))
        width = max(1, int(0.05 * len(t)))
        noise = np.convolve(noise, np.ones(width) / width, mode="same")
        result = amplitude * (noise - noise.min()) / (np.ptp(noise) + 1e-12)
    else:
        raise ValueError(f"unknown benchmark boundary mode {mode!r}")
    return np.maximum(result, 1e-8)


# All amplitudes stay within the training generator's [2, 10] amplitude range.
# The cases differ in boundary shape and parameters but are not amplitude OOD.
CASES = (
    {"name": "multi_pulse_a", "mode": "multi_pulse", "amplitude": 3.2,
     "pulses": ((0.18, 0.035, 0.85), (0.57, 0.045, 0.70))},
    {"name": "multi_pulse_b", "mode": "multi_pulse", "amplitude": 4.6,
     "pulses": ((0.34, 0.028, 0.55), (0.78, 0.052, 0.65))},
    {"name": "step_hold_a", "mode": "step_hold", "amplitude": 4.8,
     "start": 0.12, "sharpness": 90.0},
    {"name": "step_hold_b", "mode": "step_hold", "amplitude": 6.2,
     "start": 0.27, "sharpness": 150.0},
    {"name": "ramp_decay_a", "mode": "ramp_decay", "amplitude": 5.4,
     "peak": 0.16, "decay": 2.0},
    {"name": "random_fluctuation_a", "mode": "random_fluctuation", "amplitude": 4.2,
     "seed": 1701},
)


def _settings() -> dict:
    generation = CASE_CONFIG["generation"]
    domain = CASE_CONFIG["domain"]
    grids = {
        "training_grid": {
            "label": "training grid (126 saved interior points)",
            "solver_nx": int(generation["nx"]),
            "saved_nx": int(generation["nx_save"]),
            "save_all": False,
        },
        "superresolution_256": {
            "label": "direct solver (256 interior points)",
            "solver_nx": 256, "saved_nx": 256, "save_all": True,
        },
        "superresolution_512": {
            "label": "direct solver (512 interior points)",
            "solver_nx": 512, "saved_nx": 512, "save_all": True,
        },
    }
    return {
        "case": CASE_CONFIG["case"],
        "domain": domain,
        "generation": {
            "nx": int(generation["nx"]), "nt": int(generation["nt"]),
            "nx_save": int(generation["nx_save"]), "nt_save": int(generation["nt_save"]),
            "initial_value": float(generation["initial_value"]),
            "right_boundary": float(generation["right_boundary"]),
        },
        "physics": CASE_CONFIG["physics"],
        "grids": grids,
        "cases": CASES,
    }


def _solve_case(settings: dict, case: dict, grid: dict) -> dict[str, np.ndarray]:
    x0, x1 = settings["domain"]["x"]
    t0, t1 = settings["domain"]["t"]
    nx = int(grid["solver_nx"])
    nt = int(settings["generation"]["nt"])
    dx = (x1 - x0) / nx
    x = np.linspace(x0 + dx / 2.0, x1 - dx / 2.0, nx)
    t = np.linspace(t0, t1, nt)
    save_t = _save_indices(nt, int(settings["generation"]["nt_save"]))
    if grid["save_all"]:
        save_x = np.arange(nx, dtype=int)
    else:
        save_x = _save_indices(nx, int(grid["saved_nx"]))
    material_spec = settings["physics"]["material"]
    material_faces = _face_material(np.linspace(x0, x1, nx + 1), material_spec)
    material_cells = _face_material(x, material_spec)
    source = _source_grid(t, x, settings["physics"]["source"])
    initial = np.full(nx, settings["generation"]["initial_value"], dtype=float)
    left = _boundary_from_spec(t, case)
    right = np.full(nt, settings["generation"]["right_boundary"], dtype=float)
    solution = solve_radiative_1d(
        initial, t, x, material_faces, source, left, right,
        use_flux_limiter=bool(settings["physics"]["use_flux_limiter"]),
        smooth_limiter=settings["physics"]["limiter_norm"] == "l2_smooth",
        save_indices=save_t,
    )
    return {
        "u": solution[:, save_x].astype(np.float32),
        "initial": initial[save_x].astype(np.float32),
        "bL": left[save_t].astype(np.float32),
        "bR": right[save_t].astype(np.float32),
        "t": t[save_t].astype(np.float64),
        "x": x[save_x].astype(np.float64),
        "material": material_cells[save_x].astype(np.float32),
        "source": source[np.ix_(save_t, save_x)].astype(np.float32),
    }


def _solve_generalization_case(context: dict, case_index: int) -> dict[str, np.ndarray]:
    """Worker entry point for one deterministic benchmark case."""
    return _solve_case(
        context["settings"], context["cases"][int(case_index)], context["grid"]
    )


def _read_strings(dataset: h5py.Dataset) -> list[str]:
    values = dataset.asstr()[:] if hasattr(dataset, "asstr") else dataset[:]
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _attribute_text(value) -> str:
    """Normalize HDF5 string attributes across old and new h5py versions."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def generate_reference(output: str | Path = OUTPUT) -> Path:
    settings = _settings()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    workers = int(os.environ.get("PIFDM_GENERALIZATION_WORKERS", "1"))
    if workers < 1:
        raise ValueError("PIFDM_GENERALIZATION_WORKERS must be at least 1")
    progress_interval = float(
        os.environ.get("PIFDM_GENERALIZATION_PROGRESS_INTERVAL", "30.0")
    )
    if progress_interval <= 0.0:
        raise ValueError("PIFDM_GENERALIZATION_PROGRESS_INTERVAL must be positive")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle["scenario_name"] = np.asarray([case["name"] for case in CASES], dtype="S")
        handle.attrs["case"] = settings["case"]
        handle.attrs["benchmark"] = "radiative_1d_spatial_superresolution"
        handle.attrs["benchmark_version"] = BENCHMARK_VERSION
        handle.attrs["config_json"] = json.dumps(settings, ensure_ascii=False)
        handle.attrs["cases_json"] = json.dumps(CASES, ensure_ascii=False)
        grids = handle.create_group("grids")
        for grid_name, grid_spec in settings["grids"].items():
            print(
                f"[radiative_1d reference] grid={grid_name} solver_nx="
                f"{grid_spec['solver_nx']} saved_nx={grid_spec['saved_nx']} "
                f"cases={len(CASES)} workers={min(workers, len(CASES))}",
                flush=True,
            )
            results = parallel_samples(
                _solve_generalization_case,
                {"settings": settings, "cases": CASES, "grid": grid_spec},
                list(range(len(CASES))), workers,
                description=f"radiative_1d {grid_name}",
                show_progress=True,
                progress_interval=progress_interval,
            )
            group = grids.create_group(grid_name)
            for key in ("u", "initial", "bL", "bR", "t", "x", "material", "source"):
                group[key] = np.asarray([result[key] for result in results]) if key in ("u", "initial", "bL", "bR") else results[0][key]
            group.attrs["label"] = grid_spec["label"]
            group.attrs["solver_nx"] = int(grid_spec["solver_nx"])
            group.attrs["saved_nx"] = int(grid_spec["saved_nx"])
            print(json.dumps({
                "grid": grid_name, "cases": len(CASES),
                "shape": list(group["u"].shape),
            }), flush=True)
    temporary.replace(output)
    print(f"reference: {output.resolve()}", flush=True)
    return output


def check_reference_file(path: str | Path = OUTPUT) -> tuple[bool, str]:
    settings = _settings()
    path = Path(path)
    if not path.is_file():
        return False, "reference file does not exist"
    try:
        with h5py.File(path, "r") as handle:
            if int(handle.attrs.get("benchmark_version", -1)) != BENCHMARK_VERSION:
                return False, "benchmark version is missing or stale"
            if _attribute_text(handle.attrs.get("benchmark", "")) != "radiative_1d_spatial_superresolution":
                return False, "benchmark marker does not match"
            expected_config = json.loads(json.dumps(settings))
            if json.loads(handle.attrs.get("config_json", "{}")) != expected_config:
                return False, "reference configuration does not match"
            if _read_strings(handle["scenario_name"]) != [case["name"] for case in CASES]:
                return False, "scenario names do not match"
            expected_cases = json.loads(json.dumps(CASES))
            if json.loads(handle.attrs.get("cases_json", "[]")) != expected_cases:
                return False, "scenario definitions do not match"
            for name, spec in settings["grids"].items():
                if name not in handle["grids"]:
                    return False, f"missing grid {name}"
                group = handle["grids"][name]
                nx = int(spec["saved_nx"])
                expected = {
                    "u": (len(CASES), settings["generation"]["nt_save"], nx),
                    "initial": (len(CASES), nx), "bL": (len(CASES), settings["generation"]["nt_save"]),
                    "bR": (len(CASES), settings["generation"]["nt_save"]),
                    "t": (settings["generation"]["nt_save"],), "x": (nx,),
                    "material": (nx,), "source": (settings["generation"]["nt_save"], nx),
                }
                for key, shape in expected.items():
                    if key not in group or tuple(group[key].shape) != shape:
                        return False, f"{name}/{key} has an unexpected shape"
                    if not np.isfinite(group[key][:]).all():
                        return False, f"{name}/{key} contains non-finite values"
                if not np.all(np.diff(group["x"][:]) > 0.0):
                    return False, f"{name}/x is not strictly increasing"
                x0, x1 = settings["domain"]["x"]
                dx = (x1 - x0) / int(spec["solver_nx"])
                fine_x = np.linspace(x0 + dx / 2.0, x1 - dx / 2.0, int(spec["solver_nx"]))
                expected_x = fine_x[
                    np.arange(int(spec["solver_nx"]))
                    if spec["save_all"] else
                    _save_indices(int(spec["solver_nx"]), int(spec["saved_nx"]))
                ]
                if not np.allclose(group["x"][:], expected_x, rtol=0.0, atol=1e-14):
                    return False, f"{name}/x does not match the direct solver grid"
                expected_t = np.linspace(
                    settings["domain"]["t"][0], settings["domain"]["t"][1],
                    settings["generation"]["nt"],
                )[_save_indices(
                    settings["generation"]["nt"], settings["generation"]["nt_save"]
                )]
                if not np.allclose(group["t"][:], expected_t, rtol=0.0, atol=1e-14):
                    return False, f"{name}/t does not match the direct solver snapshots"
    except (OSError, KeyError, TypeError, ValueError) as error:
        return False, f"reference cannot be validated: {error}"
    return True, "reference file is complete and current"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without solving")
    args = parser.parse_args(argv)
    if args.check:
        valid, reason = check_reference_file()
        print(json.dumps({"valid": valid, "path": str(OUTPUT.resolve()), "reason": reason}), flush=True)
        return 0 if valid else 1
    generate_reference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
