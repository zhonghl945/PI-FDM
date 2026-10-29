"""Evaluate radiative_1d spatial super-resolution without touching ID/OOD runs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_studies.radiative_1d.generate_generalization import OUTPUT as REFERENCE_FILE
from case_studies.radiative_1d.generate_generalization import CASES, check_reference_file
from case_studies.radiative_1d.train import output_directory
from pifdm.core.diffusion import GaussianDiffusion
from pifdm.evaluation.residual_diagnostics import capture_last_continuous_latent
from pifdm.experiments.naming import canonical_method_name
from pifdm.experiments.physics import build_material_inputs
from pifdm.experiments.runner import build_experiment
from pifdm.physics.coordinates import CoordinateSystem
from pifdm.workflows.training import choose_device


METHODS = (
    "pifdm_unet", "c_fdm_unet",
    "deeponet", "pideeponet", "fno", "pino", "fdm_unet_bicubic",
)
METHOD_LABELS = {
    "pifdm_vit": "PI-FDM-ViT", "c_fdm_vit": "C-FDM-ViT",
    "pifdm_unet": "PI-FDM-UNet", "c_fdm_unet": "C-FDM-UNet",
    "deeponet": "DeepONet", "pideeponet": "PI-DeepONet",
    "fno": "FNO", "pino": "PINO",
    "fdm_unet_bicubic": "FDM-UNet + cubic interpolation",
    "fdm_vit_bicubic": "FDM-ViT + cubic interpolation",
}
INTERPOLATION_BASE_METHODS = {
    "fdm_unet_bicubic": "fdm_unet",
    "fdm_vit_bicubic": "fdm_vit",
}
GRID_NAMES = ("training_grid", "superresolution_256", "superresolution_512")
GRID_LABELS = {
    "training_grid": "126 points",
    "superresolution_256": "256 points",
    "superresolution_512": "512 points",
}
COMPARISON_OUTPUT = CASE_DIR / "runs" / "spatial_superresolution_comparison"
SETTINGS = {
    "device": "auto",
    "query_chunk_size": 131072,
    "visualization_times": [0.1, 0.7, 0.95],
    "diffusion": {
        "sampler": "ddim", "sampling_timesteps": 50,
        "samples_per_condition": 10, "eta": 0.0,
    },
}


def _read_strings(dataset: h5py.Dataset) -> list[str]:
    values = dataset.asstr()[:] if hasattr(dataset, "asstr") else dataset[:]
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _load_reference(path: str | Path = REFERENCE_FILE) -> dict:
    path = Path(path)
    valid, reason = check_reference_file(path)
    if not valid:
        raise ValueError(f"reference is unavailable or stale: {path} ({reason})")
    with h5py.File(path, "r") as handle:
        data = {
            "names": _read_strings(handle["scenario_name"]),
            "cases": json.loads(handle.attrs["cases_json"]),
            "grids": {},
        }
        for name in GRID_NAMES:
            group = handle["grids"][name]
            data["grids"][name] = {
                "label": str(group.attrs["label"]),
                "reference": group["u"][:].astype(np.float32),
                "initial": group["initial"][:].astype(np.float32),
                "bL": group["bL"][:].astype(np.float32),
                "bR": group["bR"][:].astype(np.float32),
                "time": group["t"][:].astype(np.float64),
                "x": group["x"][:].astype(np.float64),
            }
    if data["names"] != [case["name"] for case in CASES]:
        raise ValueError("reference scenario definitions do not match the benchmark")
    return data


def _selected_methods(values=None) -> tuple[str, ...]:
    if isinstance(values, str):
        methods = (values,)
    else:
        methods = tuple(values or METHODS)
    if not methods:
        raise ValueError("at least one method is required")
    unknown = [method for method in methods if method not in METHOD_LABELS]
    if unknown:
        raise ValueError(f"unsupported super-resolution methods: {unknown}")
    duplicates = sorted({method for method in methods if methods.count(method) > 1})
    if duplicates:
        raise ValueError(f"duplicate methods are not allowed: {duplicates}")
    return methods


def _checkpoint_path(method: str) -> Path:
    base = INTERPOLATION_BASE_METHODS.get(method, method)
    return output_directory(base) / "checkpoint.pt"


def _method_output(method: str) -> Path:
    base = INTERPOLATION_BASE_METHODS.get(method, method)
    return output_directory(base) / (
        "spatial_superresolution_bicubic" if method in INTERPOLATION_BASE_METHODS
        else "spatial_superresolution"
    )


def _training_axes(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "axes" not in checkpoint:
        raise KeyError("checkpoint does not contain the physical training axes")
    return {name: torch.as_tensor(checkpoint["axes"][name]).float() for name in ("t", "x")}


def _validate_training_grid(reference: dict, axes: dict[str, torch.Tensor]) -> None:
    grid = reference["grids"]["training_grid"]
    expected_t = axes["t"].detach().cpu().numpy()
    expected_x = axes["x"][1:-1].detach().cpu().numpy()
    for name, actual, expected in (
        ("time", grid["time"], expected_t), ("x", grid["x"], expected_x),
    ):
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"training-grid {name} axis does not match checkpoint coordinates")
    for name, grid in reference["grids"].items():
        if grid["time"].shape != expected_t.shape or not np.allclose(
            grid["time"], expected_t, rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(f"{name} time axis does not match the training snapshots")


def _physical_values(values: torch.Tensor, scalers: dict) -> torch.Tensor:
    minimum = torch.as_tensor(scalers["min"], device=values.device, dtype=values.dtype)
    maximum = torch.as_tensor(scalers["max"], device=values.device, dtype=values.dtype)
    return (values + 1.0) * 0.5 * (maximum - minimum) + minimum


def _normalise(values: torch.Tensor, scalers: dict) -> torch.Tensor:
    minimum = torch.as_tensor(scalers["min"], dtype=values.dtype)
    maximum = torch.as_tensor(scalers["max"], dtype=values.dtype)
    return 2.0 * (values - minimum) / (maximum - minimum).clamp_min(1e-12) - 1.0


def _build_condition(grid: dict, *, axes: dict[str, torch.Tensor], scalers: dict,
                     device: torch.device, dtype: torch.dtype,
                     case_index: int = 0) -> torch.Tensor:
    """Pack physical initial/Robin data, then apply the checkpoint scaler."""
    time_count, x_count = len(grid["time"]), len(grid["x"])
    packed = torch.zeros((1, time_count, x_count + 2), dtype=torch.float32)
    initial = np.asarray(grid["initial"])[case_index]
    left = np.asarray(grid["bL"])[case_index]
    right = np.asarray(grid["bR"])[case_index]
    packed[0, 0, 1:-1] = torch.from_numpy(initial)
    packed[0, :, 0] = torch.from_numpy(left)
    packed[0, :, -1] = torch.from_numpy(right)
    return _normalise(packed, scalers).unsqueeze(1).to(device=device, dtype=dtype)


def _normalized_queries(grid: dict, axes: dict[str, torch.Tensor], *, device, dtype) -> torch.Tensor:
    time = torch.as_tensor(grid["time"], device=device, dtype=dtype)
    x = torch.as_tensor(grid["x"], device=device, dtype=dtype)
    t_norm = 2.0 * (time - axes["t"][0].to(device=device, dtype=dtype)) / (
        axes["t"][-1].to(device=device, dtype=dtype) - axes["t"][0].to(device=device, dtype=dtype)
    ) - 1.0
    x_norm = 2.0 * (x - axes["x"][0].to(device=device, dtype=dtype)) / (
        axes["x"][-1].to(device=device, dtype=dtype) - axes["x"][0].to(device=device, dtype=dtype)
    ) - 1.0
    tt, xx = torch.meshgrid(t_norm, x_norm, indexing="ij")
    return torch.stack((tt, xx), dim=-1).reshape(1, -1, 2)


def _physical_prediction(flat_normalized: torch.Tensor, target_shape: tuple[int, int], scalers: dict) -> np.ndarray:
    values = _physical_values(flat_normalized, scalers)[0, :, 0]
    return values.reshape(target_shape).detach().cpu().numpy()


def _decode_operator(model: torch.nn.Module, condition: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    packed = condition[:, 0]
    chunks = []
    with torch.no_grad():
        for start in range(0, query.shape[1], int(SETTINGS["query_chunk_size"])):
            chunks.append(model(packed, None, coords=query[:, start:start + int(SETTINGS["query_chunk_size"])]))
    return torch.cat(chunks, dim=1)


def _decode_continuous(objective: GaussianDiffusion, condition: torch.Tensor,
                       query: torch.Tensor) -> torch.Tensor:
    model = objective.model
    with torch.no_grad():
        with capture_last_continuous_latent(model) as captured:
            objective.sample(
                condition, sampler=SETTINGS["diffusion"]["sampler"],
                sampling_timesteps=SETTINGS["diffusion"]["sampling_timesteps"],
                eta=SETTINGS["diffusion"]["eta"],
            )
        if captured.get("latent") is None:
            raise RuntimeError("failed to capture final continuous latent")
        chunks = []
        for start in range(0, query.shape[1], int(SETTINGS["query_chunk_size"])):
            chunks.append(model.decoder(
                captured["latent"], query[:, start:start + int(SETTINGS["query_chunk_size"])]
            ).clamp(-1.0, 1.0))
        return torch.cat(chunks, dim=1)


def _cubic_interpolate(training_values: np.ndarray, source_x: np.ndarray,
                       target_x: np.ndarray) -> np.ndarray:
    try:
        from scipy.interpolate import CubicSpline
    except ImportError as exc:
        raise RuntimeError("fdm_unet_bicubic requires scipy for cubic interpolation") from exc
    result = np.empty((training_values.shape[0], len(target_x)), dtype=np.float32)
    for index, values in enumerate(training_values):
        result[index] = CubicSpline(source_x, values, extrapolate=True)(target_x)
    return result


def _relative_l2_by_time(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(reference, axis=1).clip(min=1e-12)
    return np.linalg.norm(prediction - reference, axis=1) / denominator


def _aggregate_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1e-12))


def _visualization_indices(time: np.ndarray) -> np.ndarray:
    targets = np.asarray(SETTINGS["visualization_times"], dtype=float)
    indices = np.asarray([int(np.argmin(np.abs(time - target))) for target in targets])
    if len(np.unique(indices)) != len(indices):
        raise ValueError("visualization times map to duplicate saved snapshots")
    return indices


def _build_target_objective(config: dict, grid: dict, state: dict,
                           device: torch.device):
    axes = {
        "t": torch.from_numpy(grid["time"]).float(),
        "x": torch.cat((
            torch.tensor([config["domain"]["x"][0]]),
            torch.from_numpy(grid["x"]).float(),
            torch.tensor([config["domain"]["x"][1]]),
        )),
    }
    grid_config = deepcopy(config)
    grid_config["grid_shape"] = [len(axes["t"]), len(axes["x"])]
    coordinates = CoordinateSystem((axes["t"], axes["x"])).normalized_axes
    material_condition, material_function = build_material_inputs(grid_config, axes)
    experiment = build_experiment(
        grid_config, grid_coordinates=coordinates,
        material_condition=material_condition, material_function=material_function,
    )
    experiment.objective.load_state_dict(state)
    experiment.objective.to(device).eval()
    return experiment, axes


def _plot_comparison(records: list[dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("visualization requires matplotlib") from exc
    output.mkdir(parents=True, exist_ok=True)
    for record in records:
        for grid_name in GRID_NAMES:
            grid = record["grids"][grid_name]
            indices = _visualization_indices(np.asarray(grid["time"]))
            rows = len(record["methods"])
            figure, panels = plt.subplots(rows, len(indices), figsize=(4.2 * len(indices), 2.8 * rows), squeeze=False, sharex=True)
            for row, method in enumerate(record["methods"]):
                prediction = np.asarray(grid["predictions"][method])
                reference = np.asarray(grid["reference"])
                for column, index in enumerate(indices):
                    panel = panels[row, column]
                    panel.plot(grid["x"], reference[index], color="#222222", linewidth=1.5, label="reference")
                    panel.plot(grid["x"], prediction[index], color="#c44747", linewidth=1.1, label=METHOD_LABELS[method])
                    panel.set_title(f"{METHOD_LABELS[method]}\n t={grid['time'][index]:.3g}")
                    panel.grid(alpha=0.2)
                    if row == rows - 1:
                        panel.set_xlabel("x")
                    if column == 0:
                        panel.set_ylabel("E")
                    if row == 0 and column == 0:
                        panel.legend(fontsize=8)
            figure.suptitle(f"{record['name']}: {GRID_LABELS[grid_name]}")
            figure.tight_layout()
            figure.savefig(output / f"{record['name']}_{grid_name}.png", dpi=180, bbox_inches="tight")
            plt.close(figure)

        figure, panel = plt.subplots(1, 1, figsize=(8, 4.5), constrained_layout=True)
        for method in record["methods"]:
            curves = record["error_curves"][method]
            for grid_name in GRID_NAMES:
                panel.plot(record["grids"][grid_name]["time"], curves[grid_name], label=f"{METHOD_LABELS[method]} / {GRID_LABELS[grid_name]}")
        panel.set_title(f"{record['name']}: per-time relative L2")
        panel.set_xlabel("t")
        panel.set_ylabel("relative L2")
        panel.grid(alpha=0.25)
        panel.legend(fontsize=7, ncol=2)
        figure.savefig(output / f"{record['name']}_error_curves.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def evaluate_methods(methods: tuple[str, ...], reference: dict,
                     reference_path: str | Path = REFERENCE_FILE) -> dict:
    device = choose_device(SETTINGS["device"])
    prepared = {}
    for method in methods:
        checkpoint_path = _checkpoint_path(method)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found for {method}: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = deepcopy(checkpoint["config"])
        checkpoint_method = INTERPOLATION_BASE_METHODS.get(method, method)
        config["method"] = canonical_method_name(config["method"])
        if config.get("case") != "radiative_1d" or config["method"] != checkpoint_method:
            raise ValueError(f"checkpoint {checkpoint_path} does not match radiative_1d/{method}")
        axes = _training_axes(checkpoint)
        _validate_training_grid(reference, axes)
        state = checkpoint.get("ema", {}).get("model") if checkpoint.get("ema") else checkpoint["model"]
        config["grid_shape"] = [len(axes["t"]), len(axes["x"])]
        coordinates = CoordinateSystem((axes["t"], axes["x"])).normalized_axes
        material_condition, material_function = build_material_inputs(config, axes)
        experiment = build_experiment(
            config, grid_coordinates=coordinates,
            material_condition=material_condition, material_function=material_function,
        )
        experiment.objective.load_state_dict(state)
        experiment.objective.to(device).eval()
        prepared[method] = {
            "checkpoint": checkpoint, "config": config, "axes": axes,
            "experiment": experiment, "state": state,
        }

    records = []
    for case_index, case in enumerate(reference["cases"]):
        record = {"name": case["name"], "methods": list(methods), "grids": {}, "error_curves": {method: {} for method in methods}}
        predictions_for_grid = {grid_name: {method: [] for method in methods} for grid_name in GRID_NAMES}
        errors_for_grid = {grid_name: {method: [] for method in methods} for grid_name in GRID_NAMES}
        for method in methods:
            item = prepared[method]
            checkpoint, config, axes = item["checkpoint"], item["config"], item["axes"]
            experiment = item["experiment"]
            stochastic = isinstance(experiment.objective, GaussianDiffusion)
            draws = int(SETTINGS["diffusion"]["samples_per_condition"] if stochastic else 1)
            target_objectives = {}
            target_axes = {}
            if method in ("fno", "pino"):
                for grid_name in GRID_NAMES:
                    target_objectives[grid_name], target_axes[grid_name] = _build_target_objective(
                        config, reference["grids"][grid_name], item["state"], device
                    )
            for draw in range(draws):
                training_condition = _build_condition(
                    reference["grids"]["training_grid"], axes=axes,
                    scalers=checkpoint["scalers"], device=device,
                    dtype=next(experiment.objective.parameters()).dtype,
                    case_index=case_index,
                )
                normalized = {}
                if method in ("fno", "pino"):
                    for grid_name in GRID_NAMES:
                        target = reference["grids"][grid_name]
                        condition = _build_condition(
                            target, axes=target_axes[grid_name], scalers=checkpoint["scalers"],
                            device=device,
                            dtype=next(target_objectives[grid_name].objective.parameters()).dtype,
                            case_index=case_index,
                        )
                        with torch.no_grad():
                            packed = target_objectives[grid_name].objective.predict(condition)
                        normalized[grid_name] = _physical_values(
                            packed[0, 0, :, 1:-1].unsqueeze(0), checkpoint["scalers"]
                        )[0].detach().cpu().numpy()
                elif method in INTERPOLATION_BASE_METHODS:
                    with torch.no_grad():
                        packed = experiment.objective.sample(
                            training_condition, sampler=SETTINGS["diffusion"]["sampler"],
                            sampling_timesteps=SETTINGS["diffusion"]["sampling_timesteps"],
                            eta=SETTINGS["diffusion"]["eta"],
                        )
                    training_prediction = _physical_values(
                        packed[:, 0, :, 1:-1], checkpoint["scalers"]
                    )[0].detach().cpu().numpy()
                    source_x = reference["grids"]["training_grid"]["x"]
                    for grid_name in GRID_NAMES:
                        target = reference["grids"][grid_name]
                        normalized[grid_name] = _cubic_interpolate(
                            training_prediction, source_x, target["x"]
                        )
                elif stochastic:
                    query_by_grid = {
                        name: _normalized_queries(reference["grids"][name], axes, device=device,
                                                 dtype=next(experiment.objective.parameters()).dtype)
                        for name in GRID_NAMES
                    }
                    for grid_name, query in query_by_grid.items():
                        decoded = _decode_continuous(experiment.objective, training_condition, query)
                        normalized[grid_name] = _physical_prediction(
                            decoded, reference["grids"][grid_name]["reference"].shape[1:], checkpoint["scalers"]
                        )
                else:
                    query_by_grid = {
                        name: _normalized_queries(reference["grids"][name], axes, device=device,
                                                 dtype=next(experiment.objective.parameters()).dtype)
                        for name in GRID_NAMES
                    }
                    for grid_name, query in query_by_grid.items():
                        decoded = _decode_operator(experiment.objective.model, training_condition, query)
                        normalized[grid_name] = _physical_prediction(
                            decoded, reference["grids"][grid_name]["reference"].shape[1:], checkpoint["scalers"]
                        )
                for grid_name in GRID_NAMES:
                    target = reference["grids"][grid_name]["reference"][case_index]
                    prediction = np.asarray(normalized[grid_name])
                    # The first saved snapshot is a known condition, not a
                    # learned off-grid query.  Reapply it after decoding so
                    # DeepONet and continuous decoders follow the same
                    # freezing convention as the diffusion/grid objectives.
                    prediction[0] = reference["grids"][grid_name]["initial"][case_index]
                    predictions_for_grid[grid_name][method].append(prediction)
                    errors_for_grid[grid_name][method].append(_aggregate_error(prediction, target))

        for grid_name in GRID_NAMES:
            grid = reference["grids"][grid_name]
            record["grids"][grid_name] = {
                "label": grid["label"], "time": grid["time"].tolist(), "x": grid["x"].tolist(),
                "reference": grid["reference"][case_index].tolist(), "predictions": {}, "metrics": {},
            }
            for method in methods:
                predictions = np.asarray(predictions_for_grid[grid_name][method])
                errors = np.asarray(errors_for_grid[grid_name][method])
                best_draw = int(np.argmin(errors))
                best = predictions[best_draw]
                reference_values = grid["reference"][case_index]
                per_time = _relative_l2_by_time(best, reference_values)
                record["grids"][grid_name]["predictions"][method] = best.tolist()
                record["grids"][grid_name]["metrics"][method] = {
                    "best_draw": best_draw + 1, "best_total_relative_l2": float(errors[best_draw]),
                    "sampling_mean_total_relative_l2": float(errors.mean()),
                    "sampling_std_total_relative_l2": float(errors.std()),
                    "best_relative_l2_by_time": per_time.tolist(),
                }
                record["error_curves"][method][grid_name] = per_time.tolist()
        records.append(record)
        print(f"[superresolution] case={case_index + 1}/{len(reference['cases'])} name={case['name']}", flush=True)

    output = COMPARISON_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    _plot_comparison(records, output / "figures")
    summary = {method: {
        grid_name: {
            "best_total_relative_l2": float(np.mean([
                record["grids"][grid_name]["metrics"][method]["best_total_relative_l2"] for record in records
            ])),
            "sampling_mean_total_relative_l2": float(np.mean([
                record["grids"][grid_name]["metrics"][method]["sampling_mean_total_relative_l2"] for record in records
            ])),
        }
        for grid_name in GRID_NAMES
    } for method in methods}
    for method in methods:
        method_output = _method_output(method)
        method_output.mkdir(parents=True, exist_ok=True)
        method_cases = []
        for record in records:
            method_grids = {}
            for grid_name in GRID_NAMES:
                grid = record["grids"][grid_name]
                method_grids[grid_name] = {
                    "time": torch.as_tensor(grid["time"]),
                    "x": torch.as_tensor(grid["x"]),
                    "reference": torch.as_tensor(grid["reference"]),
                    "best_prediction": torch.as_tensor(grid["predictions"][method]),
                    "metrics": grid["metrics"][method],
                }
            method_cases.append({"name": record["name"], "grids": method_grids})
        torch.save(
            {"method": method, "reference_file": str(Path(reference_path).resolve()),
             "cases": method_cases, "summary": summary[method]},
            method_output / "superresolution_predictions.pt",
        )
        (method_output / "superresolution_metrics.json").write_text(
            json.dumps({"method": method, "summary": summary[method],
                        "cases": [
                            {"name": case["name"], "grids": {
                                name: values["metrics"] for name, values in case["grids"].items()
                            }} for case in method_cases
                        ]}, indent=2),
            encoding="utf-8",
        )
    compact_cases = []
    for record in records:
        compact_cases.append({
            "name": record["name"],
            "grids": {
                grid_name: {
                    "time": record["grids"][grid_name]["time"],
                    "x": record["grids"][grid_name]["x"],
                    "metrics": record["grids"][grid_name]["metrics"],
                }
                for grid_name in GRID_NAMES
            },
        })
    report = {
        "methods": list(methods), "grids": list(GRID_NAMES),
        "summary": summary, "cases": compact_cases,
    }
    (output / "superresolution_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# radiative_1d spatial super-resolution", "", "| Method | Grid | Best total relative L2 | Sampling mean total relative L2 |", "| --- | --- | ---: | ---: |"]
    for method in methods:
        for grid_name in GRID_NAMES:
            values = summary[method][grid_name]
            lines.append(f"| {METHOD_LABELS[method]} | {GRID_LABELS[grid_name]} | {values['best_total_relative_l2']:.6e} | {values['sampling_mean_total_relative_l2']:.6e} |")
    (output / "superresolution_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[superresolution] report={output / 'superresolution_report.md'}", flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--reference", type=Path, default=REFERENCE_FILE)
    args = parser.parse_args(argv)
    reference = _load_reference(args.reference)
    evaluate_methods(_selected_methods(args.methods), reference, args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
