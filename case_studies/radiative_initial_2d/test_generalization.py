"""Compare spatial super-resolution predictions on three direct-solver grids.

This entry point is independent of test.py. It evaluates only the spatial
super-resolution reference made by generate_generalization.py and leaves ID/OOD evaluation
artifacts untouched. Diffusion methods use ten draws and report the draw with
the lowest total space-time relative L2. Figures compare every method with the
same reference and color scale at three representative times. A separate
per-case figure retains the full 31-time error history at every resolution.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F


CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_studies.radiative_initial_2d.generate_generalization import (
    OUTPUT as REFERENCE_FILE,
    REFERENCE_GRIDS,
    build_generalization_initial,
    check_reference_file,
)
from case_studies.radiative_initial_2d.train import output_directory
from pifdm.core.diffusion import GaussianDiffusion
from pifdm.evaluation.residual_diagnostics import capture_last_continuous_latent
from pifdm.experiments.naming import canonical_method_name
from pifdm.experiments.physics import build_material_inputs
from pifdm.experiments.runner import build_experiment
from pifdm.physics.coordinates import CoordinateSystem
from pifdm.workflows.training import choose_device


METHODS = ("pifdm_vit", "c_fdm_vit", "pifdm_unet", "c_fdm_unet", "deeponet", "pideeponet", "fno", "pino", "fdm_vit_bicubic")
METHOD_LABELS = {
    "pifdm_vit": "PI-FDM-ViT", "c_fdm_vit": "C-FDM-ViT",
    "pifdm_unet": "PI-FDM-UNet", "c_fdm_unet": "C-FDM-UNet",
    "deeponet": "DeepONet", "pideeponet": "PI-DeepONet",
    "fno": "FNO", "pino": "PINO", "fdm_vit_bicubic": "FDM-ViT + bicubic",
}
GRID_NAMES = tuple(REFERENCE_GRIDS)
GRID_CURVE_LABELS = {
    "training_grid": "62x62",
    "superresolution_200": "200x200",
    "superresolution_400": "400x400",
}
TRAINING_SAMPLES = 100
COMPARISON_OUTPUT = CASE_DIR / "runs" / "spatial_superresolution_comparison"
EVALUATION_SETTINGS = {
    "reference": REFERENCE_FILE,
    "device": "auto",
    "save_visualizations": True,
    # Inference has no decoder autograd graph, so use the trained decoder's
    # full chunk size to keep 400x400 coordinate queries practical.
    "query_chunk_size": 131072,
    # Select the nearest directly solved snapshots; no temporal interpolation
    # is used for either the reference or prediction visualizations.
    "visualization_times": [0.1, 0.7, 1.4],
    "diffusion": {
        "sampler": "ddim",
        "sampling_timesteps": 50,
        "samples_per_condition": 10,
        "eta": 0.0,
    },
}


def _read_string_dataset(dataset: h5py.Dataset) -> list[str]:
    """Read UTF-8 HDF5 strings without depending on h5py 3.x APIs."""
    values = dataset.asstr()[:] if hasattr(dataset, "asstr") else dataset[:]
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _load_reference(path: str | Path) -> dict:
    path = Path(path)
    valid, reason = check_reference_file(path)
    if not valid:
        raise ValueError(f"super-resolution reference is unavailable or stale: {path} ({reason})")
    with h5py.File(path, "r") as handle:
        required = ("scenario_name", "t", "grids")
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"super-resolution reference is missing datasets: {missing}")
        cases = json.loads(handle.attrs["interpolation_cases_json"])
        data = {
            "names": _read_string_dataset(handle["scenario_name"]),
            "time": handle["t"][:],
            "cases": cases,
            "grids": {
                name: {
                    "name": name,
                    "label": str(handle["grids"][name].attrs["label"]),
                    "reference": handle["grids"][name]["u"][:],
                    "initial": handle["grids"][name]["initial"][:],
                    "time": handle["grids"][name]["t"][:],
                    "y": handle["grids"][name]["y"][:],
                    "x": handle["grids"][name]["x"][:],
                }
                for name in GRID_NAMES
            },
        }
    if len(data["cases"]) != len(data["names"]):
        raise ValueError("super-resolution scenario metadata does not match reference samples")
    for name, grid in data["grids"].items():
        expected_shape = (
            len(data["names"]), len(grid["time"]), len(grid["y"]), len(grid["x"]),
        )
        if grid["reference"].shape != expected_shape:
            raise ValueError(
                f"{name} reference shape {grid['reference'].shape} does not match {expected_shape}"
            )
    return data


def _checkpoint_path(method: str) -> Path:
    checkpoint_method = "fdm_vit" if method == "fdm_vit_bicubic" else method
    return output_directory(checkpoint_method, TRAINING_SAMPLES) / "checkpoint.pt"


def _method_output(method: str) -> Path:
    checkpoint_method = "fdm_vit" if method == "fdm_vit_bicubic" else method
    suffix = "spatial_superresolution_bicubic" if method == "fdm_vit_bicubic" else "spatial_superresolution"
    return output_directory(checkpoint_method, TRAINING_SAMPLES) / suffix


def _selected_methods() -> tuple[str, ...]:
    """Normalize a single method string or a sequence of method names."""
    methods = (METHODS,) if isinstance(METHODS, str) else tuple(METHODS)
    if not methods:
        raise ValueError("METHODS must contain at least one method")
    if any(not isinstance(method, str) or not method for method in methods):
        raise ValueError("METHODS must contain non-empty method-name strings")
    unknown = [method for method in methods if method not in METHOD_LABELS]
    if unknown:
        raise ValueError(f"unsupported super-resolution methods: {unknown}")
    return methods


def _model_dtype(model: torch.nn.Module) -> torch.dtype:
    parameter = next(model.parameters(), None)
    return torch.float32 if parameter is None else parameter.dtype


def _build_condition(initial_spec: dict, *, axes: dict[str, torch.Tensor],
                     scalers: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build the packed frozen initial/Robin condition on the training grid."""
    time = axes["t"]
    y, x = axes["y"][1:-1], axes["x"][1:-1]
    initial = torch.from_numpy(build_generalization_initial(
        x.detach().cpu().numpy(), y.detach().cpu().numpy(), initial_spec["initial_condition"]
    )).float()
    packed = torch.zeros((1, len(time), len(axes["y"]), len(axes["x"])), dtype=torch.float32)
    packed[0, 0, 1:-1, 1:-1] = initial
    minimum = torch.as_tensor(scalers["min"], dtype=packed.dtype)
    maximum = torch.as_tensor(scalers["max"], dtype=packed.dtype)
    normalized = 2.0 * (packed - minimum) / (maximum - minimum).clamp_min(1e-12) - 1.0
    return normalized.unsqueeze(1).to(device=device, dtype=dtype)


def _evaluation_grids(reference: dict, axes: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Validate and return the three independently solved evaluation grids."""
    grids = {
        name: {
            "name": name, "label": source["label"],
            "time": np.asarray(source["time"], dtype=float),
            "y": np.asarray(source["y"], dtype=float),
            "x": np.asarray(source["x"], dtype=float),
            "reference": np.asarray(source["reference"], dtype=np.float32),
        }
        for name, source in reference["grids"].items()
    }
    training = grids["training_grid"]
    expected_axes = {
        "time": axes["t"][1:].detach().cpu().numpy(),
        "y": axes["y"][1:-1].detach().cpu().numpy(),
        "x": axes["x"][1:-1].detach().cpu().numpy(),
    }
    for name, expected in expected_axes.items():
        if training[name].shape != expected.shape or not np.allclose(
            training[name], expected, rtol=0.0, atol=1e-6,
        ):
            raise RuntimeError(f"training-grid reference {name} axis does not match checkpoint coordinates")
    for grid_name, grid in grids.items():
        if grid["time"].shape != expected_axes["time"].shape or not np.allclose(
            grid["time"], expected_axes["time"], rtol=0.0, atol=1e-6,
        ):
            raise RuntimeError(f"{grid_name} does not contain the checkpoint's 31 positive times")
    return grids


def _query_coordinates(grid: dict, axes: dict[str, torch.Tensor], *, device, dtype) -> torch.Tensor:
    """Return physical evaluation coordinates normalized by the training domain."""
    time, y, x = (torch.as_tensor(grid[name], device=device, dtype=dtype) for name in ("time", "y", "x"))
    normalized = []
    for values, training_axis in zip((time, y, x), (axes["t"], axes["y"], axes["x"])):
        lower, upper = training_axis[0].to(values), training_axis[-1].to(values)
        normalized.append(2.0 * (values - lower) / (upper - lower) - 1.0)
    tt, yy, xx = torch.meshgrid(*normalized, indexing="ij")
    return torch.stack((tt, yy, xx), dim=-1).reshape(1, -1, 3)


def _physical_values(normalized: torch.Tensor, scalers: dict) -> torch.Tensor:
    minimum = torch.as_tensor(scalers["min"], device=normalized.device, dtype=normalized.dtype)
    maximum = torch.as_tensor(scalers["max"], device=normalized.device, dtype=normalized.dtype)
    return (normalized + 1.0) * 0.5 * (maximum - minimum) + minimum


def _decode_operator(model: torch.nn.Module, condition: torch.Tensor,
                     coordinates_by_grid: dict[str, torch.Tensor], chunk_size: int) -> dict[str, torch.Tensor]:
    outputs = {}
    packed = condition[:, 0]
    with torch.no_grad():
        for name, coordinates in coordinates_by_grid.items():
            chunks = []
            for start in range(0, coordinates.shape[1], chunk_size):
                chunks.append(model(packed, None, coords=coordinates[:, start:start + chunk_size]))
            outputs[name] = torch.cat(chunks, dim=1)
    return outputs


def _packed_axes(grid: dict, config: dict) -> dict[str, torch.Tensor]:
    """Add the known initial time and geometric boundary coordinates."""
    return {
        "t": torch.cat((
            torch.tensor([config["domain"]["t"][0]], dtype=torch.float32),
            torch.as_tensor(grid["time"], dtype=torch.float32),
        )),
        "y": torch.cat((
            torch.tensor([config["domain"]["y"][0]], dtype=torch.float32),
            torch.as_tensor(grid["y"], dtype=torch.float32),
            torch.tensor([config["domain"]["y"][1]], dtype=torch.float32),
        )),
        "x": torch.cat((
            torch.tensor([config["domain"]["x"][0]], dtype=torch.float32),
            torch.as_tensor(grid["x"], dtype=torch.float32),
            torch.tensor([config["domain"]["x"][1]], dtype=torch.float32),
        )),
    }


def _build_grid_operator_objectives(config: dict, grids: dict[str, dict], state: dict,
                                    device: torch.device):
    """Build one size-agnostic FNO/PINO objective for each regular grid."""
    objectives = {}
    packed_axes = {}
    for name, grid in grids.items():
        axes = _packed_axes(grid, config)
        grid_config = deepcopy(config)
        grid_config["grid_shape"] = [len(axes[axis]) for axis in ("t", "y", "x")]
        coordinates = CoordinateSystem((axes["t"], axes["y"], axes["x"])).normalized_axes
        material_condition, material_function = build_material_inputs(grid_config, axes)
        experiment = build_experiment(
            grid_config, grid_coordinates=coordinates,
            material_condition=material_condition, material_function=material_function,
        )
        experiment.objective.load_state_dict(state)
        experiment.objective.to(device).eval()
        objectives[name] = experiment.objective
        packed_axes[name] = axes
    return objectives, packed_axes


def _decode_grid_operator(objectives: dict, conditions: dict) -> dict[str, torch.Tensor]:
    """Evaluate FNO/PINO directly on each packed regular grid."""
    outputs = {}
    with torch.no_grad():
        for name, objective in objectives.items():
            packed = objective.predict(conditions[name])
            interior = packed[:, :, 1:, 1:-1, 1:-1]
            outputs[name] = interior.reshape(interior.shape[0], -1, 1)
    return outputs


def _decode_diffusion(objective: GaussianDiffusion, condition: torch.Tensor,
                      coordinates_by_grid: dict[str, torch.Tensor], settings: dict) -> dict[str, torch.Tensor]:
    """Sample once, retain its final latent, then decode both requested grids."""
    model = objective.model
    previous_chunk_size = model.decoder.chunk_size
    model.decoder.chunk_size = min(previous_chunk_size, int(settings["query_chunk_size"]))
    try:
        with torch.no_grad():
            with capture_last_continuous_latent(model) as captured:
                objective.sample(
                    condition, sampler=settings["diffusion"]["sampler"],
                    sampling_timesteps=settings["diffusion"]["sampling_timesteps"],
                    eta=settings["diffusion"]["eta"],
                )
            if captured.get("latent") is None:
                raise RuntimeError("failed to capture the final continuous-decoder latent")
            return {
                name: model.decoder(captured["latent"], coordinates).clamp(-1.0, 1.0)
                for name, coordinates in coordinates_by_grid.items()
            }
    finally:
        model.decoder.chunk_size = previous_chunk_size


def _bicubic_spatial_interpolation(values: torch.Tensor, source_y: np.ndarray,
                                   source_x: np.ndarray, target_y: np.ndarray,
                                   target_x: np.ndarray) -> torch.Tensor:
    """Apply tensor-product bicubic interpolation on physical grid coordinates."""
    if values.ndim != 3:
        raise ValueError("bicubic input must have shape (time, y, x)")
    source_y, source_x = np.asarray(source_y), np.asarray(source_x)
    target_y, target_x = np.asarray(target_y), np.asarray(target_x)
    if values.shape[1:] != (len(source_y), len(source_x)):
        raise ValueError("bicubic source axes do not match the prediction")
    # grid_sample is cubic in values but indexes a regular tensor. Map physical
    # coordinates to that tensor index first; points just beyond the outer
    # training-cell centres use the nearest available prediction.
    indexed_y = np.interp(target_y, source_y, np.linspace(-1.0, 1.0, len(source_y)))
    indexed_x = np.interp(target_x, source_x, np.linspace(-1.0, 1.0, len(source_x)))
    yy, xx = torch.meshgrid(
        torch.as_tensor(indexed_y, device=values.device, dtype=values.dtype),
        torch.as_tensor(indexed_x, device=values.device, dtype=values.dtype),
        indexing="ij",
    )
    query = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(values.shape[0], -1, -1, -1)
    result = F.grid_sample(
        values.unsqueeze(1), query, mode="bicubic", padding_mode="border", align_corners=True,
    )
    return result[:, 0]


def _decode_fdm_vit_bicubic(objective: GaussianDiffusion, condition: torch.Tensor,
                            grids: dict[str, dict], axes: dict[str, torch.Tensor],
                            settings: dict) -> dict[str, torch.Tensor]:
    """Sample FDM-ViT once on 62x62, then interpolate that draw spatially."""
    with torch.no_grad():
        packed = objective.sample(
            condition, sampler=settings["diffusion"]["sampler"],
            sampling_timesteps=settings["diffusion"]["sampling_timesteps"],
            eta=settings["diffusion"]["eta"],
        )
        training_values = packed[0, 0, 1:, 1:-1, 1:-1]
        source_y = axes["y"][1:-1].detach().cpu().numpy()
        source_x = axes["x"][1:-1].detach().cpu().numpy()
        outputs = {}
        for name, grid in grids.items():
            if name == "training_grid":
                values = training_values
            else:
                values = _bicubic_spatial_interpolation(
                    training_values, source_y, source_x, grid["y"], grid["x"],
                )
            outputs[name] = values.clamp(-1.0, 1.0).reshape(1, -1, 1)
        return outputs


def _relative_l2_by_time(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute a relative L2 independently for every requested time slice."""
    difference = np.asarray(prediction) - np.asarray(reference)
    denominator = np.linalg.norm(np.asarray(reference).reshape(len(reference), -1), axis=1).clip(min=1e-12)
    return np.linalg.norm(difference.reshape(len(reference), -1), axis=1) / denominator


def _aggregate_relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Compute one relative L2 over every requested time and spatial point."""
    difference = np.asarray(prediction) - np.asarray(reference)
    denominator = np.linalg.norm(np.asarray(reference).reshape(-1)).clip(min=1e-12)
    return float(np.linalg.norm(difference.reshape(-1)) / denominator)


def _visualization_indices(time: np.ndarray, targets) -> np.ndarray:
    """Select the directly solved snapshots nearest the requested times."""
    time = np.asarray(time, dtype=float)
    targets = np.asarray(tuple(targets), dtype=float)
    if time.ndim != 1 or time.size == 0:
        raise ValueError("visualization requires a non-empty one-dimensional time axis")
    if targets.ndim != 1 or targets.size == 0 or not np.isfinite(targets).all():
        raise ValueError("visualization_times must contain finite time values")
    indices = np.asarray([int(np.argmin(np.abs(time - target))) for target in targets])
    if len(np.unique(indices)) != len(indices):
        raise ValueError("visualization_times map to duplicate saved snapshots")
    return indices


def _robust_color_limit(*arrays: np.ndarray, percentile: float = 99.5) -> float:
    """Return a positive per-panel-group limit without letting isolated peaks dominate."""
    values = np.concatenate([
        np.asarray(array, dtype=float).reshape(-1) for array in arrays
    ])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(values), percentile))
    if limit <= 1e-12:
        limit = float(np.max(np.abs(values)))
    return max(limit, 1e-12)


def _plot_comparison_case(results: list[dict], *, case_index: int, grid_name: str,
                          output: Path) -> None:
    """Plot times by rows and the reference/methods by columns."""
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    payloads = [result["comparison_cases"][case_index]["grids"][grid_name] for result in results]
    reference = payloads[0]["reference"]
    time = payloads[0]["time"]
    x, y = payloads[0]["x"], payloads[0]["y"]
    rows, columns = len(time), len(results) + 1
    figure, panels = plt.subplots(
        rows, columns, figsize=(2.7 * columns, 2.6 * rows),
        squeeze=False, constrained_layout=True,
    )
    extent = [x[0], x[-1], y[0], y[-1]]
    for row in range(rows):
        predictions = [payload["prediction"][row] for payload in payloads]
        limit = _robust_color_limit(reference[row], *predictions, percentile=99.7)
        images = []
        for column in range(columns):
            panel = panels[row, column]
            if column == 0:
                values = reference[row]
                label = "Reference"
            else:
                result, payload = results[column - 1], payloads[column - 1]
                values = payload["prediction"][row]
                label = METHOD_LABELS[result["method"]]
            images.append(panel.imshow(
                values, origin="lower", extent=extent, aspect="equal", cmap="magma",
                vmin=0.0, vmax=limit,
            ))
            panel.set_title(label if row == 0 else "", fontsize=10)
            panel.set_ylabel(f"t={time[row]:.3g} s\ny" if column == 0 else "", fontsize=9)
            panel.set_xlabel("x" if row == rows - 1 else "")
            panel.tick_params(labelsize=8)
        figure.colorbar(images[0], ax=list(panels[row, :]), fraction=0.018, pad=0.012)
    case_name = results[0]["comparison_cases"][case_index]["name"]
    grid_label = payloads[0]["label"]
    figure.suptitle(
        f"{case_name}: {grid_label}; best total-error draw; shared per-time p99.7 scales",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_case_error_curves(results: list[dict], *, case_index: int, output: Path) -> None:
    """Plot all 31 best-draw errors for every method and spatial grid."""
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    columns = 3
    rows = (len(results) + columns - 1) // columns
    figure, panels = plt.subplots(
        rows, columns, figsize=(12.5, 3.2 * rows + 0.8), squeeze=False,
        sharex=True, constrained_layout=True,
    )
    flat_panels = panels.reshape(-1)
    colors = ("#2b6f9c", "#cf6b32", "#448c62")
    line_styles = ("-", "--", ":")
    for method_index, result in enumerate(results):
        panel = flat_panels[method_index]
        case = result["comparison_cases"][case_index]
        handles = []
        for color, line_style, grid_name in zip(colors, line_styles, GRID_NAMES):
            grid = case["grids"][grid_name]
            handle, = panel.plot(
                grid["metric_time"], grid["relative_l2_by_time"],
                color=color, linestyle=line_style, linewidth=1.8,
                label=GRID_CURVE_LABELS[grid_name],
            )
            handles.append(handle)
        if method_index == 0:
            panel.legend(handles=handles, loc="best", frameon=False, fontsize=8)
        panel.set_title(METHOD_LABELS[result["method"]], fontsize=10)
        panel.set_xlabel("Time (s)")
        panel.set_ylabel("Relative L2")
        panel.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.8)
        panel.tick_params(labelsize=8)
    for panel in flat_panels[len(results):]:
        panel.set_visible(False)
    case_name = results[0]["comparison_cases"][case_index]["name"]
    figure.suptitle(
        f"{case_name}: best-draw relative L2 over all 31 positive times",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _save_comparison_visualizations(results: list[dict], output: Path) -> None:
    if not results:
        return
    case_count = len(results[0]["comparison_cases"])
    for case_index in range(case_count):
        case_name = results[0]["comparison_cases"][case_index]["name"]
        for grid_name in GRID_NAMES:
            _plot_comparison_case(
                results, case_index=case_index, grid_name=grid_name,
                output=output / "figures" / case_name / f"{grid_name}.png",
            )
        _plot_case_error_curves(
            results, case_index=case_index,
            output=output / "figures" / case_name / "relative_l2_over_time.png",
        )


def _summarize_records(records: list[dict]) -> dict[str, dict]:
    summary = {}
    for grid_name in GRID_NAMES:
        values = [record["grids"][grid_name] for record in records]
        summary[grid_name] = {
            "shape": values[0]["shape"],
            "case_count": len(values),
            "best_total_relative_l2_mean": float(np.mean([
                value["best_total_relative_l2"] for value in values
            ])),
            "sampling_total_relative_l2_mean": float(np.mean([
                value["sampling_mean_total_relative_l2"] for value in values
            ])),
            "sampling_total_relative_l2_std": float(np.mean([
                value["sampling_std_total_relative_l2"] for value in values
            ])),
        }
    return summary


def _write_report(output: Path, *, method: str, records: list[dict], summary: dict,
                  draw_count: int) -> None:
    lines = [
        f"# radiative_initial_2d Spatial Super-Resolution: {METHOD_LABELS[method]}", "",
        "- The training grid contains `31 x 62 x 62` evaluated points (the initial time is frozen).",
        "- The spatial super-resolution grids contain `31 x 200 x 200` and `31 x 400 x 400` points at the same 31 positive times.",
        "- Every reference grid is produced directly by the finite-volume solver; reference values are not interpolated.",
        f"- Samples per condition: `{draw_count}`",
        "- Reported best errors use one draw selected by its total space-time relative L2 independently on each grid.",
        "", "## Final Total-Time Summary", "",
        "| Grid | Shape (t,y,x) | Mean best total relative L2 | Mean sampling relative L2 | Mean sampling std |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    if method == "fdm_vit_bicubic":
        lines.insert(5, "- FDM-ViT is sampled on `62 x 62`; only its prediction is bicubically interpolated to the larger grids.")
    for grid_name in GRID_NAMES:
        values = summary[grid_name]
        lines.append(
            f"| {grid_name} | `{tuple(values['shape'])}` | "
            f"{values['best_total_relative_l2_mean']:.6e} | "
            f"{values['sampling_total_relative_l2_mean']:.6e} | "
            f"{values['sampling_total_relative_l2_std']:.6e} |"
        )
    lines.extend([
        "", "## Per-Case Total-Time Summary", "",
        "| Grid | Case | Best draw | Best total relative L2 | Sampling mean relative L2 | Sampling std |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for record in records:
        for grid_name in GRID_NAMES:
            values = record["grids"][grid_name]
            lines.append(
                f"| {grid_name} | {record['name']} | {values['best_draw']} | "
                f"{values['best_total_relative_l2']:.6e} | "
                f"{values['sampling_mean_total_relative_l2']:.6e} | "
                f"{values['sampling_std_total_relative_l2']:.6e} |"
            )
    lines.extend([
        "", "Per-time best and sampling mean/std metrics remain in `superresolution_metrics.json`.",
    ])
    (output / "superresolution_report.md").write_text("\n".join(lines), encoding="utf-8")


def _print_final_summary(results: list[dict]) -> None:
    """Print mean best-draw and mean sampling errors after all sampling."""
    print("\n[superresolution] mean total space-time relative L2 across cases", flush=True)
    print(
        "| Method | 62x62 best | 62x62 sampling mean | "
        "200x200 best | 200x200 sampling mean | "
        "400x400 best | 400x400 sampling mean |",
        flush=True,
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |", flush=True)
    for result in results:
        values = []
        for name in GRID_NAMES:
            summary = result["summary"][name]
            values.extend((
                summary["best_total_relative_l2_mean"],
                summary["sampling_total_relative_l2_mean"],
            ))
        print(
            f"| {METHOD_LABELS[result['method']]} | "
            + " | ".join(f"{value:.6e}" for value in values) + " |",
            flush=True,
        )


def _write_comparison_report(results: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# radiative_initial_2d Spatial Super-Resolution Comparison", "",
        "All entries are relative L2 errors over all 31 positive times and all spatial points.",
        "For diffusion methods, each entry is the best of ten draws selected by total space-time error.",
        "Sampling mean is the mean total relative L2 across all ten draws, before best-draw selection.",
        "", "## Mean Across Cases", "",
        "| Method | 62x62 best | 62x62 sampling mean | 200x200 best | 200x200 sampling mean | 400x400 best | 400x400 sampling mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        values = []
        for name in GRID_NAMES:
            summary = result["summary"][name]
            values.extend((
                summary["best_total_relative_l2_mean"],
                summary["sampling_total_relative_l2_mean"],
            ))
        lines.append(
            f"| {METHOD_LABELS[result['method']]} | "
            + " | ".join(f"{value:.6e}" for value in values) + " |"
        )
    lines.extend([
        "", "## Per Case", "",
        "| Method | Case | 62x62 best | 62x62 sampling mean | 200x200 best | 200x200 sampling mean | 400x400 best | 400x400 sampling mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in results:
        for record in result["cases"]:
            values = []
            for name in GRID_NAMES:
                grid = record["grids"][name]
                values.extend((
                    grid["best_total_relative_l2"],
                    grid["sampling_mean_total_relative_l2"],
                ))
            lines.append(
                f"| {METHOD_LABELS[result['method']]} | {record['name']} | "
                + " | ".join(f"{value:.6e}" for value in values) + " |"
            )
    (output / "superresolution_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "superresolution_comparison.json").write_text(
        json.dumps({
            "methods": [result["method"] for result in results],
            "grids": list(GRID_NAMES),
            "summaries": {result["method"]: result["summary"] for result in results},
            "cases": {result["method"]: result["cases"] for result in results},
        }, indent=2),
        encoding="utf-8",
    )


def evaluate_method(method: str, reference: dict, *, settings: dict = EVALUATION_SETTINGS) -> dict:
    """Evaluate one checkpoint on all three spatial grids."""
    checkpoint_method = "fdm_vit" if method == "fdm_vit_bicubic" else method
    interpolation_baseline = method == "fdm_vit_bicubic"
    checkpoint_path = _checkpoint_path(method)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found for {method}: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = deepcopy(checkpoint["config"])
    config["method"] = canonical_method_name(config["method"])
    if config.get("case") != "radiative_initial_2d" or config.get("method") != checkpoint_method:
        raise ValueError(f"checkpoint {checkpoint_path} does not match radiative_initial_2d/{method}")
    if "axes" not in checkpoint:
        raise KeyError("checkpoint does not contain training axes required for dedicated generalization evaluation")
    axes = {name: torch.as_tensor(checkpoint["axes"][name]).float() for name in ("t", "y", "x")}
    state = checkpoint.get("ema", {}).get("model") if checkpoint.get("ema") else checkpoint["model"]
    device = choose_device(settings["device"])
    evaluation_grids = _evaluation_grids(reference, axes)
    grid_operator = checkpoint_method in ("fno", "pino")
    if grid_operator:
        grid_objectives, grid_axes = _build_grid_operator_objectives(
            config, evaluation_grids, state, device
        )
        experiment = None
        model = next(iter(grid_objectives.values())).model
    else:
        config["grid_shape"] = [len(axes["t"]), len(axes["y"]), len(axes["x"])]
        grid_coordinates = CoordinateSystem((axes["t"], axes["y"], axes["x"])).normalized_axes
        material_condition, material_function = build_material_inputs(config, axes)
        experiment = build_experiment(
            config, grid_coordinates=grid_coordinates,
            material_condition=material_condition, material_function=material_function,
        )
        experiment.objective.load_state_dict(state)
        experiment.objective.to(device).eval()
        model = experiment.objective.model
        grid_objectives, grid_axes = None, None
    dtype = _model_dtype(model)
    coordinates_by_grid = None if grid_operator or interpolation_baseline else {
        name: _query_coordinates(grid, axes, device=device, dtype=dtype)
        for name, grid in evaluation_grids.items()
    }
    output = _method_output(method)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    best_predictions_by_grid = {name: [] for name in evaluation_grids}
    per_time_draw_metrics = {name: [] for name in evaluation_grids}
    total_draw_metrics = {name: [] for name in evaluation_grids}
    comparison_cases = []
    stochastic = not grid_operator and isinstance(experiment.objective, GaussianDiffusion)
    draw_count = int(settings["diffusion"]["samples_per_condition"] if stochastic else 1)

    for case_index, (name, case_spec) in enumerate(zip(reference["names"], reference["cases"])):
        if grid_operator:
            conditions = {
                grid_name: _build_condition(
                    case_spec, axes=grid_axes[grid_name], scalers=checkpoint["scalers"],
                    device=device, dtype=dtype,
                )
                for grid_name in evaluation_grids
            }
        else:
            condition = _build_condition(
                case_spec, axes=axes, scalers=checkpoint["scalers"],
                device=device, dtype=dtype,
            )
        per_case_time_metrics = {grid_name: [] for grid_name in evaluation_grids}
        per_case_total_metrics = {grid_name: [] for grid_name in evaluation_grids}
        best_predictions = {}
        for draw_index in range(draw_count):
            if grid_operator:
                normalized_by_grid = _decode_grid_operator(grid_objectives, conditions)
            elif interpolation_baseline:
                normalized_by_grid = _decode_fdm_vit_bicubic(
                    experiment.objective, condition, evaluation_grids, axes, settings,
                )
            elif stochastic:
                normalized_by_grid = _decode_diffusion(
                    experiment.objective, condition, coordinates_by_grid, settings
                )
            else:
                normalized_by_grid = _decode_operator(
                    model, condition, coordinates_by_grid, int(settings["query_chunk_size"])
                )
            for grid_name, grid in evaluation_grids.items():
                target = grid["reference"][case_index]
                physical = _physical_values(normalized_by_grid[grid_name], checkpoint["scalers"])
                prediction = physical[0, :, 0].reshape(target.shape).detach().cpu().numpy()
                per_time = _relative_l2_by_time(prediction, target)
                total = _aggregate_relative_l2(prediction, target)
                per_case_time_metrics[grid_name].append(per_time)
                per_case_total_metrics[grid_name].append(total)
                if (
                    grid_name not in best_predictions
                    or total < min(per_case_total_metrics[grid_name][:-1], default=np.inf)
                ):
                    best_predictions[grid_name] = prediction.copy()
            print(
                f"[superresolution:{method}] case={case_index + 1}/{len(reference['names'])} "
                f"name={name} draw={draw_index + 1}/{draw_count}",
                flush=True,
            )
        grid_records = {}
        comparison_grids = {}
        for grid_name, grid in evaluation_grids.items():
            per_time = np.asarray(per_case_time_metrics[grid_name])
            aggregate = np.asarray(per_case_total_metrics[grid_name])
            best_draw = int(np.argmin(aggregate))
            best_by_time = per_time[best_draw]
            grid_records[grid_name] = {
                "shape": [len(grid["time"]), len(grid["y"]), len(grid["x"])],
                "best_draw": best_draw + 1,
                "best_total_relative_l2": float(aggregate[best_draw]),
                "sampling_mean_total_relative_l2": float(aggregate.mean()),
                "sampling_std_total_relative_l2": float(aggregate.std()),
                "best_relative_l2_by_time": best_by_time.tolist(),
                "sampling_mean_relative_l2_by_time": per_time.mean(axis=0).tolist(),
                "sampling_std_relative_l2_by_time": per_time.std(axis=0).tolist(),
            }
            best_predictions_by_grid[grid_name].append(best_predictions[grid_name])
            per_time_draw_metrics[grid_name].append(per_time)
            total_draw_metrics[grid_name].append(aggregate)
            visual_indices = _visualization_indices(
                grid["time"], settings["visualization_times"],
            )
            comparison_grids[grid_name] = {
                "label": grid["label"],
                "time": grid["time"][visual_indices].copy(),
                "metric_time": grid["time"].copy(),
                "y": grid["y"].copy(), "x": grid["x"].copy(),
                "reference": grid["reference"][case_index, visual_indices].copy(),
                "prediction": best_predictions[grid_name][visual_indices].copy(),
                "relative_l2": best_by_time[visual_indices].copy(),
                "relative_l2_by_time": best_by_time.copy(),
            }
        record = {
            "name": name,
            "description": case_spec["description"],
            "grids": grid_records,
        }
        records.append(record)
        comparison_cases.append({"name": name, "grids": comparison_grids})

    summary = _summarize_records(records)
    torch.save({
        "method": method,
        "reference_file": str(Path(settings["reference"]).resolve()),
        "scenario_names": reference["names"],
        "grids": {
            name: {
                "time": torch.from_numpy(np.asarray(grid["time"])),
                "y": torch.from_numpy(np.asarray(grid["y"])),
                "x": torch.from_numpy(np.asarray(grid["x"])),
                "best_predictions": torch.from_numpy(np.asarray(best_predictions_by_grid[name])),
                "relative_l2_by_time": torch.from_numpy(np.asarray(per_time_draw_metrics[name])),
                "total_relative_l2": torch.from_numpy(np.asarray(total_draw_metrics[name])),
            }
            for name, grid in evaluation_grids.items()
        },
        "summary": summary,
        "records": records,
    }, output / "superresolution_predictions.pt")
    _write_report(output, method=method, records=records, summary=summary, draw_count=draw_count)
    (output / "superresolution_metrics.json").write_text(
        json.dumps({
            "method": method,
            "reference_file": str(Path(settings["reference"]).resolve()),
            "summary": summary,
            "grids": {
                name: {
                    key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in grid.items()
                    if key in ("time", "y", "x", "shape")
                }
                for name, grid in evaluation_grids.items()
            },
            "cases": records,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[superresolution:{method}] output={output}", flush=True)
    return {
        "method": method, "output": str(output), "summary": summary,
        "cases": records, "comparison_cases": comparison_cases,
    }


def main():
    reference = _load_reference(EVALUATION_SETTINGS["reference"])
    results = [evaluate_method(method, reference) for method in _selected_methods()]
    _write_comparison_report(results, COMPARISON_OUTPUT)
    if EVALUATION_SETTINGS["save_visualizations"]:
        _save_comparison_visualizations(results, COMPARISON_OUTPUT)
    _print_final_summary(results)
    print(f"[superresolution] comparison_output={COMPARISON_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
