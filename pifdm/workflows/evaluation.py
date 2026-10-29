"""Shared evaluation workflow used by case-study test files."""

from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pifdm.cases import get_case_adapter
from pifdm.core.diffusion import GaussianDiffusion
from pifdm.data.datasets import (
    ConditionedFieldDataset, FieldDataset, load_hdf5_field,
    load_radiative_2d_field,
)
from pifdm.evaluation.metrics import evaluate_draws
from pifdm.evaluation.residual_diagnostics import (
    capture_last_continuous_latent, evaluate_radiative_residual_diagnostics,
    frozen_model_parameters, parse_residual_diagnostic_options,
    select_diagnostic_case_indices,
)
from pifdm.evaluation.scenarios import OOD_SCENARIOS, build_ood_case
from pifdm.evaluation.visualization import (
    plot_pde_residual_1d, plot_pde_residual_magnitude_1d,
    plot_pde_residual_magnitude_panels, plot_pde_residual_panels,
    plot_prediction, plot_radiative_1d_prediction, plot_radiative_2d_prediction,
    plot_robin_residual_1d, plot_robin_residual_panels,
)
from pifdm.experiments.registry import get_case
from pifdm.experiments.naming import DIFFCONPHY_BASE_METHODS, canonical_method_name
from pifdm.experiments.runner import build_experiment
from pifdm.experiments.physics import (
    build_grid_boundary_loss, build_grid_physics_loss, build_material_inputs,
)
from pifdm.physics.coordinates import CoordinateSystem
from pifdm.workflows.training import choose_device


def _pack_case_data(data, config, scalers):
    field = torch.as_tensor(data.fields).float()
    if field.ndim == 3:
        field = torch.cat((torch.as_tensor(data.boundaries["bL"]).float().unsqueeze(-1), field,
                           torch.as_tensor(data.boundaries["bR"]).float().unsqueeze(-1)), -1)
        axes = {"t": torch.as_tensor(data.axes["t"]).float(),
                "x": torch.cat((torch.tensor([config["domain"]["x"][0]]),
                                torch.as_tensor(data.axes["x"]).float(),
                                torch.tensor([config["domain"]["x"][1]])))}
    else:
        batch, nt, ny, nx = field.shape
        packed = torch.zeros(batch, nt, ny + 2, nx + 2)
        packed[:, :, 1:-1, 1:-1] = field
        packed[:, :, 1:-1, 0] = torch.as_tensor(data.boundaries["bL"])
        packed[:, :, 1:-1, -1] = torch.as_tensor(data.boundaries["bR"])
        packed[:, :, 0, 1:-1] = torch.as_tensor(data.boundaries["bB"])
        packed[:, :, -1, 1:-1] = torch.as_tensor(data.boundaries["bT"])
        field = packed
        axes = {name: torch.as_tensor(data.axes[name]).float() for name in ("t", "y", "x")}
        for name in ("x", "y"):
            axes[name] = torch.cat((torch.tensor([config["domain"][name][0]]), axes[name],
                                    torch.tensor([config["domain"][name][1]])))
    minimum, maximum = (torch.as_tensor(scalers[key]).float().cpu() for key in ("min", "max"))
    return 2.0 * (field - minimum) / (maximum - minimum).clamp_min(1e-12) - 1.0, axes


def _load_data(mode, data_path, config, checkpoint, adapter, scenarios):
    if mode == "ood":
        names = list(OOD_SCENARIOS if scenarios is None else scenarios)
        if not names:
            raise ValueError(f"{config['case']} has no configured OOD scenarios")
        cases = [build_ood_case(config, name) for name in names]
        first = cases[0]
        from pifdm.cases.base import CaseData
        data = CaseData(
            fields=np.concatenate([item.fields for item in cases]), axes=first.axes,
            boundaries={key: np.concatenate([item.boundaries[key] for item in cases]) for key in first.boundaries},
            material=first.material,
            source=np.stack([item.source for item in cases]) if first.source is not None else None,
            metadata={"mode": "ood", "scenario_names": names, **(first.metadata or {})},
        )
        fields, axes = _pack_case_data(data, config, checkpoint["scalers"])
        return data, fields, None, axes, names
    if data_path is None:
        raise ValueError("data_path is required for ID evaluation")
    data = adapter.load(data_path)
    adapter.validate(data)
    labels = [f"case_{index:04d}" for index in range(len(data.fields))]
    if config.get("case") == "radiative_2d":
        fields, g_left, axes, _ = load_radiative_2d_field(
            data_path, value_scalers=checkpoint["scalers"],
        )
        return data, fields, g_left, axes, labels
    fields, axes, _ = load_hdf5_field(
        data_path, add_boundaries=True, domain=config["domain"], value_scalers=checkpoint["scalers"],
    )
    return data, fields, None, axes, labels


def _physical_interior(normalized, checkpoint, dimension, *, keep_spatial_boundaries: bool = False):
    minimum = torch.as_tensor(checkpoint["scalers"]["min"], dtype=normalized.dtype)
    maximum = torch.as_tensor(checkpoint["scalers"]["max"], dtype=normalized.dtype)
    physical = (normalized + 1.0) * 0.5 * (maximum - minimum) + minimum
    if keep_spatial_boundaries:
        return physical
    return physical[..., 1:-1] if dimension == 1 else physical[..., 1:-1, 1:-1]


def _format_metric(value) -> str:
    return "-" if value is None else f"{float(value):.6e}"


def _write_markdown_report(path: Path, summary: dict, rows: list[dict], run_tag: str, mode: str) -> None:
    sampling = ("deterministic" if summary["sampler"] == "deterministic" else
                f"{summary['sampler'].upper()} ({summary['sampling_timesteps']} steps)")
    lines = [
        f"# Evaluation Report: {run_tag}", "",
        f"- Case: `{summary['case']}`",
        f"- Method: `{summary['method']}`",
        f"- Training samples: `{summary.get('training_samples', 'unknown')}`",
        f"- Mode: `{mode.upper()}`",
        f"- Sampling: `{sampling}`",
        f"- Samples per condition: `{summary['draws_per_condition']}`", "",
    ]
    if summary["method"] == "diffconphy":
        lines[7:7] = [
            f"- Base checkpoint method: `{summary['checkpoint_method']}`",
            f"- Guidance weight: `{summary['guidance_weight']:.6e}`",
            f"- Guidance Huber delta: `{summary['guidance_huber_delta']:.6e}`",
        ]
    if mode == "id":
        lines.extend([
            "## ID Aggregate Error", "",
            "| Metric | Value |", "| --- | ---: |",
            f"| Relative L2 mean | {_format_metric(summary['relative_l2_mean'])} |",
            f"| Relative L2 std | {_format_metric(summary['relative_l2_std'])} |",
            f"| Relative L2 minimum | {_format_metric(summary['relative_l2_min'])} |",
            "",
        ])
    else:
        lines.extend([
            "## OOD Error By Scenario", "",
            "| Scenario | Draws | Best draw | Relative L2 mean | Relative L2 std | Relative L2 minimum |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in rows:
            lines.append(
                f"| {row['label']} | {row['draws']} | {row['best_draw']} | "
                f"{_format_metric(row['relative_l2_mean'])} | {_format_metric(row['relative_l2_std'])} | "
                f"{_format_metric(row['relative_l2_min'])} |"
            )
        lines.append("")
    diagnostics = summary.get("autodiff_residual_diagnostics")
    if diagnostics is not None:
        lines.extend([
            "## Autodiff Residual Diagnostics", "",
            f"- Selected cases: `{diagnostics['case_count']}`",
            f"- Best-draw autodiff PDE L1 mean residual: `{diagnostics['best_pde_mean_absolute']:.6e}`",
            f"- Ensemble autodiff PDE L1 mean residual: `{diagnostics['ensemble_pde_mean_absolute']:.6e}`",
            f"- Best-draw autodiff Robin L1 mean residual: `{diagnostics['best_robin_mean_absolute']:.6e}`",
            f"- Ensemble autodiff Robin L1 mean residual: `{diagnostics['ensemble_robin_mean_absolute']:.6e}`",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_error_summary(mode: str, summary: dict, rows: list[dict]) -> None:
    if mode == "id":
        print(f"[ID] relative_L2: mean={summary['relative_l2_mean']:.6e}, "
              f"std={summary['relative_l2_std']:.6e}", flush=True)
    else:
        for row in rows:
            print(f"[OOD:{row['label']}] samples={row['draws']} relative_L2: "
                  f"mean={row['relative_l2_mean']:.6e}, std={row['relative_l2_std']:.6e}", flush=True)
    diagnostics = summary.get("autodiff_residual_diagnostics")
    if diagnostics is not None:
        print(
            f"[autodiff residuals] cases={diagnostics['case_count']} "
            f"best_PDE_mean_abs={diagnostics['best_pde_mean_absolute']:.6e} "
            f"ensemble_PDE_mean_abs={diagnostics['ensemble_pde_mean_absolute']:.6e} "
            f"best_Robin_mean_abs={diagnostics['best_robin_mean_absolute']:.6e} "
            f"ensemble_Robin_mean_abs={diagnostics['ensemble_robin_mean_absolute']:.6e}",
            flush=True,
        )


def _residual_diagnostic_kind(config, adapter, model, options, *, grid_guidance: bool = False):
    if options is None:
        return None
    if grid_guidance:
        # DiffConPhy modifies the decoded grid after each network call.  The
        # corrected result has no single continuous latent representation, so
        # re-decoding a captured latent would mislabel pre-guidance residuals
        # as residuals of the final sampled field.
        print("[residual diagnostics] skipped: grid-guided DiffConPhy samples have no matching continuous field", flush=True)
        return None
    case = get_case(config["case"])
    if adapter.dimension not in (1, 2) or case.residual != "radiative_diffusion":
        print("[residual diagnostics] skipped: unsupported radiative residual dimension", flush=True)
        return None
    if hasattr(model, "encode") and hasattr(model, "decoder"):
        return "continuous"
    if hasattr(model, "branch") and hasattr(model, "trunk"):
        return "operator"
    print("[residual diagnostics] skipped: this model has no continuous coordinate decoder", flush=True)
    return None


def _model_dtype(model, fallback: torch.dtype) -> torch.dtype:
    parameter = next(model.parameters(), None)
    return fallback if parameter is None else parameter.dtype


def _continuous_sample_field(model, latent):
    """Return the bounded continuous field represented by a sampled latent."""
    return lambda query: model.decoder(latent, query).clamp(-1.0, 1.0)


def _save_residual_diagnostics(*, output: Path, model, kind: str, fields: torch.Tensor,
                                g_left: torch.Tensor | None,
                                axes: dict[str, torch.Tensor], checkpoint: dict, config: dict,
                               rows: list[dict], labels: list[str], selected_cases: tuple[int, ...],
                               latent_cache: dict[tuple[int, int], dict], draw_count: int,
                               options, device: torch.device):
    """Write autodiff residual diagnostics for selected cases and all draws."""
    root = output / "residual_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    # Residual diagnostics call the model through a coordinate-query closure,
    # so make the device contract explicit for the current boundary-inflow
    # case without changing the archived case's evaluation path.
    if config.get("case") == "radiative_2d":
        model.to(device).eval()
    is_2d = "y" in axes
    material_rectangles = tuple(
        (float(region["x"][0]), float(region["x"][1]),
         float(region["y"][0]), float(region["y"][1]))
        for region in config.get("physics", {}).get("material", {}).get("regions", [])
        if is_2d and "x" in region and "y" in region
    )
    records = []
    dtype = _model_dtype(model, fields.dtype)
    with frozen_model_parameters(model):
        for case_index in selected_cases:
            row = rows[case_index]
            best_draw = int(row["best_draw"])
            condition = fields[case_index].to(device=device, dtype=dtype)
            model_timestep = None
            if kind == "continuous":
                captured = latent_cache.get((best_draw, case_index))
                if captured is None:
                    raise RuntimeError("missing final decoder latent for a selected residual diagnostic")
                latent = captured["latent"].to(device=device, dtype=dtype)
                model_timestep = captured.get("timestep")
                normalized_field = _continuous_sample_field(model, latent)
            else:
                branch_condition = condition.unsqueeze(0)
                normalized_field = lambda query, branch_condition=branch_condition: model(
                    branch_condition, None, coords=query,
                    g_left=(None if g_left is None else g_left[case_index:case_index + 1].to(device=device, dtype=dtype)),
                )

            print(f"[residual diagnostics] case={case_index} best_draw={best_draw}", flush=True)
            diagnostics = evaluate_radiative_residual_diagnostics(
                normalized_field, packed_condition=condition, g_left=(
                    None if g_left is None else g_left[case_index].to(device=device, dtype=dtype)
                ), axes=axes,
                scalers=checkpoint["scalers"], config=config, options=options,
            )
            best_pde_mean_absolute = float(diagnostics["pde"]["residual"].abs().mean())
            pde_absolute_sum = diagnostics["pde"]["residual"].abs()
            best_robin_mean_absolute_by_side = {
                side: float(values["residual"].abs().mean())
                for side, values in diagnostics["robin"].items()
            }
            best_robin_mean_absolute = (
                sum(best_robin_mean_absolute_by_side.values())
                / len(best_robin_mean_absolute_by_side)
            )
            robin_absolute_sum = {
                side: values["residual"].abs()
                for side, values in diagnostics["robin"].items()
            }
            for draw_index in range(draw_count):
                if draw_index == best_draw:
                    continue
                if kind == "continuous":
                    captured = latent_cache.get((draw_index, case_index))
                    if captured is None:
                        raise RuntimeError("missing final decoder latent for a selected residual diagnostic")
                    latent = captured["latent"].to(device=device, dtype=dtype)
                    ensemble_field = _continuous_sample_field(model, latent)
                else:
                    branch_condition = condition.unsqueeze(0)
                    ensemble_field = lambda query, branch_condition=branch_condition: model(
                        branch_condition, None, coords=query,
                        g_left=(None if g_left is None else g_left[case_index:case_index + 1].to(device=device, dtype=dtype)),
                    )
                draw_diagnostics = evaluate_radiative_residual_diagnostics(
                    ensemble_field, packed_condition=condition, g_left=(
                        None if g_left is None else g_left[case_index].to(device=device, dtype=dtype)
                    ), axes=axes,
                    scalers=checkpoint["scalers"], config=config, options=options,
                )
                pde_absolute_sum = pde_absolute_sum + draw_diagnostics["pde"]["residual"].abs()
                for side, values in draw_diagnostics["robin"].items():
                    robin_absolute_sum[side] = (
                        robin_absolute_sum[side] + values["residual"].abs()
                    )
            pde_mean_absolute = pde_absolute_sum / float(draw_count)
            ensemble_pde_mean_absolute = float(pde_mean_absolute.mean())
            robin_mean_absolute = {
                side: values / float(draw_count)
                for side, values in robin_absolute_sum.items()
            }
            ensemble_robin_mean_absolute_by_side = {
                side: float(values.mean()) for side, values in robin_mean_absolute.items()
            }
            ensemble_robin_mean_absolute = (
                sum(ensemble_robin_mean_absolute_by_side.values())
                / len(ensemble_robin_mean_absolute_by_side)
            )
            case_output = root / f"case_{case_index:04d}"
            label = labels[case_index]
            if is_2d:
                plot_pde_residual_panels(
                    diagnostics["pde"]["residual"], diagnostics["pde"]["time"],
                    diagnostics["pde"]["y"], diagnostics["pde"]["x"],
                    case_output / "pde_residual.png",
                    title=(f"{label}: signed PDE residual (best draw {best_draw}; "
                           f"mean |r|={best_pde_mean_absolute:.3e})"),
                    material_rectangles=material_rectangles,
                )
                plot_pde_residual_magnitude_panels(
                    pde_mean_absolute, diagnostics["pde"]["time"],
                    diagnostics["pde"]["y"], diagnostics["pde"]["x"],
                    case_output / "pde_residual_mean_absolute.png",
                    title=(f"{label}: mean absolute PDE residual across {draw_count} draws "
                           f"(global mean={ensemble_pde_mean_absolute:.3e})"),
                    material_rectangles=material_rectangles,
                )
            else:
                plot_pde_residual_1d(
                    diagnostics["pde"]["residual"], diagnostics["pde"]["time"],
                    diagnostics["pde"]["x"], case_output / "pde_residual.png",
                    title=(f"{label}: signed PDE residual (best draw {best_draw}; "
                           f"mean |r|={best_pde_mean_absolute:.3e})"),
                )
                plot_pde_residual_magnitude_1d(
                    pde_mean_absolute, diagnostics["pde"]["time"],
                    diagnostics["pde"]["x"], case_output / "pde_residual_mean_absolute.png",
                    title=(f"{label}: mean absolute PDE residual across {draw_count} draws "
                           f"(global mean={ensemble_pde_mean_absolute:.3e})"),
                )
            for side, values in diagnostics["robin"].items():
                if is_2d:
                    plot_robin_residual_panels(
                        values["residual"], values["time"], values["tangent"],
                        case_output / f"robin_{side}_residual.png", side=side,
                        title=f"{label}: Robin residual at {side} boundary (best draw {best_draw})",
                    )
                else:
                    plot_robin_residual_1d(
                        values["residual"], values["time"],
                        case_output / f"robin_{side}_residual.png", side=side,
                        title=f"{label}: Robin residual at {side} boundary (best draw {best_draw})",
                    )
            torch.save(
                {
                    "best_draw": best_draw,
                    "model_timestep": model_timestep,
                    "pde_ensemble": {
                        "mean_absolute_residual": pde_mean_absolute,
                        "draw_count": draw_count,
                    },
                    "robin_ensemble": {
                        "mean_absolute_residual": robin_mean_absolute,
                        "draw_count": draw_count,
                    },
                    "residual_summary": {
                        "best_pde_mean_absolute": best_pde_mean_absolute,
                        "ensemble_pde_mean_absolute": ensemble_pde_mean_absolute,
                        "best_robin_mean_absolute_by_side": best_robin_mean_absolute_by_side,
                        "best_robin_mean_absolute": best_robin_mean_absolute,
                        "ensemble_robin_mean_absolute_by_side": ensemble_robin_mean_absolute_by_side,
                        "ensemble_robin_mean_absolute": ensemble_robin_mean_absolute,
                    },
                    **diagnostics,
                },
                case_output / "residuals.pt",
            )
            records.append({
                "case_index": case_index,
                "label": label,
                "best_draw": best_draw,
                "model_timestep": model_timestep,
                "best_pde_mean_absolute": best_pde_mean_absolute,
                "ensemble_pde_mean_absolute": ensemble_pde_mean_absolute,
                "best_robin_mean_absolute_by_side": best_robin_mean_absolute_by_side,
                "best_robin_mean_absolute": best_robin_mean_absolute,
                "ensemble_robin_mean_absolute_by_side": ensemble_robin_mean_absolute_by_side,
                "ensemble_robin_mean_absolute": ensemble_robin_mean_absolute,
            })
            print(
                f"[residual diagnostics] case={case_index} "
                f"best_PDE_mean_abs={best_pde_mean_absolute:.6e} "
                f"ensemble_PDE_mean_abs={ensemble_pde_mean_absolute:.6e} "
                f"best_Robin_mean_abs={best_robin_mean_absolute:.6e} "
                f"ensemble_Robin_mean_abs={ensemble_robin_mean_absolute:.6e}",
                flush=True,
            )
    if not records:
        return None
    diagnostic_summary = {
        "case_count": len(records),
        "best_pde_mean_absolute": float(np.mean([row["best_pde_mean_absolute"] for row in records])),
        "ensemble_pde_mean_absolute": float(np.mean([row["ensemble_pde_mean_absolute"] for row in records])),
        "best_robin_mean_absolute": float(np.mean([
            row["best_robin_mean_absolute"] for row in records
        ])),
        "ensemble_robin_mean_absolute": float(np.mean([
            row["ensemble_robin_mean_absolute"] for row in records
        ])),
    }
    metadata = {
        "selection_seed": options.seed,
        "selected_cases": records,
        "pde_sampling": (
            {
                "time_points": options.pde_time_points,
                "x_points": options.pde_x_points,
                "total_points": options.pde_time_points * options.pde_x_points,
            }
            if not is_2d else
            {
                "time_points": options.pde_time_points,
                "y_points": options.pde_y_points,
                "x_points": options.pde_x_points,
                "total_points": options.pde_time_points * options.pde_y_points * options.pde_x_points,
            }
        ),
        "robin_sampling_per_side": {
            "time_points": options.boundary_time_points,
            "tangent_points": options.boundary_spatial_points if is_2d else 1,
            "total_points": options.boundary_time_points * (
                options.boundary_spatial_points if is_2d else 1
            ),
        },
        "pde_ensemble": {
            "draw_count": draw_count,
            "statistic": "mean_absolute_residual",
        },
        "robin_ensemble": {
            "draw_count": draw_count,
            "statistic": "mean_absolute_residual",
        },
        "autodiff_residual_summary": diagnostic_summary,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"residual diagnostics: {root}", flush=True)
    return diagnostic_summary


def _evaluate_single(*, checkpoint_path: str | Path, output_dir: str | Path,
                     data_path: str | Path | None, mode: str, device: str, batch_size: int,
                     samples_per_condition: int | None, sampler: str | None,
                     sampling_timesteps: int | None, eta: float | None,
                     scenarios, save_visualizations: bool | None,
                     expected_training_samples: int | None, residual_diagnostics,
                     guidance_weight: float | None, guidance_huber_delta: float):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = deepcopy(checkpoint["config"])
    config["method"] = canonical_method_name(config["method"])
    diffusion_guidance = guidance_weight is not None
    if diffusion_guidance and config.get("method") not in DIFFCONPHY_BASE_METHODS:
        raise ValueError(
            f"DiffConPhy guidance requires a checkpoint from {DIFFCONPHY_BASE_METHODS}, "
            f"got {config.get('method')!r}"
        )
    if diffusion_guidance and float(guidance_weight) <= 0.0:
        raise ValueError("DiffConPhy guidance_weight must be positive")
    trained_samples = config.get("training_dataset", {}).get("samples")
    if expected_training_samples is not None:
        if trained_samples is None:
            raise ValueError(
                "checkpoint does not record its training sample count; select a checkpoint "
                "created by the current training workflow"
            )
        if int(trained_samples) != int(expected_training_samples):
            raise ValueError(
                f"requested a {int(expected_training_samples)}-sample model, but checkpoint "
                f"was trained with {int(trained_samples)} samples"
            )
    evaluation = config.get("evaluation", {})
    adapter = get_case_adapter(config["case"])
    case_data, fields, g_left, axes, labels = _load_data(
        mode, data_path, config, checkpoint, adapter, scenarios
    )
    config["grid_shape"] = list(fields.shape[1:])
    axis_names = ("t", "x") if adapter.dimension == 1 else ("t", "y", "x")
    physical_bounds = (
        tuple(tuple(float(value) for value in config["domain"][name]) for name in axis_names)
        if config.get("case") == "radiative_2d" else None
    )
    grid_coordinates = CoordinateSystem(
        tuple(axes[name] for name in axis_names), physical_bounds=physical_bounds
    ).normalized_axes
    material_condition, material_function = build_material_inputs(config, axes)
    experiment = build_experiment(
        config, grid_coordinates=grid_coordinates,
        material_condition=material_condition, material_function=material_function,
    )
    state = checkpoint.get("ema", {}).get("model") if checkpoint.get("ema") else checkpoint["model"]
    experiment.objective.load_state_dict(state)
    torch_device = choose_device(device)
    experiment.objective.to(torch_device).eval()
    if config.get("case") == "radiative_2d":
        # Protect the operator path if a restored checkpoint leaves a nested
        # branch or trunk submodule on CPU.
        experiment.objective.model.to(torch_device).eval()
    guidance_physics_loss = None
    guidance_boundary_loss = None
    if diffusion_guidance:
        guidance_config = deepcopy(config)
        guidance_config["physics_training"] = {
            "point_sampling": "grid", "huber_delta": float(guidance_huber_delta),
        }
        guidance_physics_loss = build_grid_physics_loss(
            guidance_config, axes, checkpoint["scalers"]
        )
        guidance_boundary_loss = build_grid_boundary_loss(
            guidance_config, axes, checkpoint["scalers"]
        )
    diagnostic_options = parse_residual_diagnostic_options(residual_diagnostics)
    diagnostic_kind = _residual_diagnostic_kind(
        config, adapter, experiment.objective.model, diagnostic_options,
        grid_guidance=diffusion_guidance,
    )
    selected_diagnostic_cases = (
        select_diagnostic_case_indices(len(fields), diagnostic_options)
        if diagnostic_kind is not None else ()
    )
    selected_diagnostic_set = set(selected_diagnostic_cases)
    latent_cache = {}

    stochastic = isinstance(experiment.objective, GaussianDiffusion)
    if not stochastic:
        supplied = {
            "samples_per_condition": samples_per_condition, "sampler": sampler,
            "sampling_timesteps": sampling_timesteps, "eta": eta,
        }
        invalid = sorted(key for key, value in supplied.items() if value is not None)
        if invalid:
            raise ValueError(f"operator-model evaluation does not accept diffusion options: {invalid}")
    draw_count = int(samples_per_condition or evaluation.get("samples_per_condition", 10)) if stochastic else 1
    sampler = sampler or evaluation.get("sampler", "auto")
    if stochastic:
        requested_steps = int(sampling_timesteps or experiment.objective.sampling_timesteps)
        effective_sampler = sampler
        if effective_sampler == "auto":
            effective_sampler = "ddim" if requested_steps < experiment.objective.num_timesteps else "ddpm"
        effective_steps = (experiment.objective.num_timesteps if effective_sampler == "ddpm" else requested_steps)
        run_tag = f"{config['method']}_{effective_sampler}_{effective_steps}steps"
    else:
        effective_sampler, effective_steps = "deterministic", None
        run_tag = f"{config['method']}_deterministic"
    dataset = (
        ConditionedFieldDataset(fields, g_left)
        if g_left is not None else FieldDataset(fields)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    draws = []
    for draw_index in range(draw_count):
        batches = []
        batch_start = 0
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    field_batch = batch["field"].to(torch_device)
                    g_left_batch = batch["g_left"].to(torch_device)
                    condition = {"field": field_batch, "g_left": g_left_batch}
                else:
                    field_batch = batch.to(torch_device)
                    g_left_batch = None
                    condition = field_batch.unsqueeze(1)
                capture = (capture_last_continuous_latent(experiment.objective.model)
                           if diagnostic_kind == "continuous" else nullcontext(None))
                with capture as captured:
                    if stochastic:
                        prediction = experiment.objective.sample(
                            condition, sampler=sampler, sampling_timesteps=sampling_timesteps, eta=eta,
                            guidance_weight=float(guidance_weight or 0.0),
                            guidance_physics_loss_fn=guidance_physics_loss,
                            guidance_boundary_loss_fn=guidance_boundary_loss,
                        )
                    else:
                        prediction = experiment.objective.predict(condition)
                if captured is not None and captured.get("latent") is not None:
                    for local_index in range(field_batch.shape[0]):
                        case_index = batch_start + local_index
                        if case_index not in selected_diagnostic_set:
                            continue
                        timestep = captured.get("timestep")
                        latent_cache[(draw_index, case_index)] = {
                            "latent": captured["latent"][local_index:local_index + 1].detach().cpu().clone(),
                            "timestep": (None if timestep is None else int(timestep[local_index].item())),
                        }
                batches.append(prediction[:, 0].cpu())
                batch_start += field_batch.shape[0]
        draws.append(torch.cat(batches))
    normalized = torch.stack(draws)
    physical = torch.stack([
        _physical_interior(
            draw, checkpoint, adapter.dimension,
            keep_spatial_boundaries=config.get("case") == "radiative_2d",
        )
        for draw in normalized
    ]).numpy()
    summary, rows, best, relative_l2 = evaluate_draws(adapter, physical, case_data)
    summary.update({"case": config["case"], "method": config["method"], "mode": mode,
                    "sampler": effective_sampler, "sampling_timesteps": effective_steps,
                    "labels": labels, "training_samples": trained_samples})
    if diffusion_guidance:
        summary.update({
            "method": "diffconphy",
            "guidance_weight": float(guidance_weight),
            "guidance_huber_delta": float(guidance_huber_delta),
            "checkpoint_method": config["method"],
            "checkpoint_physics_weights": {
                key: float(config.get("physics_training", {}).get(key, 0.0))
                for key in ("lambda_phy", "lambda_bc")
                if key in config.get("physics_training", {})
            },
        })
    for row, label in zip(rows, labels):
        row["label"] = label

    if diffusion_guidance:
        run_tag = run_tag.replace(config["method"], "diffconphy", 1)
    output = Path(output_dir) / run_tag / mode
    output.mkdir(parents=True, exist_ok=True)
    make_figures = evaluation.get("save_visualizations", True) if save_visualizations is None else save_visualizations
    if make_figures:
        numpy_axes = {key: np.asarray(value) for key, value in case_data.axes.items()}
        for index, (label, row) in enumerate(zip(labels, rows)):
            use_radiative_2d_plot = config.get("case") in {
                "radiative_2d", "radiative_initial_2d"
            }
            if use_radiative_2d_plot:
                plotter = plot_radiative_2d_prediction
            elif config.get("case") == "radiative_1d":
                plotter = plot_radiative_1d_prediction
            else:
                plotter = plot_prediction
            plot_kwargs = {
                "title": (f"{label}: best draw {row['best_draw']}, "
                           f"relative L2={row['relative_l2_min']:.3e}")
            }
            if use_radiative_2d_plot:
                plot_kwargs["energy_scale"] = (
                    "per_time"
                    if config.get("case") == "radiative_initial_2d"
                    else "global"
                )
                if config.get("case") == "radiative_initial_2d":
                    plot_kwargs["first_time"] = 0.15
            plotter(
                case_data.fields[index], best[index], numpy_axes,
                output / "figures" / f"{label}_best.png", **plot_kwargs,
            )
    if diagnostic_kind is not None:
        summary["autodiff_residual_diagnostics"] = _save_residual_diagnostics(
            output=output, model=experiment.objective.model, kind=diagnostic_kind,
            fields=fields, g_left=g_left, axes=axes, checkpoint=checkpoint, config=config, rows=rows,
            labels=labels, selected_cases=selected_diagnostic_cases, latent_cache=latent_cache,
            draw_count=draw_count, options=diagnostic_options, device=torch_device,
        )
    torch.save({
        "prediction_normalized": normalized, "prediction_physical": torch.from_numpy(physical),
        "best_prediction": torch.from_numpy(best), "relative_l2": torch.from_numpy(relative_l2),
        "autodiff_residual_summary": summary.get("autodiff_residual_diagnostics"),
        "scalers": checkpoint["scalers"],
        "axes": axes, "labels": labels, "summary": summary, "per_case": rows,
    }, output / f"{run_tag}_{mode}_predictions.pt")
    report_path = output / f"{run_tag}_{mode}_report.md"
    _write_markdown_report(report_path, summary, rows, run_tag, mode)
    _print_error_summary(mode, summary, rows)
    print(f"report: {report_path}", flush=True)
    return summary


def evaluate_checkpoint(*, checkpoint_path: str | Path, output_dir: str | Path,
                        data_path: str | Path | None = None, mode: str = "id",
                        device: str = "auto", batch_size: int = 8,
                        samples_per_condition: int | None = None, sampler: str | None = None,
                        sampling_timesteps: int | None = None, eta: float | None = None,
                        scenarios=None, save_visualizations: bool | None = None,
                        expected_training_samples: int | None = None,
                        residual_diagnostics=None, guidance_weight: float | None = None,
                        guidance_huber_delta: float = 1.0):
    """Evaluate ID, OOD, or both without allowing their artifacts to overwrite."""
    mode = "id" if mode == "dataset" else mode
    if mode not in ("id", "ood", "both"):
        raise ValueError("mode must be 'id', 'ood' or 'both'")
    modes = ("id", "ood") if mode == "both" else (mode,)
    results = {}
    for selected in modes:
        if selected == "ood" and scenarios is not None and not tuple(scenarios):
            print("[OOD] skipped: this case has no configured OOD scenarios", flush=True)
            continue
        results[selected] = _evaluate_single(
            checkpoint_path=checkpoint_path, output_dir=output_dir, data_path=data_path,
            mode=selected, device=device, batch_size=batch_size,
            samples_per_condition=samples_per_condition, sampler=sampler,
            sampling_timesteps=sampling_timesteps, eta=eta,
            scenarios=scenarios, save_visualizations=save_visualizations,
            expected_training_samples=expected_training_samples,
            residual_diagnostics=residual_diagnostics,
            guidance_weight=guidance_weight,
            guidance_huber_delta=guidance_huber_delta,
        )
    if not results:
        raise ValueError("no evaluation mode could run with the configured scenarios")
    return results if mode == "both" else results[mode]
