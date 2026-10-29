"""Shared training workflow used by every case study."""

from __future__ import annotations

import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pifdm.cases import get_case_adapter
from pifdm.data.datasets import (
    ConditionedFieldDataset, FieldDataset, load_hdf5_field,
    load_radiative_2d_field,
)
from pifdm.experiments.naming import CONTINUOUS_DIFFUSION_METHODS, canonical_method_name
from pifdm.experiments.physics import (
    build_boundary_loss, build_grid_boundary_loss, build_grid_physics_loss,
    build_material_inputs, build_physics_loss, build_fused_physics_losses,
)
from pifdm.experiments.contracts import GRID_PHYSICS_METHODS, validate_experiment_config
from pifdm.experiments.runner import build_experiment
from pifdm.physics.coordinates import CoordinateSystem


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _startup_log(started_at: float, message: str) -> None:
    print(f"[startup +{time.perf_counter() - started_at:8.2f}s] {message}", flush=True)


def _move_batch(batch, device: torch.device):
    """Move tensor or structured case batch without changing its layout."""
    if isinstance(batch, dict):
        return {key: value.to(device) for key, value in batch.items()}
    return batch.to(device)


def _coordinate_bounds(config: dict, axis_names: tuple[str, ...]):
    if config.get("case") != "radiative_2d":
        return None
    return tuple(tuple(float(value) for value in config["domain"][name]) for name in axis_names)


def _configure_nonnegative_output(config: dict, scalers: dict) -> None:
    """Configure a positive latent output in normalized model coordinates."""
    if not config.get("physics", {}).get("nonnegative_solution", False):
        return

    minimum = torch.as_tensor(scalers["min"]).float()
    maximum = torch.as_tensor(scalers["max"]).float()
    value_range = (maximum - minimum).clamp_min(1e-12)
    model = config.setdefault("model", {})
    model["output_floor"] = float(2.0 * (0.0 - minimum) / value_range - 1.0)

    range_fraction = model.get("output_softplus_range_fraction")
    if range_fraction is None:
        if config.get("method") in CONTINUOUS_DIFFUSION_METHODS:
            # A continuous decoder has an extra MLP head.  Its default
            # normalized-zero output is a poor starting point for sparse
            # positive fields whose typical value is close to physical zero.
            range_fraction = 1.0
        else:
            model.setdefault("output_softplus_beta", 10.0)
            return
    range_fraction = float(range_fraction)
    if not np.isfinite(range_fraction) or range_fraction <= 0.0:
        raise ValueError("model.output_softplus_range_fraction must be finite and positive")

    initial_value = float(model.get("output_softplus_initial_value", 1.0))
    if not np.isfinite(initial_value) or initial_value <= 0.0:
        raise ValueError("model.output_softplus_initial_value must be finite and positive")
    scaled_initial = initial_value / (float(value_range) * range_fraction)
    if scaled_initial > 20.0:
        initial_bias = scaled_initial
    else:
        initial_bias = math.log(math.expm1(scaled_initial))

    # E_pred = range_fraction * data_range * softplus(latent).  In normalized
    # coordinates this becomes floor + 2 * range_fraction * softplus(latent).
    # Initializing the latent bias from a physical reference value avoids both
    # the large positive prediction of a zero logit and an all-zero attractor.
    model["output_softplus_beta"] = 1.0
    model["output_softplus_scale"] = 2.0 * range_fraction
    model["output_softplus_initial_bias"] = initial_bias


def _save_checkpoint(path, experiment, config, scalers, axes, progress, progress_unit,
                     metadata, history):
    payload = {
        "model": experiment.objective.state_dict(),
        "optimizer": experiment.trainer.optimizer.state_dict(),
        "scheduler": experiment.trainer.lr_scheduler.state_dict() if experiment.trainer.lr_scheduler else None,
        "scaler": experiment.trainer.scaler.state_dict(),
        "ema": experiment.trainer.ema.state_dict() if experiment.trainer.ema else None,
        "config": config,
        "scalers": {key: value for key, value in scalers.items() if key != "missing_boundaries"},
        "axes": axes,
        "progress": progress,
        "progress_unit": progress_unit,
        # Kept for loading early refactor checkpoints; new code reads progress.
        "step": progress,
        "case_metadata": metadata,
        "history": history,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_experiment(config: dict, *, data_path: str | Path, output_dir: str | Path,
                     device: str = "auto", duration: int | None = None,
                     resume: str | Path | None = None, extra_duration: int | None = None):
    """Train every method for an explicit number of optimizer-input steps."""
    startup = time.perf_counter()
    data_path = Path(data_path)
    _startup_log(startup, f"starting {config.get('case')}/{config.get('method')} from {data_path}")
    config = deepcopy(config)
    validate_experiment_config(config)
    method = config["method"]
    progress_unit = "step"

    seed_everything(int(config.get("seed", 42)))
    torch_device = choose_device(device)
    if torch_device.type == "cuda":
        device_summary = f"cuda, torch={torch.__version__}, gpu={torch.cuda.get_device_name(torch_device)}"
    else:
        device_summary = f"{torch_device}, torch={torch.__version__}"
    _startup_log(startup, f"device selected: {device_summary}")
    checkpoint = torch.load(resume, map_location=torch_device) if resume else None
    if checkpoint is not None:
        _startup_log(startup, f"resume checkpoint loaded: {resume}")
    if checkpoint is not None:
        for key in ("case", "method"):
            checkpoint_value = checkpoint["config"].get(key)
            if key == "method":
                checkpoint_value = canonical_method_name(checkpoint_value)
            if checkpoint_value != config.get(key):
                raise ValueError(f"resume checkpoint {key} does not match current settings")
        requested_samples = config.get("training_dataset", {}).get("samples")
        checkpoint_samples = checkpoint["config"].get("training_dataset", {}).get("samples")
        if (requested_samples is not None and checkpoint_samples is not None
                and int(requested_samples) != int(checkpoint_samples)):
            raise ValueError(
                f"resume checkpoint was trained with {int(checkpoint_samples)} samples, "
                f"not the requested {int(requested_samples)}"
            )

    adapter = get_case_adapter(config["case"])
    with h5py.File(data_path, "r") as handle:
        case_metadata = dict(handle.attrs)
    if config["case"] == "radiative_2d":
        fields, g_left, axes, scalers = load_radiative_2d_field(
            data_path, value_scalers=checkpoint.get("scalers") if checkpoint else None,
        )
        dataset = ConditionedFieldDataset(fields, g_left)
    else:
        fields, axes, scalers = load_hdf5_field(
            data_path, add_boundaries=True, domain=config["domain"],
            value_scalers=checkpoint.get("scalers") if checkpoint else None,
        )
        dataset = FieldDataset(fields)
    expected_axes = ("t", "x") if adapter.dimension == 1 else ("t", "y", "x")
    if fields.ndim != adapter.dimension + 2:
        raise ValueError(f"{adapter.name}: invalid packed training shape {tuple(fields.shape)}")
    for axis, size in zip(expected_axes, fields.shape[1:]):
        if axis not in axes or len(axes[axis]) != size:
            raise ValueError(f"{adapter.name}: axis {axis!r} does not match {tuple(fields.shape)}")
    if not torch.isfinite(fields).all():
        raise ValueError(f"{adapter.name}: training field contains NaN or infinity")
    selected_dataset = dict(config.get("training_dataset", {}))
    expected_samples = selected_dataset.get("samples")
    if expected_samples is not None and int(expected_samples) != len(fields):
        raise ValueError(
            f"selected training dataset expects {int(expected_samples)} samples, "
            f"but {data_path} contains {len(fields)}"
        )
    # The checkpoint carries the exact data provenance and normalization
    # scalers together, so evaluation cannot silently recompute test scalers.
    config["training_dataset"] = {
        **selected_dataset,
        "samples": int(len(fields)),
        "path": str(data_path.resolve()),
    }
    _configure_nonnegative_output(config, scalers)
    _startup_log(startup, f"dataset loaded once: shape={tuple(fields.shape)}, size={data_path.stat().st_size / 2**20:.1f} MiB")
    config["grid_shape"] = list(fields.shape[1:])
    training = config.get("training", {})
    num_workers = int(training.get("num_workers", 0))
    loader = DataLoader(
        dataset, batch_size=int(training.get("batch_size", 8)), shuffle=True,
        num_workers=num_workers, pin_memory=torch_device.type == "cuda",
    )
    _startup_log(startup, f"DataLoader ready: batches={len(loader)}, num_workers={num_workers}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    start_progress, history = 0, []
    if checkpoint is not None:
        checkpoint_unit = checkpoint.get("progress_unit", "step")
        if checkpoint_unit != progress_unit:
            raise ValueError(f"checkpoint records {checkpoint_unit}s, current method uses {progress_unit}s")
        start_progress = int(checkpoint.get("progress", checkpoint.get("step", 0)))
        history = list(checkpoint.get("history", []))
    if extra_duration is not None:
        if checkpoint is None or extra_duration <= 0:
            raise ValueError("extra_duration requires resume and must be positive")
        total_progress = start_progress + extra_duration
    else:
        configured_duration = training["steps"]
        total_progress = int(duration or configured_duration)
    if start_progress >= total_progress:
        raise ValueError(f"resume {progress_unit} {start_progress} is not below total {total_progress}")

    accumulation = max(1, int(training.get("gradient_accumulation_steps", 1)))
    scheduler_updates = (total_progress + accumulation - 1) // accumulation
    if method in GRID_PHYSICS_METHODS:
        physics_loss = build_grid_physics_loss(config, axes, scalers)
        boundary_loss = build_grid_boundary_loss(config, axes, scalers)
    elif config.get("fuse_physics_queries", False):
        physics_loss, boundary_loss = build_fused_physics_losses(config, axes, scalers)
    else:
        physics_loss = build_physics_loss(config, axes, scalers)
        boundary_loss = build_boundary_loss(config, axes, scalers)
    grid_coordinates = CoordinateSystem(
        tuple(axes[name] for name in expected_axes),
        physical_bounds=_coordinate_bounds(config, expected_axes),
    ).normalized_axes
    material_condition, material_function = build_material_inputs(config, axes)
    experiment = build_experiment(config, physics_loss_fn=physics_loss,
                                  boundary_loss_fn=boundary_loss,
                                  scheduler_updates=scheduler_updates,
                                  grid_coordinates=grid_coordinates,
                                  material_condition=material_condition,
                                  material_function=material_function)
    parameter_count = sum(parameter.numel() for parameter in experiment.objective.parameters())
    _startup_log(startup, f"model, optimizer and EMA built: {parameter_count:,} trainable parameters")
    experiment.objective.to(torch_device)
    if experiment.trainer.ema is not None:
        experiment.trainer.ema.to(torch_device)
    _startup_log(startup, "model and EMA moved to device")

    if checkpoint is not None:
        experiment.objective.load_state_dict(checkpoint["model"])
        experiment.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        if experiment.trainer.lr_scheduler and checkpoint.get("scheduler"):
            experiment.trainer.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            experiment.trainer.scaler.load_state_dict(checkpoint["scaler"])
        if experiment.trainer.ema and checkpoint.get("ema"):
            experiment.trainer.ema.load_state_dict(checkpoint["ema"])
        _startup_log(startup, "checkpoint state restored")

    save_every = int(training.get("save_every", 0))
    log_every = int(training.get("log_every", 100))
    progress = tqdm(
        total=total_progress, initial=start_progress,
        desc=f"{config['case']}/{method}", unit=progress_unit,
        file=sys.stdout, mininterval=5.0, maxinterval=30.0,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    iterator = iter(loader)
    _startup_log(startup, "DataLoader iterator created; entering first optimization step")
    for step in range(start_progress, total_progress):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        losses = experiment.trainer.step(
            _move_batch(batch, torch_device), step, final_step=step + 1 == total_progress
        )
        progress.update(1)
        if step % log_every == 0:
            record = _loss_record("step", step, losses, experiment)
            history.append(record)
            progress.set_postfix(total=f"{record['total_loss']:.3e}",
                                 phy=f"{record['physics_loss']:.3e}",
                                 bc=f"{record['boundary_loss']:.3e}", refresh=False)
            _print_record(record, progress)
        completed = step + 1
        if save_every and completed % save_every == 0:
            _save_checkpoint(output / f"checkpoint_step_{completed}.pt", experiment, config,
                             scalers, axes, completed, progress_unit,
                             case_metadata, history)
            _write_history(output, history)

    progress.close()
    _save_checkpoint(output / "checkpoint.pt", experiment, config, scalers, axes,
                     total_progress, progress_unit, case_metadata, history)
    _write_history(output, history)
    print(f"saved checkpoint: {output / 'checkpoint.pt'}")
    return output / "checkpoint.pt"


def _loss_record(unit, index, losses, experiment):
    return {
        unit: index,
        "total_loss": float(losses["total_loss"].detach().cpu()),
        "reconstruction_loss": float(losses["L_recon"].detach().cpu()),
        "physics_loss": float(losses["L_phy"].detach().cpu()),
        "boundary_loss": float(losses["L_bc"].detach().cpu()),
        "learning_rate": float(experiment.trainer.optimizer.param_groups[0]["lr"]),
    }


def _print_record(record, progress=None):
    unit = "step"
    message = (f"{unit}={record[unit]} total={record['total_loss']:.6e} "
               f"recon={record['reconstruction_loss']:.6e} "
               f"phy={record['physics_loss']:.6e} bc={record['boundary_loss']:.6e}")
    if progress is None:
        print(message, flush=True)
    else:
        progress.write(message, file=sys.stdout)


def _write_history(output, history):
    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
