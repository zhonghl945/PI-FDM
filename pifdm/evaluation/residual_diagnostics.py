"""Continuous residual diagnostics used only during checkpoint evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType

import torch

from pifdm.experiments.physics import _material_function
from pifdm.physics.coordinates import CoordinateSystem
from pifdm.physics.residuals import radiative_diffusion_residual, radiative_robin_residual


@dataclass(frozen=True)
class ResidualDiagnosticOptions:
    """Fixed-resolution residual plots for a small random evaluation subset."""

    case_count: int = 3
    seed: int = 42
    pde_time_points: int = 10
    pde_y_points: int = 50
    pde_x_points: int = 50
    boundary_time_points: int = 10
    boundary_spatial_points: int = 100

    def __post_init__(self):
        if self.case_count < 1:
            raise ValueError("residual diagnostics require at least one case")
        if self.pde_time_points < 1 or self.pde_y_points < 1 or self.pde_x_points < 1:
            raise ValueError("PDE diagnostics require positive time and spatial dimensions")
        if self.boundary_time_points < 1 or self.boundary_spatial_points < 1:
            raise ValueError("Robin diagnostics require positive time and boundary dimensions")


def parse_residual_diagnostic_options(options) -> ResidualDiagnosticOptions | None:
    """Parse an opt-in evaluation setting without affecting normal evaluation."""
    if options is None or options is False:
        return None
    if options is True:
        return ResidualDiagnosticOptions()
    if not isinstance(options, Mapping):
        raise TypeError("residual_diagnostics must be a mapping, True, False or None")
    values = dict(options)
    if not values.pop("enabled", True):
        return None
    aliases = {
        "pde_spatial_shape": ("pde_y_points", "pde_x_points"),
    }
    if "pde_spatial_points" in values:
        values["pde_x_points"] = int(values.pop("pde_spatial_points"))
    for key, destination in aliases.items():
        if key in values:
            shape = tuple(values.pop(key))
            if len(shape) != 2:
                raise ValueError("residual_diagnostics.pde_spatial_shape must be (y_points, x_points)")
            values[destination[0]], values[destination[1]] = map(int, shape)
    allowed = set(ResidualDiagnosticOptions.__dataclass_fields__)
    unexpected = sorted(set(values).difference(allowed))
    if unexpected:
        raise ValueError(f"unknown residual_diagnostics options: {unexpected}")
    return ResidualDiagnosticOptions(**values)


def select_diagnostic_case_indices(total_cases: int, options: ResidualDiagnosticOptions) -> tuple[int, ...]:
    """Select a reproducible random subset, preserving dataset order in artifacts."""
    if total_cases < 1:
        raise ValueError("cannot select residual diagnostics from an empty dataset")
    generator = torch.Generator().manual_seed(int(options.seed))
    count = min(int(options.case_count), int(total_cases))
    return tuple(sorted(torch.randperm(total_cases, generator=generator)[:count].tolist()))


def _interior_limits(axis: torch.Tensor, *, skip_initial: bool = False,
                     exclude_endpoints: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    if axis.ndim != 1 or len(axis) < 3:
        raise ValueError("residual diagnostics require axes with at least three saved coordinates")
    margin = torch.finfo(axis.dtype).eps * 32
    lower = axis[1] if (skip_initial or exclude_endpoints) else axis[0] + margin
    upper = axis[-2] if exclude_endpoints else axis[-1] - margin
    if not lower < upper:
        raise ValueError("diagnostic axis has no open interior")
    return lower, upper


def build_interior_coordinates(normalized_axes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                               options: ResidualDiagnosticOptions, *, device, dtype):
    """Build the fixed uniform interior grid used for PDE plots."""
    time_axis, y_axis, x_axis = (axis.to(device=device, dtype=dtype) for axis in normalized_axes)
    time_lo, time_hi = _interior_limits(time_axis, skip_initial=True)
    y_lo, y_hi = _interior_limits(y_axis, exclude_endpoints=True)
    x_lo, x_hi = _interior_limits(x_axis, exclude_endpoints=True)
    times = torch.linspace(time_lo, time_hi, options.pde_time_points, device=device, dtype=dtype)
    y_values = torch.linspace(y_lo, y_hi, options.pde_y_points, device=device, dtype=dtype)
    x_values = torch.linspace(x_lo, x_hi, options.pde_x_points, device=device, dtype=dtype)
    time_grid, y_grid, x_grid = torch.meshgrid(times, y_values, x_values, indexing="ij")
    coordinates = torch.stack((time_grid, y_grid, x_grid), dim=-1).reshape(1, -1, 3)
    return coordinates.requires_grad_(True), times, y_values, x_values


def build_boundary_coordinates(normalized_axes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                               options: ResidualDiagnosticOptions, *, device, dtype):
    """Build four configurable open-boundary grids and their outward normals."""
    time_axis, y_axis, x_axis = (axis.to(device=device, dtype=dtype) for axis in normalized_axes)
    time_lo, time_hi = _interior_limits(time_axis, skip_initial=True)
    y_lo, y_hi = _interior_limits(y_axis, exclude_endpoints=True)
    x_lo, x_hi = _interior_limits(x_axis, exclude_endpoints=True)
    times = torch.linspace(time_lo, time_hi, options.boundary_time_points, device=device, dtype=dtype)
    y_values = torch.linspace(y_lo, y_hi, options.boundary_spatial_points, device=device, dtype=dtype)
    x_values = torch.linspace(x_lo, x_hi, options.boundary_spatial_points, device=device, dtype=dtype)
    time_grid_y, tangent_y = torch.meshgrid(times, y_values, indexing="ij")
    time_grid_x, tangent_x = torch.meshgrid(times, x_values, indexing="ij")

    def make(time_grid, tangent, *, y, x, normal):
        coordinates = torch.stack((time_grid, y, x), dim=-1).reshape(1, -1, 3)
        normal_values = torch.as_tensor(normal, dtype=dtype, device=device).reshape(1, 1, 2)
        normal_values = normal_values.expand(1, coordinates.shape[1], -1)
        return coordinates.requires_grad_(True), normal_values, tangent

    return {
        "left": make(time_grid_y, tangent_y, y=tangent_y, x=tangent_y * 0.0 + x_axis[0], normal=(0.0, -1.0)),
        "right": make(time_grid_y, tangent_y, y=tangent_y, x=tangent_y * 0.0 + x_axis[-1], normal=(0.0, 1.0)),
        "bottom": make(time_grid_x, tangent_x, y=tangent_x * 0.0 + y_axis[0], x=tangent_x, normal=(-1.0, 0.0)),
        "top": make(time_grid_x, tangent_x, y=tangent_x * 0.0 + y_axis[-1], x=tangent_x, normal=(1.0, 0.0)),
    }, times, {"left": y_values, "right": y_values, "bottom": x_values, "top": x_values}


def build_1d_interior_coordinates(
    normalized_axes: tuple[torch.Tensor, torch.Tensor],
    options: ResidualDiagnosticOptions, *, device, dtype,
):
    """Build a uniform open interior grid for a ``(t, x)`` decoder."""
    time_axis, x_axis = (axis.to(device=device, dtype=dtype) for axis in normalized_axes)
    time_lo, time_hi = _interior_limits(time_axis, skip_initial=True)
    x_lo, x_hi = _interior_limits(x_axis, exclude_endpoints=True)
    times = torch.linspace(time_lo, time_hi, options.pde_time_points, device=device, dtype=dtype)
    x_values = torch.linspace(x_lo, x_hi, options.pde_x_points, device=device, dtype=dtype)
    time_grid, x_grid = torch.meshgrid(times, x_values, indexing="ij")
    coordinates = torch.stack((time_grid, x_grid), dim=-1).reshape(1, -1, 2)
    return coordinates.requires_grad_(True), times, x_values


def build_1d_boundary_coordinates(
    normalized_axes: tuple[torch.Tensor, torch.Tensor],
    options: ResidualDiagnosticOptions, *, device, dtype,
):
    """Build the left and right open-boundary query lines for 1D Robin data."""
    time_axis, x_axis = (axis.to(device=device, dtype=dtype) for axis in normalized_axes)
    time_lo, time_hi = _interior_limits(time_axis, skip_initial=True)
    times = torch.linspace(time_lo, time_hi, options.boundary_time_points, device=device, dtype=dtype)
    boundaries = {}
    for side, x_value, normal in (
        ("left", x_axis[0], -1.0),
        ("right", x_axis[-1], 1.0),
    ):
        coordinates = torch.stack((times, times * 0.0 + x_value), dim=-1).reshape(1, -1, 2)
        normal_values = torch.full(
            (1, len(times), 1), normal, device=device, dtype=dtype
        )
        boundaries[side] = (
            coordinates.requires_grad_(True), normal_values,
            torch.full((1,), x_value, device=device, dtype=dtype),
        )
    return boundaries, times


def _bracket(axis: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    upper = torch.searchsorted(axis, query.contiguous(), right=False).clamp(1, len(axis) - 1)
    lower = upper - 1
    fraction = (query - axis[lower]) / (axis[upper] - axis[lower]).clamp_min(torch.finfo(axis.dtype).eps)
    return lower, upper, fraction


def interpolate_boundary_values(values: torch.Tensor, time_axis: torch.Tensor, tangent_axis: torch.Tensor,
                                time_query: torch.Tensor, tangent_query: torch.Tensor) -> torch.Tensor:
    """Bilinearly interpolate saved Robin data at continuous boundary queries."""
    if values.ndim != 2:
        raise ValueError("boundary values must be shaped (time, tangent)")
    time_query, tangent_query = time_query.reshape(-1), tangent_query.reshape(-1)
    t0, t1, wt = _bracket(time_axis, time_query)
    s0, s1, ws = _bracket(tangent_axis, tangent_query)
    lower = values[t0, s0] * (1.0 - ws) + values[t0, s1] * ws
    upper = values[t1, s0] * (1.0 - ws) + values[t1, s1] * ws
    return lower * (1.0 - wt) + upper * wt


def _physical_axis(normalized: torch.Tensor, physical_axis: torch.Tensor,
                   bounds: tuple[float, float] | None = None) -> torch.Tensor:
    physical_axis = physical_axis.to(device=normalized.device, dtype=normalized.dtype)
    if bounds is None:
        lower, upper = physical_axis[0], physical_axis[-1]
    else:
        lower = physical_axis.new_tensor(float(bounds[0]))
        upper = physical_axis.new_tensor(float(bounds[1]))
    return (normalized + 1.0) * 0.5 * (upper - lower) + lower


def _boundary_values(packed: torch.Tensor, side: str) -> torch.Tensor:
    if packed.ndim == 2:
        if side == "left":
            return packed[:, 0]
        if side == "right":
            return packed[:, -1]
        raise ValueError(f"unknown 1D boundary side {side!r}")
    if side == "left":
        return packed[:, :, 0]
    if side == "right":
        return packed[:, :, -1]
    if side == "bottom":
        return packed[:, 0, :]
    if side == "top":
        return packed[:, -1, :]
    raise ValueError(f"unknown boundary side {side!r}")


def _residual_kwargs(config: dict, axes: dict[str, torch.Tensor]) -> dict:
    axis_names = ("t", "y", "x") if "y" in axes else ("t", "x")
    physical_bounds = (
        tuple(tuple(float(value) for value in config["domain"][name]) for name in axis_names)
        if config.get("case") == "radiative_2d" else None
    )
    coordinates = CoordinateSystem(
        tuple(axes[name] for name in axis_names), physical_bounds=physical_bounds
    )
    scales = coordinates.derivative_scales
    physics = config.get("physics", {})
    source = physics.get("source", {"type": "constant", "value": 0.0})
    if source.get("type", "constant") != "constant":
        raise ValueError("residual diagnostics currently require a constant radiative source")
    return {
        "dt_scale": float(scales[0]),
        "dy_scale": float(scales[1]) if len(scales) == 3 else None,
        "dx_scale": float(scales[-1]),
        "material": _material_function(physics.get("material", {}), coordinates),
        "source": float(source.get("value", 0.0)),
        "use_flux_limiter": physics.get("use_flux_limiter", True),
        "limiter_norm": physics.get("limiter_norm", "l2_smooth"),
    }


def evaluate_radiative_residual_diagnostics(
    normalized_field: Callable[[torch.Tensor], torch.Tensor], *, packed_condition: torch.Tensor,
    g_left: torch.Tensor | None = None,
    axes: dict[str, torch.Tensor], scalers: dict, config: dict,
    options: ResidualDiagnosticOptions, include_robin: bool = True,
) -> dict:
    """Evaluate a signed strong-form PDE residual and, optionally, Robin residuals."""
    if packed_condition.ndim not in (2, 3):
        raise ValueError("packed_condition must be (time, x) or (time, y, x)")
    is_1d = packed_condition.ndim == 2
    device, dtype = packed_condition.device, packed_condition.dtype
    axis_names = ("t", "x") if is_1d else ("t", "y", "x")
    physical_bounds = (
        tuple(tuple(float(value) for value in config["domain"][name]) for name in axis_names)
        if config.get("case") == "radiative_2d" else None
    )
    coordinate_system = CoordinateSystem(
        tuple(axes[name] for name in axis_names), physical_bounds=physical_bounds
    )
    normalized_axes = coordinate_system.normalized_axes
    residual_kwargs = _residual_kwargs(config, axes)
    minimum = torch.as_tensor(scalers["min"], device=device, dtype=dtype)
    maximum = torch.as_tensor(scalers["max"], device=device, dtype=dtype)

    def physical_field(query: torch.Tensor) -> torch.Tensor:
        values = normalized_field(query)
        return (values + 1.0) * 0.5 * (maximum - minimum) + minimum

    with torch.enable_grad():
        if is_1d:
            interior, times, x_values = build_1d_interior_coordinates(
                normalized_axes, options, device=device, dtype=dtype
            )
        else:
            interior, times, y_values, x_values = build_interior_coordinates(
                normalized_axes, options, device=device, dtype=dtype
            )
        pde_residual = radiative_diffusion_residual(physical_field, interior, **residual_kwargs)
        pde_residual = pde_residual[0, :, 0].reshape(
            options.pde_time_points, *(
                (options.pde_x_points,) if is_1d else
                (options.pde_y_points, options.pde_x_points)
            )
        ).detach().cpu()

        result = {"pde": {
            "residual": pde_residual,
            "time": _physical_axis(
                times, axes["t"], None if physical_bounds is None else physical_bounds[0]
            ).detach().cpu(),
            "x": _physical_axis(
                x_values, axes["x"], None if physical_bounds is None else physical_bounds[-1]
            ).detach().cpu(),
        }}
        if not is_1d:
            result["pde"]["y"] = _physical_axis(
                y_values, axes["y"], None if physical_bounds is None else physical_bounds[1]
            ).detach().cpu()
        if not include_robin:
            return result

        boundary_residuals = {}
        tangents = {}
        if is_1d:
            boundaries, boundary_times = build_1d_boundary_coordinates(
                normalized_axes, options, device=device, dtype=dtype
            )
        elif config.get("case") == "radiative_2d":
            if g_left is None:
                raise ValueError("radiative_2d Robin diagnostics require the saved g_left condition")
            if g_left.ndim == 3 and g_left.shape[0] == 1:
                g_left = g_left[0]
            expected = (len(axes["t"]), len(axes["y"]))
            if g_left.ndim != 2 or tuple(g_left.shape) != expected:
                raise ValueError(f"radiative_2d g_left must have shape {expected}, got {tuple(g_left.shape)}")
            # ``axes`` originate from the HDF5 loader and are CPU tensors.
            # The special left-Robin grid is built directly here (unlike the
            # generic diagnostic grids), so move both axes explicitly before
            # constructing coordinate queries for a CUDA model.
            time_values = normalized_axes[0][1:].to(device=device, dtype=dtype)
            tangent_values = normalized_axes[1].to(device=device, dtype=dtype)
            time_grid, tangent_grid = torch.meshgrid(
                time_values, tangent_values, indexing="ij"
            )
            coordinates = torch.stack((
                time_grid, tangent_grid, torch.full_like(time_grid, -1.0),
            ), dim=-1).reshape(1, -1, 3).requires_grad_(True)
            normal = torch.zeros(
                1, coordinates.shape[1], 2, device=device, dtype=dtype
            )
            normal[..., 1] = -1.0
            rhs = g_left[1:].to(device=device, dtype=dtype).reshape(1, -1, 1)
            boundary_residuals = {
                "left": radiative_robin_residual(
                physical_field, coordinates, normal,
                    (rhs + 1.0) * 0.5 * (maximum - minimum) + minimum,
                    **{key: value for key, value in residual_kwargs.items()
                       if key not in ("source", "dt_scale")},
                )[0, :, 0].reshape(len(time_values), len(tangent_values)).detach().cpu()
            }
            boundary_times = time_values
            tangents = {"left": tangent_values}
        else:
            boundaries, boundary_times, tangents = build_boundary_coordinates(
                normalized_axes, options, device=device, dtype=dtype
            )
        if config.get("case") != "radiative_2d":
            for side, (coordinates, normal, tangent) in boundaries.items():
                if is_1d:
                    # The 1D boundary condition has no tangential coordinate; only
                    # interpolate the saved time trace at the diagnostic times.
                    time_query = coordinates[0, :, 0]
                    t0, t1, wt = _bracket(
                        normalized_axes[0].to(device=device, dtype=dtype), time_query
                    )
                    saved = _boundary_values(packed_condition, side)
                    rhs_normalized = saved[t0] * (1.0 - wt) + saved[t1] * wt
                else:
                    tangent_axis = normalized_axes[1] if side in ("left", "right") else normalized_axes[2]
                    rhs_normalized = interpolate_boundary_values(
                        _boundary_values(packed_condition, side), normalized_axes[0].to(device=device, dtype=dtype),
                        tangent_axis.to(device=device, dtype=dtype), coordinates[0, :, 0], tangent.reshape(-1),
                    )
                rhs = (rhs_normalized + 1.0) * 0.5 * (maximum - minimum) + minimum
                robin_residual = radiative_robin_residual(
                    physical_field, coordinates, normal, rhs.unsqueeze(-1),
                    **{key: value for key, value in residual_kwargs.items()
                       if key not in ("source", "dt_scale")},
                )
                boundary_residuals[side] = robin_residual[0, :, 0].reshape(
                    options.boundary_time_points,
                    1 if is_1d else options.boundary_spatial_points,
                ).detach().cpu()

    result["robin"] = {}
    for side, residual in boundary_residuals.items():
        entry = {
            "residual": residual,
            "time": _physical_axis(
                boundary_times, axes["t"], None if physical_bounds is None else physical_bounds[0]
            ).detach().cpu(),
        }
        if is_1d:
            entry["tangent"] = torch.as_tensor(
                axes["x"][0 if side == "left" else -1]
            ).reshape(1).cpu()
        else:
            tangent_axis = axes["y"] if side in ("left", "right") else axes["x"]
            bound = None
            if physical_bounds is not None:
                bound = physical_bounds[1] if side in ("left", "right") else physical_bounds[2]
            entry["tangent"] = _physical_axis(tangents[side], tangent_axis, bound).detach().cpu()
        result["robin"][side] = entry
    return result


@contextmanager
def capture_last_continuous_latent(model):
    """Capture the decoder latent from the final model call in one sampling run."""
    if not hasattr(model, "encode") or not hasattr(model, "decoder"):
        yield None
        return
    original_encode = model.encode
    had_instance_value = "encode" in model.__dict__
    previous_value = model.__dict__.get("encode")
    captured = {}

    def capture(_model, values, timestep):
        latent = original_encode(values, timestep)
        captured["latent"] = latent
        captured["timestep"] = timestep
        return latent

    model.encode = MethodType(capture, model)
    try:
        yield captured
    finally:
        if had_instance_value:
            model.encode = previous_value
        else:
            delattr(model, "encode")


@contextmanager
def frozen_model_parameters(model):
    """Keep coordinate derivatives without retaining parameter-gradient buffers."""
    parameters = list(model.parameters())
    requires_grad = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, enabled in zip(parameters, requires_grad):
            parameter.requires_grad_(enabled)
