"""Build case physics from serializable experiment configuration."""

from __future__ import annotations

import torch

from pifdm.physics import (CollocationPhysicsLoss, CoordinateSystem, GridPhysicsLoss,
                           GridRobinBoundaryLoss, RobinBoundaryLoss)
from pifdm.physics.losses import FusedRadiativePhysicsLoss
from pifdm.experiments.registry import get_case


def _coordinate_system(config: dict, axes: dict[str, torch.Tensor], axis_names):
    """Use geometric bounds for radiative_2d cell-center coordinates only."""
    bounds = None
    if config.get("case") == "radiative_2d":
        bounds = tuple(
            tuple(float(value) for value in config["domain"][name])
            for name in axis_names
        )
    return CoordinateSystem(tuple(axes[name] for name in axis_names), bounds)


def _material_function(spec: dict, coordinates: CoordinateSystem):
    kind = spec.get("type", "constant")
    if kind == "constant":
        return float(spec.get("value", spec.get("background", 1.0)))
    if kind != "rectangles":
        raise ValueError(f"unknown material type {kind!r}")

    def material(normalized_coords):
        physical = coordinates.to_physical(normalized_coords)
        result = torch.full_like(physical[..., :1], float(spec.get("background", 1.0)))
        for region in spec.get("regions", []):
            inside = torch.ones_like(result, dtype=torch.bool)
            axis_names = ("t", "y", "x") if physical.shape[-1] == 3 else ("t", "x")
            for axis, bounds in region.items():
                if axis == "value":
                    continue
                index = axis_names.index(axis)
                inside &= (physical[..., index:index + 1] >= bounds[0]) & (physical[..., index:index + 1] <= bounds[1])
            result = torch.where(inside, float(region["value"]), result)
        return result

    return material


def _material_value_bounds(spec: dict) -> tuple[float, float]:
    """Return stable bounds for the normalized material conditioning feature."""
    kind = spec.get("type", "constant")
    if kind == "constant":
        value = float(spec.get("value", spec.get("background", 1.0)))
        return value, value
    if kind != "rectangles":
        raise ValueError(f"unknown material type {kind!r}")
    values = [float(spec.get("background", 1.0))]
    values.extend(float(region["value"]) for region in spec.get("regions", []))
    return min(values), max(values)


def build_material_inputs(config: dict, axes: dict[str, torch.Tensor]):
    """Build the clean grid condition and coordinate feature for material ``z``.

    The returned grid has shape ``(1, 1, *solution_shape)`` and is kept outside
    the diffusion noising path.  The callable uses the same physical material
    definition at arbitrary decoder coordinates, including off-grid queries.
    """
    if not bool(config.get("material_conditioning", False)):
        return None, None
    case = get_case(config["case"])
    axis_names = ("t", "y", "x") if case.dimension == 2 else ("t", "x")
    coordinates = _coordinate_system(config, axes, axis_names)
    physical_material = _material_function(
        config.get("physics", {}).get("material", {}), coordinates
    )
    minimum, maximum = _material_value_bounds(
        config.get("physics", {}).get("material", {})
    )
    scale = maximum - minimum

    def normalized_material(query: torch.Tensor) -> torch.Tensor:
        value = physical_material(query)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=query.dtype, device=query.device)
        else:
            value = value.to(device=query.device, dtype=query.dtype)
        if value.ndim == 0:
            value = torch.ones(
                (*query.shape[:-1], 1), device=query.device, dtype=query.dtype
            ) * value
        elif value.ndim == query.ndim - 1:
            value = value.unsqueeze(-1)
        if value.shape != (*query.shape[:-1], 1):
            raise ValueError(
                f"material feature has shape {tuple(value.shape)}, expected "
                f"{(*query.shape[:-1], 1)}"
            )
        if scale <= 1e-12:
            return torch.zeros_like(value)
        return 2.0 * (value - minimum) / scale - 1.0

    material_grid = normalized_material(coordinates.mesh())
    return material_grid.squeeze(-1).unsqueeze(0).unsqueeze(0).contiguous(), normalized_material


def build_physics_loss(config: dict, axes: dict[str, torch.Tensor], scalers: dict):
    """Return ``None`` for non-physics runs, otherwise a collocation loss."""
    controls = config.get("physics_training")
    if controls is None:
        return None
    physics = config.get("physics", {})
    case = config["case"]
    axis_names = ("t", "y", "x") if get_case(case).dimension == 2 else ("t", "x")
    coordinates = _coordinate_system(config, axes, axis_names)
    scales = coordinates.derivative_scales
    point_sampling = controls.get("point_sampling", "random")
    collocation_points = None if point_sampling == "grid" else controls.get("collocation_points")
    equation = {}
    residual = "linear_diffusion" if case == "linear_1d" else "radiative_diffusion"
    if residual == "linear_diffusion":
        equation["diffusivity"] = float(physics.get("diffusivity", 1.0))
        source = physics.get("source", {"type": "constant", "value": 0.0})
        if source.get("type", "constant") != "constant":
            raise ValueError("non-constant sources need a case-specific callable")
        equation["source"] = float(source.get("value", 0.0))
    else:
        equation["material"] = _material_function(physics.get("material", {}), coordinates)
        equation["use_flux_limiter"] = physics.get("use_flux_limiter", True)
        equation["limiter_norm"] = physics.get("limiter_norm", "l2_smooth")
        source = physics.get("source", {"type": "constant", "value": 0.0})
        if source.get("type", "constant") != "constant":
            raise ValueError("non-constant sources need a case-specific callable")
        equation["source"] = float(source.get("value", 0.0))
    return CollocationPhysicsLoss(
        coordinates.mesh().reshape(-1, len(axis_names)), residual,
        samples=collocation_points, point_sampling=point_sampling, dt_scale=scales[0],
        dy_scale=scales[1] if len(scales) == 3 else None, dx_scale=scales[-1],
        value_min=scalers["min"], value_max=scalers["max"], **equation,
        huber_delta=float(controls.get("huber_delta", 1.0)),
        exclude_grid_spatial_edges=config.get("case") == "radiative_2d",
    )


def build_boundary_loss(config: dict, axes: dict[str, torch.Tensor], scalers: dict):
    """Build a radiative Robin loss while keeping packed ``g`` as conditioning."""
    controls = config.get("physics_training")
    case = get_case(config["case"])
    if controls is None or case.boundary_slots != "operator":
        return None
    axis_names = ("t", "y", "x") if case.dimension == 2 else ("t", "x")
    coordinates = _coordinate_system(config, axes, axis_names)
    scales = coordinates.derivative_scales
    physics = config.get("physics", {})
    # PDE and Robin terms may intentionally use different point sets.  In
    # particular, PI-DeepONet uses a deterministic interior grid subsample
    # but evaluates Robin on every saved post-initial boundary node.
    point_sampling = controls.get(
        "boundary_point_sampling", controls.get("point_sampling", "random")
    )
    boundary_points = None if point_sampling == "grid" else controls.get(
        "boundary_points", controls.get("collocation_points", 1024)
    )
    return RobinBoundaryLoss(
        coordinates.normalized_axes,
        samples=boundary_points, point_sampling=point_sampling,
        dx_scale=scales[-1], dy_scale=scales[1] if len(scales) == 3 else None,
        value_min=scalers["min"], value_max=scalers["max"],
        material=_material_function(physics.get("material", {}), coordinates),
        use_flux_limiter=physics.get("use_flux_limiter", True),
        limiter_norm=physics.get("limiter_norm", "l2_smooth"),
        left_only=config.get("case") == "radiative_2d",
    )


def build_fused_physics_losses(config: dict, axes: dict[str, torch.Tensor], scalers: dict):
    """Build shared-query PDE and Robin losses for an explicitly opted-in case."""
    physics_loss = build_physics_loss(config, axes, scalers)
    boundary_loss = build_boundary_loss(config, axes, scalers)
    if physics_loss is None or boundary_loss is None:
        raise ValueError("fused physics queries require both PDE and Robin losses")
    fused = FusedRadiativePhysicsLoss(physics_loss, boundary_loss)
    return fused.physics, fused.boundary


def _grid_material(config: dict, axes: dict[str, torch.Tensor], axis_names):
    coordinates = _coordinate_system(config, axes, axis_names)
    material = _material_function(config.get("physics", {}).get("material", {}), coordinates)
    if callable(material):
        material = material(coordinates.mesh()).squeeze(-1)
    return material


def build_grid_physics_loss(config: dict, axes: dict[str, torch.Tensor], scalers: dict):
    """Build a saved-grid residual without altering collocation-loss behavior."""
    controls = config.get("physics_training")
    if controls is None:
        return None
    case = get_case(config["case"])
    axis_names = ("t", "y", "x") if case.dimension == 2 else ("t", "x")
    physics = config.get("physics", {})
    source = physics.get("source", {"type": "constant", "value": 0.0})
    if source.get("type", "constant") != "constant":
        raise ValueError("grid physics currently supports constant sources only")
    options = {
        "value_min": scalers["min"], "value_max": scalers["max"],
        "huber_delta": float(controls.get("huber_delta", 1.0)),
        "source": float(source.get("value", 0.0)),
    }
    if case.residual == "linear_diffusion":
        options["diffusivity"] = float(physics.get("diffusivity", 1.0))
    else:
        options.update({
            "material": _grid_material(config, axes, axis_names),
            "use_flux_limiter": physics.get("use_flux_limiter", True),
            "limiter_norm": physics.get("limiter_norm", "l2_smooth"),
        })
    return GridPhysicsLoss(
        tuple(axes[name] for name in axis_names), case.residual,
        interior_only=config.get("case") == "radiative_2d", **options
    )


def build_grid_boundary_loss(config: dict, axes: dict[str, torch.Tensor], scalers: dict):
    """Build the discrete Robin term for PINO/DiffusionPDE/DiffConPhy."""
    controls = config.get("physics_training")
    case = get_case(config["case"])
    if controls is None or case.boundary_slots != "operator":
        return None
    axis_names = ("t", "y", "x") if case.dimension == 2 else ("t", "x")
    physics = config.get("physics", {})
    radiative_2d_left_only = config.get("case") == "radiative_2d"
    return GridRobinBoundaryLoss(
        tuple(axes[name] for name in axis_names),
        value_min=scalers["min"], value_max=scalers["max"],
        huber_delta=float(controls.get("huber_delta", 1.0)),
        material=_grid_material(config, axes, axis_names),
        use_flux_limiter=physics.get("use_flux_limiter", True),
        limiter_norm=physics.get("limiter_norm", "l2_smooth"),
        left_only=radiative_2d_left_only,
        domain_bounds=(
            tuple(float(value) for value in config["domain"]["y"]),
            tuple(float(value) for value in config["domain"]["x"]),
        ) if radiative_2d_left_only else None,
    )
