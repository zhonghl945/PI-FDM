"""Differentiable grid physics losses for discrete baseline methods."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_float(value, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _to_physical(field: torch.Tensor, value_min, value_max) -> torch.Tensor:
    minimum = _as_float(value_min, field)
    maximum = _as_float(value_max, field)
    return (field + 1.0) * 0.5 * (maximum - minimum) + minimum


def _gradient(values: torch.Tensor, coordinates: torch.Tensor, dim: int) -> torch.Tensor:
    """First derivative on a possibly nonuniform axis, preserving autograd."""
    coordinates = coordinates.to(device=values.device, dtype=values.dtype)
    if coordinates.ndim != 1 or coordinates.numel() != values.shape[dim]:
        raise ValueError("derivative coordinates must match the differentiated axis")
    if coordinates.numel() < 2 or torch.any(coordinates[1:] <= coordinates[:-1]):
        raise ValueError("derivative coordinates must be strictly increasing")

    result = torch.empty_like(values)
    if coordinates.numel() == 2:
        first = [slice(None)] * values.ndim
        second = [slice(None)] * values.ndim
        first[dim], second[dim] = 0, 1
        slope = (
            values[tuple(second)] - values[tuple(first)]
        ) / (coordinates[1] - coordinates[0])
        result[tuple(first)] = slope
        result[tuple(second)] = slope
    else:
        middle = [slice(None)] * values.ndim
        before = [slice(None)] * values.ndim
        after = [slice(None)] * values.ndim
        middle[dim], before[dim], after[dim] = slice(1, -1), slice(0, -2), slice(2, None)
        h0 = coordinates[1:-1] - coordinates[:-2]
        h1 = coordinates[2:] - coordinates[1:-1]

        def broadcast(coefficients):
            return coefficients.reshape(
                *((1,) * dim), -1, *((1,) * (values.ndim - dim - 1))
            )

        result[tuple(middle)] = (
            broadcast(-h1 / (h0 * (h0 + h1))) * values[tuple(before)]
            + broadcast((h1 - h0) / (h0 * h1)) * values[tuple(middle)]
            + broadcast(h0 / (h1 * (h0 + h1))) * values[tuple(after)]
        )

        first = [slice(None)] * values.ndim
        second = [slice(None)] * values.ndim
        third = [slice(None)] * values.ndim
        first[dim], second[dim], third[dim] = 0, 1, 2
        left_h0, left_h1 = coordinates[1] - coordinates[0], coordinates[2] - coordinates[1]
        result[tuple(first)] = (
            -(2.0 * left_h0 + left_h1) / (left_h0 * (left_h0 + left_h1))
            * values[tuple(first)]
            + (left_h0 + left_h1) / (left_h0 * left_h1) * values[tuple(second)]
            - left_h0 / (left_h1 * (left_h0 + left_h1)) * values[tuple(third)]
        )

        before_last = [slice(None)] * values.ndim
        previous = [slice(None)] * values.ndim
        last = [slice(None)] * values.ndim
        before_last[dim], previous[dim], last[dim] = -3, -2, -1
        right_h0 = coordinates[-2] - coordinates[-3]
        right_h1 = coordinates[-1] - coordinates[-2]
        result[tuple(last)] = (
            right_h1 / (right_h0 * (right_h0 + right_h1)) * values[tuple(before_last)]
            - (right_h0 + right_h1) / (right_h0 * right_h1) * values[tuple(previous)]
            + (right_h0 + 2.0 * right_h1) / (right_h1 * (right_h0 + right_h1))
            * values[tuple(last)]
        )
    return result


def _radiative_diffusivity(
    energy: torch.Tensor,
    *gradients: torch.Tensor,
    material: torch.Tensor,
    use_flux_limiter: bool,
    limiter_norm: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    safe = energy.clamp_min(eps)
    material = material.to(device=energy.device, dtype=energy.dtype)
    if not use_flux_limiter:
        return safe.pow(0.75) / (3.0 * material.pow(3)).clamp_min(eps)
    if limiter_norm == "l2_smooth":
        norm_squared = torch.zeros_like(safe)
        for gradient in gradients:
            norm_squared = norm_squared + gradient.square()
        gradient_norm = torch.sqrt(norm_squared + eps)
    elif limiter_norm == "l1":
        gradient_norm = torch.zeros_like(safe)
        for gradient in gradients:
            gradient_norm = gradient_norm + gradient.abs()
    else:
        raise ValueError("limiter_norm must be 'l1' or 'l2_smooth'")
    denominator = 3.0 * material.pow(3) * safe.pow(0.25) + gradient_norm
    return safe / denominator.clamp_min(eps)


def _source_scalar(source, reference: torch.Tensor) -> torch.Tensor:
    if source is None:
        return reference.new_zeros(())
    if callable(source):
        raise ValueError("grid physics currently supports constant source terms only")
    return _as_float(source, reference)


def _control_volume_widths(axis: torch.Tensor) -> torch.Tensor:
    """Widths around packed cell centres with geometric boundaries at both ends."""
    if axis.numel() < 4:
        raise ValueError("grid physics needs at least two interior cell centres")
    centres = axis[1:-1]
    faces = torch.cat((axis[:1], 0.5 * (centres[:-1] + centres[1:]), axis[-1:]))
    widths = faces[1:] - faces[:-1]
    if torch.any(widths <= 0.0):
        raise ValueError("grid control-volume widths must be positive")
    return widths


def _harmonic_mean(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return 2.0 * left * right / (left + right).clamp_min(eps)


def _extrapolate_to_boundary(near_values: torch.Tensor, next_values: torch.Tensor,
                             near_coordinate: torch.Tensor,
                             next_coordinate: torch.Tensor,
                             boundary_coordinate: torch.Tensor) -> torch.Tensor:
    """Linearly extrapolate from two interior locations to a boundary."""
    near = near_coordinate.to(near_values)
    next_ = next_coordinate.to(near_values)
    boundary = boundary_coordinate.to(near_values)
    return near_values + (boundary - near) / (next_ - near) * (next_values - near_values)


class _GridLossBase:
    def __init__(self, axes: tuple[torch.Tensor, ...], *, value_min, value_max,
                 huber_delta: float, material: torch.Tensor | float):
        if len(axes) not in (2, 3):
            raise ValueError("grid axes must be (t,x) or (t,y,x)")
        self.axes = tuple(torch.as_tensor(axis).detach().float().cpu() for axis in axes)
        self.value_min = value_min
        self.value_max = value_max
        self.huber_delta = float(huber_delta)
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        self.material = (
            material.detach().float().cpu() if torch.is_tensor(material) else float(material)
        )

    def _field(self, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.ndim == len(self.axes) + 1:
            prediction = prediction.unsqueeze(1)
        if prediction.ndim != len(self.axes) + 2 or prediction.shape[1] != 1:
            raise ValueError(f"unexpected prediction shape {tuple(prediction.shape)}")
        expected = tuple(axis.numel() for axis in self.axes)
        if tuple(prediction.shape[2:]) != expected:
            raise ValueError(
                f"prediction grid {tuple(prediction.shape[2:])} does not match axes {expected}"
            )
        return _to_physical(prediction[:, 0], self.value_min, self.value_max)

    def _condition(self, model_input: torch.Tensor | None, field: torch.Tensor,
                   condition_field: torch.Tensor | None = None) -> torch.Tensor:
        source = model_input if condition_field is None else condition_field
        if source is None:
            raise ValueError("a clean condition field is required for grid physics")
        if source.ndim == len(self.axes) + 1:
            source = source.unsqueeze(1)
        if source.ndim != field.ndim + 1 or source.shape[1] < 1:
            raise ValueError(f"unexpected condition shape {tuple(source.shape)}")
        condition = _to_physical(
            source[:, 0].float(), self.value_min, self.value_max
        )
        if condition.shape != field.shape:
            raise ValueError("packed condition does not match prediction grid")
        return condition

    def _material_like(self, field: torch.Tensor) -> torch.Tensor:
        material = torch.as_tensor(self.material, device=field.device, dtype=field.dtype)
        if material.ndim == field.ndim + 1 and material.shape[1] == 1:
            material = material[:, 0]
        if material.ndim == 0:
            return torch.ones_like(field) * material
        while material.ndim < field.ndim:
            material = material.unsqueeze(0)
        try:
            return material.expand_as(field)
        except RuntimeError as exc:
            raise ValueError(
                f"material shape {tuple(material.shape)} cannot match field {tuple(field.shape)}"
            ) from exc

    def _reduce(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.numel() == 0:
            raise ValueError("grid residual has no valid points")
        return F.huber_loss(
            residual, torch.zeros_like(residual), delta=self.huber_delta, reduction="mean"
        )


class GridPhysicsLoss(_GridLossBase):
    """Finite-difference PDE residual over saved interior grid points."""

    def __init__(self, axes: tuple[torch.Tensor, ...], residual: str, *,
                 value_min: float = -1.0, value_max: float = 1.0,
                 huber_delta: float = 1.0, diffusivity: float = 1.0,
                 material: torch.Tensor | float = 1.0, source=None,
                 use_flux_limiter: bool = True, limiter_norm: str = "l2_smooth",
                 interior_only: bool = False):
        super().__init__(axes, value_min=value_min, value_max=value_max,
                         huber_delta=huber_delta, material=material)
        self.residual = residual
        self.diffusivity = float(diffusivity)
        self.source = source
        self.use_flux_limiter = bool(use_flux_limiter)
        self.limiter_norm = limiter_norm
        self.interior_only = bool(interior_only)

    def __call__(self, prediction=None, *, model_input=None,
                 condition_field=None, **_ignored) -> torch.Tensor:
        if prediction is None or model_input is None:
            raise ValueError("grid physics loss requires prediction and model_input")
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            field = self._field(prediction.float())
            condition = self._condition(model_input, field, condition_field)
            if self.residual == "linear_diffusion":
                residual = self._linear_residual(field, condition)
            elif self.residual == "radiative_diffusion":
                residual = (
                    self._radiative_interior_residual(field, condition)
                    if self.interior_only else self._radiative_residual(field, condition)
                )
            else:
                raise ValueError(f"unknown grid residual {self.residual!r}")
            return self._reduce(residual)

    def _radiative_interior_residual(self, field: torch.Tensor,
                                     condition: torch.Tensor) -> torch.Tensor:
        """Evaluate the PDE on cell-center interiors without ghost boundaries."""
        t_axis, y_axis, x_axis = (axis.to(field) for axis in self.axes)
        cells = torch.cat((condition[:, :1], field[:, 1:]), dim=1)
        if min(cells.shape[-2:]) < 3 or cells.shape[1] < 2:
            raise ValueError("interior radiative residual needs at least 3x3 cells and 2 times")
        material = self._material_like(cells)
        e_y = _gradient(cells, y_axis, dim=2)
        e_x = _gradient(cells, x_axis, dim=3)
        diffusion = _radiative_diffusivity(
            cells, e_y, e_x, material=material,
            use_flux_limiter=self.use_flux_limiter, limiter_norm=self.limiter_norm,
        )
        flux_y = -diffusion * e_y
        flux_x = -diffusion * e_x
        divergence = _gradient(flux_y, y_axis, dim=2) + _gradient(flux_x, x_axis, dim=3)
        dt = (t_axis[1:] - t_axis[:-1]).reshape(1, -1, 1, 1)
        e_t = (cells[:, 1:] - cells[:, :-1]) / dt
        source = _source_scalar(self.source, cells)
        return (e_t + divergence[:, 1:] - source)[:, :, 1:-1, 1:-1]

    def _linear_residual(self, field: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        t_axis, x_axis = (axis.to(field) for axis in self.axes)
        cells = field[:, :, 1:-1]
        cells = torch.cat((condition[:, :1, 1:-1], cells[:, 1:]), dim=1)
        centres = x_axis[1:-1]
        dt = (t_axis[1:] - t_axis[:-1]).reshape(1, -1, 1)
        u_t = (cells[:, 1:] - cells[:, :-1]) / dt

        internal_gradient = (
            cells[:, :, 1:] - cells[:, :, :-1]
        ) / (centres[1:] - centres[:-1]).reshape(1, 1, -1)
        left_gradient = (
            cells[:, :, :1] - condition[:, :, :1]
        ) / (centres[0] - x_axis[0])
        # The right packed slot is the prescribed outward Neumann derivative.
        right_gradient = condition[:, :, -1:]
        face_gradient = torch.cat((left_gradient, internal_gradient, right_gradient), dim=2)
        widths = _control_volume_widths(x_axis).reshape(1, 1, -1)
        u_xx = (face_gradient[:, :, 1:] - face_gradient[:, :, :-1]) / widths
        source = _source_scalar(self.source, field)
        return u_t - self.diffusivity * u_xx[:, 1:] - source

    def _radiative_residual(self, field: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        source = _source_scalar(self.source, field)
        material = self._material_like(field)
        if len(self.axes) == 2:
            t_axis, x_axis = (axis.to(field) for axis in self.axes)
            cells = field[:, :, 1:-1]
            cells = torch.cat((condition[:, :1, 1:-1], cells[:, 1:]), dim=1)
            cell_material = material[:, :, 1:-1]
            centres = x_axis[1:-1]
            dt = (t_axis[1:] - t_axis[:-1]).reshape(1, -1, 1)
            e_t = (cells[:, 1:] - cells[:, :-1]) / dt
            e_x = _gradient(cells, centres, dim=2)
            cell_diffusion = _radiative_diffusivity(
                cells, e_x, material=cell_material,
                use_flux_limiter=self.use_flux_limiter, limiter_norm=self.limiter_norm,
            )
            internal_diffusion = _harmonic_mean(
                cell_diffusion[:, :, :-1], cell_diffusion[:, :, 1:]
            )
            internal_flux = -internal_diffusion * (
                cells[:, :, 1:] - cells[:, :, :-1]
            ) / (centres[1:] - centres[:-1]).reshape(1, 1, -1)
            left_normal = (
                field[:, :, 0] - cells[:, :, 0]
            ) / (centres[0] - x_axis[0])
            right_normal = (
                field[:, :, -1] - cells[:, :, -1]
            ) / (x_axis[-1] - centres[-1])
            left_diffusion = _radiative_diffusivity(
                field[:, :, 0], left_normal, material=cell_material[:, :, 0],
                use_flux_limiter=self.use_flux_limiter, limiter_norm=self.limiter_norm,
            )
            right_diffusion = _radiative_diffusivity(
                field[:, :, -1], right_normal, material=cell_material[:, :, -1],
                use_flux_limiter=self.use_flux_limiter, limiter_norm=self.limiter_norm,
            )
            flux = torch.cat((
                (left_diffusion * left_normal).unsqueeze(-1), internal_flux,
                (-right_diffusion * right_normal).unsqueeze(-1),
            ), dim=2)
            widths = _control_volume_widths(x_axis).reshape(1, 1, -1)
            divergence = (flux[:, :, 1:] - flux[:, :, :-1]) / widths
            return e_t + divergence[:, 1:] - source

        t_axis, y_axis, x_axis = (axis.to(field) for axis in self.axes)
        cells = field[:, :, 1:-1, 1:-1]
        cells = torch.cat((condition[:, :1, 1:-1, 1:-1], cells[:, 1:]), dim=1)
        cell_material = material[:, :, 1:-1, 1:-1]
        y_centres, x_centres = y_axis[1:-1], x_axis[1:-1]
        dt = (t_axis[1:] - t_axis[:-1]).reshape(1, -1, 1, 1)
        e_t = (cells[:, 1:] - cells[:, :-1]) / dt
        e_y = _gradient(cells, y_centres, dim=2)
        e_x = _gradient(cells, x_centres, dim=3)
        cell_diffusion = _radiative_diffusivity(
            cells, e_y, e_x, material=cell_material,
            use_flux_limiter=self.use_flux_limiter, limiter_norm=self.limiter_norm,
        )
        x_diffusion = _harmonic_mean(
            cell_diffusion[:, :, :, :-1], cell_diffusion[:, :, :, 1:]
        )
        x_flux = -x_diffusion * (
            cells[:, :, :, 1:] - cells[:, :, :, :-1]
        ) / (x_centres[1:] - x_centres[:-1]).reshape(1, 1, 1, -1)
        y_diffusion = _harmonic_mean(
            cell_diffusion[:, :, :-1, :], cell_diffusion[:, :, 1:, :]
        )
        y_flux = -y_diffusion * (
            cells[:, :, 1:, :] - cells[:, :, :-1, :]
        ) / (y_centres[1:] - y_centres[:-1]).reshape(1, 1, -1, 1)

        left_normal = (
            field[:, :, 1:-1, 0] - cells[:, :, :, 0]
        ) / (x_centres[0] - x_axis[0])
        right_normal = (
            field[:, :, 1:-1, -1] - cells[:, :, :, -1]
        ) / (x_axis[-1] - x_centres[-1])
        bottom_normal = (
            field[:, :, 0, 1:-1] - cells[:, :, 0, :]
        ) / (y_centres[0] - y_axis[0])
        top_normal = (
            field[:, :, -1, 1:-1] - cells[:, :, -1, :]
        ) / (y_axis[-1] - y_centres[-1])
        left_tangent = _extrapolate_to_boundary(
            e_y[:, :, :, 0], e_y[:, :, :, 1],
            x_centres[0], x_centres[1], x_axis[0],
        )
        right_tangent = _extrapolate_to_boundary(
            e_y[:, :, :, -1], e_y[:, :, :, -2],
            x_centres[-1], x_centres[-2], x_axis[-1],
        )
        bottom_tangent = _extrapolate_to_boundary(
            e_x[:, :, 0, :], e_x[:, :, 1, :],
            y_centres[0], y_centres[1], y_axis[0],
        )
        top_tangent = _extrapolate_to_boundary(
            e_x[:, :, -1, :], e_x[:, :, -2, :],
            y_centres[-1], y_centres[-2], y_axis[-1],
        )

        def boundary_diffusion(face, normal, tangent, boundary_material):
            return _radiative_diffusivity(
                face, normal, tangent, material=boundary_material,
                use_flux_limiter=self.use_flux_limiter,
                limiter_norm=self.limiter_norm,
            )

        left_diffusion = boundary_diffusion(
            field[:, :, 1:-1, 0], left_normal, left_tangent, cell_material[:, :, :, 0]
        )
        right_diffusion = boundary_diffusion(
            field[:, :, 1:-1, -1], right_normal, right_tangent, cell_material[:, :, :, -1]
        )
        bottom_diffusion = boundary_diffusion(
            field[:, :, 0, 1:-1], bottom_normal, bottom_tangent, cell_material[:, :, 0, :]
        )
        top_diffusion = boundary_diffusion(
            field[:, :, -1, 1:-1], top_normal, top_tangent, cell_material[:, :, -1, :]
        )
        x_flux = torch.cat((
            (left_diffusion * left_normal).unsqueeze(-1), x_flux,
            (-right_diffusion * right_normal).unsqueeze(-1),
        ), dim=3)
        y_flux = torch.cat((
            (bottom_diffusion * bottom_normal).unsqueeze(2), y_flux,
            (-top_diffusion * top_normal).unsqueeze(2),
        ), dim=2)
        dx = _control_volume_widths(x_axis).reshape(1, 1, 1, -1)
        dy = _control_volume_widths(y_axis).reshape(1, 1, -1, 1)
        divergence = (
            (x_flux[:, :, :, 1:] - x_flux[:, :, :, :-1]) / dx
            + (y_flux[:, :, 1:, :] - y_flux[:, :, :-1, :]) / dy
        )
        return e_t + divergence[:, 1:] - source


class GridRobinBoundaryLoss(_GridLossBase):
    """Robin residual matching the finite-volume face/ghost closure."""

    def __init__(self, axes: tuple[torch.Tensor, ...], *,
                 value_min: float = -1.0, value_max: float = 1.0,
                 huber_delta: float = 1.0, material: torch.Tensor | float = 1.0,
                 use_flux_limiter: bool = True, limiter_norm: str = "l2_smooth",
                 left_only: bool = False,
                 domain_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None):
        super().__init__(axes, value_min=value_min, value_max=value_max,
                         huber_delta=huber_delta, material=material)
        self.use_flux_limiter = bool(use_flux_limiter)
        self.limiter_norm = limiter_norm
        self.left_only = bool(left_only)
        self.domain_bounds = domain_bounds
        if self.left_only:
            if len(axes) != 3:
                raise ValueError("left-only grid Robin loss requires (t,y,x) axes")
            if domain_bounds is None or len(domain_bounds) != 2:
                raise ValueError(
                    "left-only grid Robin loss requires (y_bounds, x_bounds)"
                )
            if any(len(bounds) != 2 or not float(bounds[1]) > float(bounds[0])
                   for bounds in domain_bounds):
                raise ValueError("grid Robin domain bounds must be increasing pairs")

    def __call__(self, prediction=None, *, model_input=None, g_left=None, **_ignored) -> torch.Tensor:
        if prediction is None or model_input is None:
            raise ValueError("grid Robin loss requires prediction and model_input")
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            field = self._field(prediction.float())
            if self.left_only:
                residuals = (self._robin_2d_left(field, g_left),)
            else:
                rhs = self._condition(model_input, field)
                residuals = (
                    self._robin_1d(field, rhs)
                    if len(self.axes) == 2 else self._robin_2d(field, rhs)
                )
            return self._reduce(torch.cat([
                residual.flatten(1) for residual in residuals
            ], dim=1))

    def _robin_2d_left(self, field: torch.Tensor, g_left: torch.Tensor) -> torch.Tensor:
        """Evaluate only the saved left ``gL(t,y)`` nodes on a cell-centred grid.

        ``radiative_2d`` stores values at cell centres and deliberately does
        not store an artificial boundary E slot.  Reconstruct the geometric
        left face from the first two centres, then apply the same
        ``0.5 E + D grad(E).n = g`` operator used by the continuous loss.
        The initial slice is omitted because the generated initial field is an
        independent known condition and is not required to satisfy the later
        inflow closure at exactly t=0.
        """
        if g_left is None:
            raise ValueError("left-only grid Robin loss requires g_left")
        if g_left.ndim == 4 and g_left.shape[1] == 1:
            g_left = g_left[:, 0]
        expected = (field.shape[0], field.shape[1], field.shape[2])
        if g_left.ndim != 3 or tuple(g_left.shape) != expected:
            raise ValueError(f"g_left must have shape {expected}, got {tuple(g_left.shape)}")
        if field.shape[1] < 2 or field.shape[-1] < 2 or field.shape[-2] < 2:
            raise ValueError("left-only grid Robin loss needs two time and spatial points")

        y_axis, x_axis = (axis.to(field) for axis in self.axes[1:])
        y_boundary, _ = self.domain_bounds[0]
        x_boundary, _ = self.domain_bounds[1]
        x0, x1 = x_axis[0], x_axis[1]
        if not x0 > x_boundary or not x1 > x0:
            raise ValueError("left boundary must lie before strictly increasing cell centres")

        values = field[:, 1:]
        first_cell = values[..., 0]
        second_cell = values[..., 1]
        face = _extrapolate_to_boundary(first_cell, second_cell, x0, x1, x_axis.new_tensor(x_boundary))
        normal = (face - first_cell) / (x0 - x_axis.new_tensor(x_boundary))
        tangent = _gradient(first_cell, y_axis, dim=2)
        material = self._material_like(field)[:, 1:, :, 0]
        diffusion = _radiative_diffusivity(
            face, normal, tangent, material=material,
            use_flux_limiter=self.use_flux_limiter,
            limiter_norm=self.limiter_norm,
        )
        rhs = _to_physical(g_left[:, 1:], self.value_min, self.value_max)
        return 0.5 * face + diffusion * normal - rhs

    def _robin_residual(self, face, cell, rhs, distance, material, tangent=None):
        # The solver uses normal = 2 * (face - cell) / cell_width.  Since the
        # first saved coordinate is the cell center, distance=cell_width/2.
        normal = (face - cell) / distance.to(face)
        gradients = (normal,) if tangent is None else (normal, tangent)
        diffusion = _radiative_diffusivity(
            face, *gradients, material=material,
            use_flux_limiter=self.use_flux_limiter, limiter_norm=self.limiter_norm,
        )
        return 0.5 * face + diffusion * normal - rhs

    def _robin_1d(self, field, rhs):
        if field.shape[-1] < 4:
            raise ValueError("Robin residual needs at least two interior points")
        x = self.axes[1].to(field)
        material = self._material_like(field)[:, 1:]
        left = self._robin_residual(
            field[:, 1:, 0], field[:, 1:, 1], rhs[:, 1:, 0],
            x[1] - x[0], material[:, :, 1],
        )
        right = self._robin_residual(
            field[:, 1:, -1], field[:, 1:, -2], rhs[:, 1:, -1],
            x[-1] - x[-2], material[:, :, -2],
        )
        return left, right

    def _robin_2d(self, field, rhs):
        if min(field.shape[-2:]) < 4:
            raise ValueError("Robin residual needs at least two interior points")
        y, x = self.axes[1].to(field), self.axes[2].to(field)
        material = self._material_like(field)[:, 1:]
        cell_field = field[:, 1:, 1:-1, 1:-1]
        cell_y = _gradient(cell_field, y[1:-1], dim=2)
        cell_x = _gradient(cell_field, x[1:-1], dim=3)
        left_tangent = _extrapolate_to_boundary(
            cell_y[:, :, :, 0], cell_y[:, :, :, 1],
            x[1], x[2], x[0],
        )
        right_tangent = _extrapolate_to_boundary(
            cell_y[:, :, :, -1], cell_y[:, :, :, -2],
            x[-2], x[-3], x[-1],
        )
        bottom_tangent = _extrapolate_to_boundary(
            cell_x[:, :, 0, :], cell_x[:, :, 1, :],
            y[1], y[2], y[0],
        )
        top_tangent = _extrapolate_to_boundary(
            cell_x[:, :, -1, :], cell_x[:, :, -2, :],
            y[-2], y[-3], y[-1],
        )

        left_face, right_face = field[:, 1:, 1:-1, 0], field[:, 1:, 1:-1, -1]
        left = self._robin_residual(
            left_face, field[:, 1:, 1:-1, 1], rhs[:, 1:, 1:-1, 0], x[1] - x[0],
            material[:, :, 1:-1, 1], left_tangent,
        )
        right = self._robin_residual(
            right_face, field[:, 1:, 1:-1, -2], rhs[:, 1:, 1:-1, -1], x[-1] - x[-2],
            material[:, :, 1:-1, -2], right_tangent,
        )

        bottom_face, top_face = field[:, 1:, 0, 1:-1], field[:, 1:, -1, 1:-1]
        bottom = self._robin_residual(
            bottom_face, field[:, 1:, 1, 1:-1], rhs[:, 1:, 0, 1:-1], y[1] - y[0],
            material[:, :, 1, 1:-1], bottom_tangent,
        )
        top = self._robin_residual(
            top_face, field[:, 1:, -2, 1:-1], rhs[:, 1:, -1, 1:-1], y[-1] - y[-2],
            material[:, :, -2, 1:-1], top_tangent,
        )
        return left, right, bottom, top
