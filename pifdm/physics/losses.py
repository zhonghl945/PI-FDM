"""Physics-loss plumbing kept separate from the diffusion objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .residuals import (
    _first_derivative,
    _material_at,
    _radiative_diffusivity,
    linear_diffusion_residual,
    radiative_diffusion_residual,
    radiative_robin_residual,
)
class CollocationPhysicsLoss:
    """Evaluate a differentiable PDE residual on random or saved-grid points.

    Initial values are supervised by reconstruction. Boundary operators are
    handled by a separate loss, so this class evaluates only open-domain PDE
    points.
    """

    def __init__(self, normalized_grid: torch.Tensor, residual: str, *, samples: int | None = None,
                 point_sampling: str | None = None,
                 dt_scale: float, dx_scale: float, dy_scale: float | None = None,
                 value_min: float = -1.0, value_max: float = 1.0,
                 huber_delta: float = 1.0,
                 exclude_grid_spatial_edges: bool = False,
                 **equation):
        if normalized_grid.ndim != 2:
            raise ValueError("normalized_grid must be (N, coordinate_dim)")
        self.grid = normalized_grid
        self.residual = residual
        if point_sampling is None:
            point_sampling = "grid" if samples is None else "random"
        if point_sampling not in ("random", "grid", "grid_subsample"):
            raise ValueError("point_sampling must be 'random', 'grid', or 'grid_subsample'")
        self.point_sampling = point_sampling
        self.samples = None
        if point_sampling in ("random", "grid_subsample"):
            if samples is None or int(samples) <= 0:
                raise ValueError("collocation samples must be positive in sampled-grid mode")
            self.samples = int(samples)
        self.value_min = value_min
        self.value_max = value_max
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        self.huber_delta = float(huber_delta)
        self.exclude_grid_spatial_edges = bool(exclude_grid_spatial_edges)
        self.kwargs = dict(dt_scale=dt_scale, dx_scale=dx_scale, **equation)
        if dy_scale is not None:
            self.kwargs["dy_scale"] = dy_scale
        self._grid_interior = None
        if point_sampling == "grid_subsample":
            interior = self._saved_interior_grid()
            count = min(self.samples, interior.shape[0])
            indices = torch.linspace(0, interior.shape[0] - 1, count).long()
            self._grid_interior = interior[indices].contiguous()

    def _saved_interior_grid(self) -> torch.Tensor:
        interior = self.grid[self.grid[:, 0] > -1]
        for axis in range(1, self.grid.shape[-1]):
            lower, upper = self.grid[:, axis].amin(), self.grid[:, axis].amax()
            if self.exclude_grid_spatial_edges:
                interior = interior[(interior[:, axis] > lower) & (interior[:, axis] < upper)]
            else:
                interior = interior[(interior[:, axis] > -1) & (interior[:, axis] < 1)]
        return interior

    def sample_coordinates(self, batch_size: int, device: torch.device,
                           dtype: torch.dtype) -> torch.Tensor:
        """Sample the exact collocation set used by ``__call__``."""
        if self.point_sampling == "random":
            margin = 1e-4
            coords = torch.empty(
                batch_size, int(self.samples), self.grid.shape[-1],
                device=device, dtype=dtype,
            )
            coords[..., 0].uniform_(-1.0 + margin, 1.0 - margin)
            for axis in range(1, coords.shape[-1]):
                coords[..., axis].uniform_(-1.0 + margin, 1.0 - margin)
        elif self.point_sampling == "grid_subsample":
            coords = self._grid_interior.to(device).unsqueeze(0).expand(
                batch_size, -1, -1
            ).clone()
        else:
            interior = self._saved_interior_grid().to(device)
            coords = interior.unsqueeze(0).expand(batch_size, -1, -1).clone()
        return coords.requires_grad_(True)

    def __call__(self, prediction=None, *, field=None, model=None, model_input=None,
                 timestep=None, latent_grid=None, **_ignored) -> torch.Tensor:
        """Evaluate a field callable or construct one from a continuous model.

        The keyword form matches ``GaussianDiffusion.physics_loss_fn`` so a
        ``CollocationPhysicsLoss`` instance can be passed directly to it.
        """
        # The surrounding training forward may use CUDA AMP.  Re-enter this
        # method with autocast disabled so first/second coordinate derivatives
        # and the nonlinear PDE coefficients are evaluated in float32.  The
        # cast keeps the graph connected to the mixed-precision latent.
        if torch.is_autocast_enabled():
            if model_input is None:
                raise ValueError("model_input is required for mixed-precision physics loss")
            with torch.autocast(device_type=model_input.device.type, enabled=False):
                return self(
                    prediction=prediction.float() if prediction is not None else None,
                    field=field, model=model,
                    model_input=model_input.float(), timestep=timestep,
                    latent_grid=latent_grid.float() if latent_grid is not None else None,
                    **_ignored,
                )
        if field is None:
            if model is None or model_input is None:
                raise ValueError("provide field=... or model/model_input")
            if not hasattr(model, "encode") or not hasattr(model, "decoder"):
                if not hasattr(model, "branch") or not hasattr(model, "trunk"):
                    raise TypeError("physics loss requires a continuous decoder or DeepONet trunk")
                field = lambda query: model(
                    model_input, timestep, coords=query,
                    g_left=_ignored.get("g_left"),
                )
            else:
                if latent_grid is None:
                    latent_grid = model.encode(model_input, timestep)
                field = lambda query: model.decoder(latent_grid, query.expand(latent_grid.shape[0], -1, -1))
        normalized_field = field
        value_min = torch.as_tensor(self.value_min)
        value_max = torch.as_tensor(self.value_max)
        field = lambda query: (normalized_field(query) + 1.0) * 0.5 * (value_max.to(query) - value_min.to(query)) + value_min.to(query)
        device = model_input.device if model_input is not None else self.grid.device
        batch_size = model_input.shape[0] if model_input is not None else (
            prediction.shape[0] if prediction is not None else 1
        )
        coords = self.sample_coordinates(batch_size, device, self.grid.dtype)
        if self.residual == "linear_diffusion":
            residual = linear_diffusion_residual(field, coords, **self.kwargs)
        elif self.residual == "radiative_diffusion":
            residual = radiative_diffusion_residual(field, coords, **self.kwargs)
        else:
            raise ValueError(f"unknown residual {self.residual!r}")
        point_loss = F.huber_loss(
            residual, torch.zeros_like(residual), delta=self.huber_delta, reduction="none",
        )
        point_loss = point_loss.reshape(batch_size, coords.shape[1], -1).mean(dim=-1)
        return point_loss.mean()


class RobinBoundaryLoss:
    """Enforce the radiative operator on random or saved-grid boundary points."""

    def __init__(self, normalized_axes: tuple[torch.Tensor, ...], *, samples: int | None = None,
                 point_sampling: str = "random",
                 dx_scale: float, dy_scale: float | None = None,
                 value_min: float = -1.0, value_max: float = 1.0,
                 material: torch.Tensor | float = 1.0,
                 use_flux_limiter: bool = True,
                 limiter_norm: str = "l2_smooth", left_only: bool = False):
        if len(normalized_axes) not in (2, 3):
            raise ValueError("Robin boundary axes must be (t,x) or (t,y,x)")
        if any(axis.ndim != 1 or len(axis) < 2 for axis in normalized_axes):
            raise ValueError("each Robin boundary axis must contain at least two points")
        if len(normalized_axes) == 3 and any(len(axis) < 3 for axis in normalized_axes[1:]):
            raise ValueError("2D Robin boundary sampling needs interior tangential nodes")
        self.axes = tuple(axis.detach().cpu() for axis in normalized_axes)
        if point_sampling not in ("random", "grid", "grid_subsample"):
            raise ValueError("point_sampling must be 'random', 'grid', or 'grid_subsample'")
        self.point_sampling = point_sampling
        self.samples = None if samples is None else int(samples)
        if self.point_sampling == "random":
            if self.samples is None or self.samples <= 0:
                raise ValueError("Robin boundary samples must be positive in random mode")
        self.value_min = value_min
        self.value_max = value_max
        self.left_only = bool(left_only)
        self.kwargs = {
            "dx_scale": dx_scale,
            "material": material,
            "use_flux_limiter": use_flux_limiter,
            "limiter_norm": limiter_norm,
        }
        if dy_scale is not None:
            self.kwargs["dy_scale"] = dy_scale

    def _sample_left_saved(self, g_left: torch.Tensor, batch_size: int,
                           device: torch.device, dtype: torch.dtype):
        """Use the stored (t,y) nodes directly, with x fixed at the face."""
        if len(self.axes) != 3:
            raise ValueError("left-only Robin sampling is defined for 2D radiative_2d data")
        if g_left.ndim == 4 and g_left.shape[1] == 1:
            g_left = g_left[:, 0]
        expected = (batch_size, len(self.axes[0]), len(self.axes[1]))
        if g_left.ndim != 3 or tuple(g_left.shape) != expected:
            raise ValueError(f"g_left must have shape {expected}, got {tuple(g_left.shape)}")
        time_axis, tangent_axis = (axis.to(device=device, dtype=dtype) for axis in self.axes[:2])
        # The generated initial field is an independent known state.  The
        # solver applies the inflow closure starting at the first positive
        # time step, so do not make those two conditions compete at t=0.
        if len(time_axis) < 2:
            raise ValueError("left-only Robin sampling needs at least two saved times")
        time_axis = time_axis[1:]
        g_left = g_left[:, 1:]
        time_grid, tangent_grid = torch.meshgrid(time_axis, tangent_axis, indexing="ij")
        coordinates = torch.stack((
            time_grid, tangent_grid, torch.full_like(time_grid, -1.0),
        ), dim=-1).reshape(1, -1, 3).expand(batch_size, -1, -1).clone()
        normal = torch.zeros(batch_size, coordinates.shape[1], 2, device=device, dtype=dtype)
        normal[..., 1] = -1.0
        rhs = g_left.to(device=device, dtype=dtype).reshape(batch_size, -1, 1)
        minimum = torch.as_tensor(self.value_min, device=device, dtype=dtype)
        maximum = torch.as_tensor(self.value_max, device=device, dtype=dtype)
        physical_rhs = (rhs + 1.0) * 0.5 * (maximum - minimum) + minimum
        return coordinates.requires_grad_(True), normal, physical_rhs

    def _sample(self, packed: torch.Tensor, g_left: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = packed.shape[0]
        device, dtype = packed.device, packed.dtype
        axes = tuple(axis.to(device=device, dtype=dtype) for axis in self.axes)
        if self.left_only:
            return self._sample_left_saved(g_left, batch_size, device, dtype)
        if self.point_sampling == "grid":
            return self._grid_sample(packed, axes, batch_size, device, dtype)
        count = self.samples
        batch = torch.arange(batch_size, device=device)[:, None]
        time_index = torch.randint(1, len(axes[0]), (batch_size, count), device=device)
        side = torch.randint(0, 2 if len(axes) == 2 else 4,
                             (batch_size, count), device=device)
        coordinates = torch.empty(batch_size, count, len(axes), device=device, dtype=dtype)
        coordinates[..., 0] = axes[0][time_index]

        if len(axes) == 2:
            space_index = torch.where(side == 0, 0, len(axes[1]) - 1)
            coordinates[..., 1] = axes[1][space_index]
            normal = torch.where(side == 0, -1.0, 1.0).to(dtype).unsqueeze(-1)
            rhs = packed[batch, time_index, space_index].unsqueeze(-1)
        else:
            y_tangent = torch.randint(1, len(axes[1]) - 1,
                                      (batch_size, count), device=device)
            x_tangent = torch.randint(1, len(axes[2]) - 1,
                                      (batch_size, count), device=device)
            y_index = torch.where(
                side < 2, y_tangent,
                torch.where(side == 2, 0, len(axes[1]) - 1),
            )
            x_index = torch.where(
                side < 2, torch.where(side == 0, 0, len(axes[2]) - 1), x_tangent,
            )
            coordinates[..., 1] = axes[1][y_index]
            coordinates[..., 2] = axes[2][x_index]
            normal = torch.zeros(batch_size, count, 2, device=device, dtype=dtype)
            normal[..., 0] = torch.where(
                side == 2, -1.0, torch.where(side == 3, 1.0, 0.0)
            )
            normal[..., 1] = torch.where(
                side == 0, -1.0, torch.where(side == 1, 1.0, 0.0)
            )
            rhs = packed[batch, time_index, y_index, x_index].unsqueeze(-1)

        minimum = torch.as_tensor(self.value_min, device=device, dtype=dtype)
        maximum = torch.as_tensor(self.value_max, device=device, dtype=dtype)
        physical_rhs = (rhs + 1.0) * 0.5 * (maximum - minimum) + minimum
        return coordinates.requires_grad_(True), normal, physical_rhs

    @staticmethod
    def _expand_batch(values: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Broadcast deterministic coordinates/normals without duplicating storage."""
        return values.unsqueeze(0).expand(batch_size, -1, -1)

    def _grid_sample(self, packed: torch.Tensor, axes: tuple[torch.Tensor, ...],
                     batch_size: int, device: torch.device,
                     dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use every saved post-initial-time boundary node exactly once.

        Corners are excluded in 2D so each Robin point has one unambiguous
        outward normal and the point set matches the packed boundary slots.
        """
        time_axis = axes[0]
        time_index = torch.arange(1, len(time_axis), device=device)
        if len(axes) == 2:
            space_axis = axes[1]
            time_index = time_index.repeat_interleave(2)
            space_index = torch.tensor(
                [0, len(space_axis) - 1], device=device, dtype=torch.long
            ).repeat(len(time_axis) - 1)
            base_coordinates = torch.stack((
                time_axis[time_index], space_axis[space_index],
            ), dim=-1)
            base_normal = torch.where(
                space_index == 0,
                torch.tensor(-1.0, device=device, dtype=dtype),
                torch.tensor(1.0, device=device, dtype=dtype),
            ).unsqueeze(-1)
            rhs = packed[:, time_index, space_index].unsqueeze(-1)
        else:
            y_axis, x_axis = axes[1:]
            y_tangent = torch.arange(1, len(y_axis) - 1, device=device)
            x_tangent = torch.arange(1, len(x_axis) - 1, device=device)

            def side_indices(tangent: torch.Tensor, *, fixed_axis: int, fixed_index: int):
                side_time = time_index[:, None].expand(-1, tangent.numel()).reshape(-1)
                side_tangent = tangent[None, :].expand(time_index.numel(), -1).reshape(-1)
                if fixed_axis == 0:
                    side_y = torch.full_like(side_tangent, fixed_index)
                    side_x = side_tangent
                else:
                    side_y = side_tangent
                    side_x = torch.full_like(side_tangent, fixed_index)
                return side_time, side_y, side_x

            left = side_indices(y_tangent, fixed_axis=1, fixed_index=0)
            right = side_indices(y_tangent, fixed_axis=1, fixed_index=len(x_axis) - 1)
            bottom = side_indices(x_tangent, fixed_axis=0, fixed_index=0)
            top = side_indices(x_tangent, fixed_axis=0, fixed_index=len(y_axis) - 1)
            time_index, y_index, x_index = tuple(
                torch.cat(parts) for parts in zip(left, right, bottom, top)
            )
            base_coordinates = torch.stack((
                time_axis[time_index], y_axis[y_index], x_axis[x_index],
            ), dim=-1)
            count = base_coordinates.shape[0]
            base_normal = torch.zeros(count, 2, device=device, dtype=dtype)
            left_count = left[0].numel()
            right_count = right[0].numel()
            bottom_count = bottom[0].numel()
            base_normal[:left_count, 1] = -1.0
            base_normal[left_count:left_count + right_count, 1] = 1.0
            base_normal[left_count + right_count:left_count + right_count + bottom_count, 0] = -1.0
            base_normal[left_count + right_count + bottom_count:, 0] = 1.0
            rhs = packed[:, time_index, y_index, x_index].unsqueeze(-1)

        coordinates = self._expand_batch(base_coordinates, batch_size).clone().requires_grad_(True)
        normal = self._expand_batch(base_normal, batch_size).clone()
        minimum = torch.as_tensor(self.value_min, device=device, dtype=dtype)
        maximum = torch.as_tensor(self.value_max, device=device, dtype=dtype)
        physical_rhs = (rhs + 1.0) * 0.5 * (maximum - minimum) + minimum
        return coordinates, normal, physical_rhs

    def __call__(self, prediction=None, *, field=None, model=None, model_input=None,
                 timestep=None, latent_grid=None, **_ignored) -> torch.Tensor:
        if model_input is None:
            raise ValueError("model_input is required for Robin boundary loss")
        if torch.is_autocast_enabled():
            with torch.autocast(device_type=model_input.device.type, enabled=False):
                return self(
                    prediction=prediction.float() if prediction is not None else None,
                    field=field, model=model, model_input=model_input.float(),
                    timestep=timestep,
                    latent_grid=latent_grid.float() if latent_grid is not None else None,
                    **_ignored,
                )
        if field is None:
            if model is None:
                raise ValueError("provide field=... or model=...")
            if hasattr(model, "encode") and hasattr(model, "decoder"):
                if latent_grid is None:
                    latent_grid = model.encode(model_input, timestep)
                field = lambda query: model.decoder(latent_grid, query)
            elif hasattr(model, "branch") and hasattr(model, "trunk"):
                g_left = _ignored.get("g_left")
                field = lambda query: model(
                    model_input, timestep, coords=query, g_left=g_left
                )
            else:
                raise TypeError("Robin loss requires a continuous decoder or DeepONet trunk")
        normalized_field = field
        minimum = torch.as_tensor(self.value_min)
        maximum = torch.as_tensor(self.value_max)
        field = lambda query: (
            (normalized_field(query) + 1.0) * 0.5
            * (maximum.to(query) - minimum.to(query)) + minimum.to(query)
        )
        coordinates, normal, rhs = self._sample(
            model_input[:, 0], g_left=_ignored.get("g_left")
        )
        residual = radiative_robin_residual(
            field, coordinates, normal, rhs, **self.kwargs
        )
        return F.smooth_l1_loss(residual, torch.zeros_like(residual))


class FusedRadiativePhysicsLoss:
    """Share continuous decoding and first derivatives for PDE and Robin terms.

    This is intentionally a small adapter around the existing loss objects.
    It is enabled only by a case configuration flag, so the historical
    separate-loss path remains unchanged for every other experiment.
    """

    fused = True

    def __init__(self, physics_loss: CollocationPhysicsLoss,
                 boundary_loss: RobinBoundaryLoss):
        if physics_loss.residual != "radiative_diffusion":
            raise ValueError("fused loss currently supports radiative diffusion only")
        self.physics_loss = physics_loss
        self.boundary_loss = boundary_loss
        self._cached_key = None
        self._cached_values = None

    @staticmethod
    def _field(model, model_input, timestep, latent_grid):
        if hasattr(model, "encode") and hasattr(model, "decoder"):
            if latent_grid is None:
                latent_grid = model.encode(model_input, timestep)
            return (
                lambda query: model.decoder(
                    latent_grid, query.expand(latent_grid.shape[0], -1, -1)
                ),
                latent_grid,
            )
        raise TypeError("fused radiative loss requires a continuous decoder")

    def _evaluate(self, *, model, model_input, timestep, latent_grid, g_left,
                  condition_field=None, **_ignored):
        batch_size = model_input.shape[0]
        device, dtype = model_input.device, model_input.dtype
        pde_coords = self.physics_loss.sample_coordinates(batch_size, device, dtype)
        robin_coords, normal, rhs = self.boundary_loss._sample(
            model_input[:, 0], g_left=g_left
        )
        pde_count = pde_coords.shape[1]
        coordinates = torch.cat((pde_coords, robin_coords), dim=1).requires_grad_(True)
        normalized_field, _ = self._field(model, model_input, timestep, latent_grid)
        minimum = torch.as_tensor(self.physics_loss.value_min, device=device, dtype=dtype)
        maximum = torch.as_tensor(self.physics_loss.value_max, device=device, dtype=dtype)
        field = lambda query: (
            (normalized_field(query) + 1.0) * 0.5 * (maximum - minimum) + minimum
        )

        energy = field(coordinates)
        gradient = _first_derivative(energy, coordinates)

        pde_energy = energy[:, :pde_count]
        pde_coords = coordinates[:, :pde_count]
        pde_gradient = gradient[:, :pde_count]
        e_t = pde_gradient[..., 0:1] * self.physics_loss.kwargs["dt_scale"]
        e_x = pde_gradient[..., 1:2] * self.physics_loss.kwargs["dx_scale"]
        material = self.physics_loss.kwargs["material"]
        diffusivity = _radiative_diffusivity(
            pde_energy, e_x, None, _material_at(material, pde_coords, pde_energy),
            use_flux_limiter=self.physics_loss.kwargs["use_flux_limiter"],
            limiter_norm=self.physics_loss.kwargs["limiter_norm"], eps=1e-8,
        )
        flux = -diffusivity * e_x
        flux_gradient = _first_derivative(
            flux, coordinates
        )
        div_flux = flux_gradient[:, :pde_count, 1:2] * self.physics_loss.kwargs["dx_scale"]
        residual = e_t + div_flux
        source = self.physics_loss.kwargs.get("source")
        if source is not None:
            residual = residual - (
                source(pde_coords) if callable(source) else
                torch.as_tensor(source, dtype=residual.dtype, device=residual.device)
            )
        pde_point_loss = F.huber_loss(
            residual, torch.zeros_like(residual),
            delta=self.physics_loss.huber_delta, reduction="none",
        )
        pde_point_loss = pde_point_loss.reshape(batch_size, pde_count, -1).mean(dim=-1)
        pde_value = pde_point_loss.mean()

        robin_energy = energy[:, pde_count:]
        robin_gradient = gradient[:, pde_count:]
        e_x = robin_gradient[..., 1:2] * self.boundary_loss.kwargs["dx_scale"]
        robin_material = self.boundary_loss.kwargs["material"]
        robin_diffusivity = _radiative_diffusivity(
            robin_energy, e_x, None,
            _material_at(robin_material, robin_coords, robin_energy),
            use_flux_limiter=self.boundary_loss.kwargs["use_flux_limiter"],
            limiter_norm=self.boundary_loss.kwargs["limiter_norm"], eps=1e-8,
        )
        normal_derivative = e_x * normal[..., 0:1]
        robin_residual = 0.5 * robin_energy + robin_diffusivity * normal_derivative - rhs
        robin_value = F.smooth_l1_loss(
            robin_residual, torch.zeros_like(robin_residual)
        )
        return pde_value, robin_value

    def _get(self, kind: str, kwargs: dict) -> torch.Tensor:
        key = tuple(id(kwargs.get(name)) for name in ("model_input", "latent_grid", "g_left"))
        if self._cached_key != key or self._cached_values is None:
            self._cached_values = self._evaluate(**kwargs)
            self._cached_key = key
        value = self._cached_values[0 if kind == "physics" else 1]
        if kind == "boundary":
            self._cached_key = None
            self._cached_values = None
        return value

    def physics(self, prediction=None, **kwargs):
        return self._get("physics", kwargs)

    def boundary(self, prediction=None, **kwargs):
        return self._get("boundary", kwargs)
