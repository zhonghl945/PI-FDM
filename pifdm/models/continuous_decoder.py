"""Continuous coordinate decoder shared by continuous diffusion and operator models."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _cubic_bspline(distance: torch.Tensor) -> torch.Tensor:
    """Centered cardinal cubic B-spline, with C2 coordinate derivatives."""
    radius = distance.abs()
    inner = 2.0 / 3.0 - radius.square() + 0.5 * radius.pow(3)
    outer = (2.0 - radius).clamp_min(0.0).pow(3) / 6.0
    return torch.where(radius < 1.0, inner, torch.where(radius < 2.0, outer, torch.zeros_like(radius)))


def _axis_stencil(coordinate: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = (coordinate + 1.0) * 0.5 * (size - 1)
    base = scaled.floor().long()
    offsets = torch.arange(-1, 3, device=coordinate.device)
    raw_indices = base[..., None] + offsets
    weights = _cubic_bspline(scaled[..., None] - raw_indices.to(scaled.dtype))
    return raw_indices.clamp(0, size - 1), weights


def _sample_bspline(latent_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sample a 2D/3D latent lattice with a local separable C2 stencil."""
    batch, channels = latent_grid.shape[:2]
    spatial_shape = latent_grid.shape[2:]
    stencils = [_axis_stencil(coords[..., axis], size) for axis, size in enumerate(spatial_shape)]
    if len(spatial_shape) == 2:
        (i0, w0), (i1, w1) = stencils
        linear = i0[..., :, None] * spatial_shape[1] + i1[..., None, :]
        weights = w0[..., :, None] * w1[..., None, :]
    elif len(spatial_shape) == 3:
        (i0, w0), (i1, w1), (i2, w2) = stencils
        linear = ((i0[..., :, None, None] * spatial_shape[1] + i1[..., None, :, None])
                  * spatial_shape[2] + i2[..., None, None, :])
        weights = w0[..., :, None, None] * w1[..., None, :, None] * w2[..., None, None, :]
    else:
        raise ValueError("only 2D and 3D coordinate fields are supported")
    stencil_size = 4 ** len(spatial_shape)
    linear = linear.reshape(batch, 1, -1).expand(-1, channels, -1)
    gathered = latent_grid.flatten(2).gather(2, linear)
    gathered = gathered.reshape(batch, channels, coords.shape[1], stencil_size)
    weights = weights.reshape(batch, 1, coords.shape[1], stencil_size)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
    return (gathered * weights).sum(dim=-1).transpose(1, 2)


def _axis_interpolation_matrix(
    input_size: int, coordinates: torch.Tensor
) -> torch.Tensor:
    """Return the normalized cubic B-spline map for one regular axis.

    Clamped stencil entries are accumulated into the same matrix column.  This
    exactly preserves the boundary behavior of :func:`_sample_bspline` while
    avoiding an explicit four-neighbor dimension in the full-grid path.
    """
    indices, weights = _axis_stencil(coordinates, input_size)
    matrix = weights.new_zeros(len(coordinates), input_size)
    matrix.scatter_add_(1, indices, weights)
    return matrix / matrix.sum(dim=1, keepdim=True).clamp_min(torch.finfo(coordinates.dtype).eps)


def _apply_axis_interpolation(
    values: torch.Tensor, matrix: torch.Tensor, spatial_axis: int
) -> torch.Tensor:
    """Apply ``(output,input)`` interpolation along one spatial tensor axis."""
    tensor_axis = spatial_axis + 2
    moved = values.movedim(tensor_axis, -1)
    interpolated = torch.matmul(moved, matrix.transpose(0, 1))
    return interpolated.movedim(-1, tensor_axis)


def _grid_axes(
    output_shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype,
    axes: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    if axes is None:
        return tuple(
            torch.linspace(-1.0, 1.0, int(size), device=device, dtype=dtype)
            for size in output_shape
        )
    if len(axes) != len(output_shape):
        raise ValueError("grid axes must match output_shape")
    result = tuple(torch.as_tensor(axis, device=device, dtype=dtype) for axis in axes)
    for size, axis in zip(output_shape, result):
        if axis.ndim != 1 or len(axis) != int(size):
            raise ValueError("each grid axis must be one-dimensional and match output_shape")
        if len(axis) > 1 and not torch.all(axis[1:] > axis[:-1]):
            raise ValueError("grid axes must be strictly increasing")
        if axis.min() < -1.00001 or axis.max() > 1.00001:
            raise ValueError("grid axes must be normalized to [-1, 1]")
    return result


def _natural_cubic_second_derivatives(
    knots: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Return natural-cubic second derivatives for fixed one-dimensional data."""
    if knots.ndim != 1 or values.ndim != 1 or len(knots) != len(values):
        raise ValueError("spline knots and values must be matching one-dimensional tensors")
    if len(knots) < 2 or not torch.all(knots[1:] > knots[:-1]):
        raise ValueError("spline knots must contain at least two strictly increasing values")
    if len(knots) == 2:
        return torch.zeros_like(values)
    spacing = knots[1:] - knots[:-1]
    matrix = knots.new_zeros((len(knots) - 2, len(knots) - 2))
    rhs = knots.new_zeros(len(knots) - 2)
    diagonal = 2.0 * (spacing[:-1] + spacing[1:])
    matrix.diagonal().copy_(diagonal)
    if len(knots) > 3:
        matrix.diagonal(-1).copy_(spacing[1:-1])
        matrix.diagonal(1).copy_(spacing[1:-1])
    rhs.copy_(6.0 * (
        (values[2:] - values[1:-1]) / spacing[1:]
        - (values[1:-1] - values[:-2]) / spacing[:-1]
    ))
    second = torch.zeros_like(values)
    second[1:-1] = torch.linalg.solve(matrix, rhs)
    return second


def _warp_axis_to_uniform_index(
    query: torch.Tensor, knots: torch.Tensor,
    second_derivatives: torch.Tensor | None = None,
    extrapolate: bool = False,
) -> torch.Tensor:
    """Map a physical normalized axis to its uniformly indexed tensor axis.

    Saved fields are tensors indexed by sample number, while their physical
    coordinates may be nonuniform after subsampling.  This C2 spline maps the
    physical coordinates back to ``linspace(-1, 1, len(knots))`` so local
    latent lookup follows tensor locations.  Fourier features still receive
    the original physical coordinates.
    """
    work_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    x = knots.to(device=query.device, dtype=work_dtype)
    q = query.to(dtype=work_dtype)
    if not extrapolate:
        q = q.clamp(x[0], x[-1])
    target = torch.linspace(-1.0, 1.0, len(x), device=query.device, dtype=work_dtype)
    if second_derivatives is None:
        second = _natural_cubic_second_derivatives(x, target)
    else:
        second = second_derivatives.to(device=query.device, dtype=work_dtype)
    interval = torch.searchsorted(x, q.contiguous(), right=True).sub(1).clamp(0, len(x) - 2)
    left_x, right_x = x[interval], x[interval + 1]
    width = right_x - left_x
    left_weight = (right_x - q) / width
    right_weight = (q - left_x) / width
    mapped = (
        left_weight * target[interval]
        + right_weight * target[interval + 1]
        + (
            (left_weight.pow(3) - left_weight) * second[interval]
            + (right_weight.pow(3) - right_weight) * second[interval + 1]
        ) * width.square() / 6.0
    )
    return mapped.to(dtype=query.dtype)


def _sample_bspline_grid(
    latent_grid: torch.Tensor, output_shape: tuple[int, ...],
    axes: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Interpolate a complete regular grid using separable one-dimensional maps.

    The result is mathematically identical to querying every point with
    :func:`_sample_bspline`, but never constructs ``(B,C,N,4**dimension)``.
    """
    spatial_shape = latent_grid.shape[2:]
    if len(spatial_shape) not in (2, 3) or len(output_shape) != len(spatial_shape):
        raise ValueError("output_shape must match a 2D or 3D latent lattice")
    query_axes = _grid_axes(
        output_shape, device=latent_grid.device, dtype=latent_grid.dtype, axes=axes
    )
    result = latent_grid
    for axis, (input_size, coordinates) in enumerate(zip(spatial_shape, query_axes)):
        matrix = _axis_interpolation_matrix(input_size, coordinates)
        result = _apply_axis_interpolation(result, matrix, axis)
    return result


def _regular_grid_coordinates(
    output_shape: tuple[int, ...], *, batch: int, device: torch.device, dtype: torch.dtype,
    axes: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    query_axes = _grid_axes(output_shape, device=device, dtype=dtype, axes=axes)
    coordinates = torch.stack(torch.meshgrid(*query_axes, indexing="ij"), dim=-1).reshape(1, -1, len(output_shape))
    return coordinates.expand(batch, -1, -1)


class ContinuousDecoder(nn.Module):
    """Decode a latent grid at arbitrary normalized coordinates.

    A normalized cubic B-spline stencil is used instead of ``grid_sample``.
    PyTorch grid sampling does not reliably provide the double backward needed
    by second-order PDE residuals; the polynomial stencil is C2 in coordinates.
    A fixed isotropic random projection of coordinates and material features is
    added to the interpolated latent feature before the decoder MLP.
    """

    def __init__(self, latent_dim: int, coord_dim: int = 2,
                 coordinate_projection_scale: float = 0.05,
                 coordinate_projection_seed: int = 0, chunk_size: int = 16384,
                 checkpoint_decoding: bool = True, grid_axes=None, material_function=None,
                 output_floor: float | None = None, output_softplus_beta: float = 10.0,
                 output_softplus_scale: float | None = None,
                 output_softplus_initial_bias: float | None = None,
                 extrapolate_grid_axes: bool = False):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.coord_dim = coord_dim
        self.chunk_size = max(1, int(chunk_size))
        self.checkpoint_decoding = bool(checkpoint_decoding)
        self.material_function = material_function
        self.extrapolate_grid_axes = bool(extrapolate_grid_axes)
        self.material_feature_dim = 1 if material_function is not None else 0
        self.coordinate_projection_scale = float(coordinate_projection_scale)
        self.coordinate_projection_seed = int(coordinate_projection_seed)
        if self.coordinate_projection_scale <= 0.0:
            raise ValueError("coordinate_projection_scale must be positive")
        self.output_floor = None if output_floor is None else float(output_floor)
        self.output_softplus_beta = float(output_softplus_beta)
        self.output_softplus_scale = (
            None if output_softplus_scale is None else float(output_softplus_scale)
        )
        if self.output_softplus_scale is not None and self.output_softplus_scale <= 0.0:
            raise ValueError("output_softplus_scale must be positive")
        if self.output_softplus_beta <= 0.0:
            raise ValueError("output_softplus_beta must be positive")
        self._grid_axis_count = 0
        if grid_axes is not None:
            if len(grid_axes) != coord_dim:
                raise ValueError("grid_axes must match coord_dim")
            for index, axis in enumerate(grid_axes):
                values = torch.as_tensor(axis, dtype=torch.float64)
                if values.ndim != 1 or len(values) < 2:
                    raise ValueError("each decoder grid axis must contain at least two values")
                if not torch.all(values[1:] > values[:-1]):
                    raise ValueError("decoder grid axes must be strictly increasing")
                if values.min() < -1.00001 or values.max() > 1.00001:
                    raise ValueError("decoder grid axes must be normalized to [-1, 1]")
                target = torch.linspace(-1.0, 1.0, len(values), dtype=values.dtype)
                second = _natural_cubic_second_derivatives(values, target)
                self.register_buffer(f"_grid_axis_{index}", values, persistent=False)
                self.register_buffer(f"_grid_axis_second_{index}", second, persistent=False)
            self._grid_axis_count = coord_dim
        projection_input_dim = self.coord_dim + self.material_feature_dim
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.coordinate_projection_seed)
        random_matrix = torch.randn(
            self.latent_dim, projection_input_dim, generator=generator
        )
        if self.latent_dim >= projection_input_dim:
            orthogonal, _ = torch.linalg.qr(random_matrix, mode="reduced")
            projection = orthogonal * self.coordinate_projection_scale
        else:
            projection = random_matrix / random_matrix.norm(dim=0, keepdim=True).clamp_min(1e-12)
            projection = projection * self.coordinate_projection_scale
        self.register_buffer("coordinate_projection", projection)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, 1)
        )
        if output_softplus_initial_bias is not None:
            nn.init.constant_(self.mlp[-1].bias, float(output_softplus_initial_bias))

    def _material_features(self, coords: torch.Tensor) -> torch.Tensor:
        if self.material_function is None:
            return coords.new_empty((*coords.shape[:-1], 0))
        value = self.material_function(coords)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=coords.device, dtype=coords.dtype)
        else:
            value = value.to(device=coords.device, dtype=coords.dtype)
        if value.ndim == 0:
            value = torch.ones(
                (*coords.shape[:-1], 1), device=coords.device, dtype=coords.dtype
            ) * value
        elif value.ndim == coords.ndim - 1:
            value = value.unsqueeze(-1)
        expected = (*coords.shape[:-1], 1)
        if value.shape[0] == 1 and coords.shape[0] != 1:
            value = value.expand(coords.shape[0], -1, -1)
        if value.shape != expected:
            raise ValueError(
                f"material feature has shape {tuple(value.shape)}, expected {expected}"
            )
        return value

    def _decode_local(self, local: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, coords.shape[1], self.chunk_size):
            stop = min(start + self.chunk_size, coords.shape[1])
            query = coords[:, start:stop]
            encoded = torch.cat(
                (query, self._material_features(query)), dim=-1
            )
            hidden = self.mlp[0](
                local[:, start:stop]
                + F.linear(encoded, self.coordinate_projection)
            )
            hidden = self.mlp[1](hidden)
            hidden = self.mlp[2](hidden)
            hidden = self.mlp[3](hidden)
            output = self.mlp[4](hidden)
            if self.output_floor is not None:
                if self.output_softplus_scale is None:
                    output = self.output_floor + F.softplus(
                        output - self.output_floor, beta=self.output_softplus_beta
                    )
                else:
                    output = self.output_floor + self.output_softplus_scale * F.softplus(
                        output, beta=self.output_softplus_beta
                    )
            outputs.append(output)
        return torch.cat(outputs, dim=1)

    def _interpolation_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        if not self._grid_axis_count:
            return coords
        warped = [
            _warp_axis_to_uniform_index(
                coords[..., index],
                getattr(self, f"_grid_axis_{index}"),
                getattr(self, f"_grid_axis_second_{index}"),
                extrapolate=self.extrapolate_grid_axes,
            )
            for index in range(self._grid_axis_count)
        ]
        return torch.stack(warped, dim=-1)

    def _decode_queries(self, latent_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        local = _sample_bspline(latent_grid, self._interpolation_coordinates(coords))
        return self._decode_local(local, coords)

    def _decode_regular_grid(self, latent_grid: torch.Tensor, output_shape: tuple[int, ...],
                             axes: tuple[torch.Tensor, ...] | None) -> torch.Tensor:
        physical_axes = _grid_axes(
            output_shape, device=latent_grid.device, dtype=latent_grid.dtype, axes=axes
        )
        if self._grid_axis_count:
            interpolation_axes = tuple(
                _warp_axis_to_uniform_index(
                    physical_axis,
                    getattr(self, f"_grid_axis_{index}"),
                    getattr(self, f"_grid_axis_second_{index}"),
                    extrapolate=self.extrapolate_grid_axes,
                )
                for index, physical_axis in enumerate(physical_axes)
            )
        else:
            interpolation_axes = physical_axes
        local_grid = _sample_bspline_grid(
            latent_grid, output_shape, axes=interpolation_axes
        )
        local = local_grid.flatten(2).transpose(1, 2)
        coords = _regular_grid_coordinates(
            output_shape, batch=1, device=latent_grid.device, dtype=latent_grid.dtype,
            axes=physical_axes,
        )
        return self._decode_local(local, coords)

    def _should_checkpoint(self) -> bool:
        return self.checkpoint_decoding and self.training and torch.is_grad_enabled()

    def forward(self, latent_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Decode arbitrary coordinates, used by off-grid physics collocation."""
        if latent_grid.ndim not in (4, 5):
            raise ValueError("latent_grid must be (B,C,H,W) or (B,C,D,H,W)")
        if coords.ndim != 3 or coords.shape[-1] != self.coord_dim:
            raise ValueError(f"coords must have shape (B,N,{self.coord_dim})")
        if coords.shape[0] != latent_grid.shape[0]:
            raise ValueError("coords and latent_grid must have the same batch size")
        outputs = []
        for start in range(0, coords.shape[1], self.chunk_size):
            query = coords[:, start:start + self.chunk_size]
            if self._should_checkpoint():
                decoded = checkpoint(self._decode_queries, latent_grid, query, use_reentrant=False)
            else:
                decoded = self._decode_queries(latent_grid, query)
            outputs.append(decoded)
        return torch.cat(outputs, dim=1)

    def forward_grid(self, latent_grid: torch.Tensor, output_shape: tuple[int, ...],
                     axes: tuple[torch.Tensor, ...] | None = None) -> torch.Tensor:
        """Decode every point of a regular grid without discarding supervision."""
        output_shape = tuple(int(size) for size in output_shape)
        if latent_grid.ndim not in (4, 5):
            raise ValueError("latent_grid must be (B,C,H,W) or (B,C,D,H,W)")
        if len(output_shape) != self.coord_dim:
            raise ValueError(f"output_shape must contain {self.coord_dim} axes")
        if self._should_checkpoint():
            function = lambda values: self._decode_regular_grid(values, output_shape, axes)
            return checkpoint(function, latent_grid, use_reentrant=False)
        return self._decode_regular_grid(latent_grid, output_shape, axes)
