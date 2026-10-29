"""Coordinate transforms used by continuous decoders and PDE residuals."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CoordinateSystem:
    """Map physical coordinates to the decoder's [-1, 1] coordinates.

    Derivative scales are stored explicitly, avoiding the common error of
    applying a single ``2/L`` scale to both time and space.
    """

    axes: tuple[torch.Tensor, ...]
    physical_bounds: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self):
        if self.physical_bounds is not None:
            if len(self.physical_bounds) != len(self.axes):
                raise ValueError("physical_bounds must have one pair per coordinate axis")
            for bounds in self.physical_bounds:
                if len(bounds) != 2 or not float(bounds[1]) > float(bounds[0]):
                    raise ValueError("each physical coordinate bound must be [lower, upper]")

    @property
    def _bounds(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if self.physical_bounds is None:
            return tuple((axis.min(), axis.max()) for axis in self.axes)
        return tuple(
            (torch.as_tensor(bounds[0], dtype=axis.dtype, device=axis.device),
             torch.as_tensor(bounds[1], dtype=axis.dtype, device=axis.device))
            for axis, bounds in zip(self.axes, self.physical_bounds)
        )

    @property
    def normalized_axes(self):
        return tuple(
            2 * (axis - lower) / (upper - lower) - 1
            for axis, (lower, upper) in zip(self.axes, self._bounds)
        )

    @property
    def derivative_scales(self):
        return tuple(2 / (upper - lower) for lower, upper in self._bounds)

    def mesh(self):
        axes = self.normalized_axes
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)

    def to_physical(self, coords: torch.Tensor) -> torch.Tensor:
        mins = torch.stack([bounds[0] for bounds in self._bounds]).to(coords)
        maxs = torch.stack([bounds[1] for bounds in self._bounds]).to(coords)
        return (coords + 1) * 0.5 * (maxs - mins) + mins
