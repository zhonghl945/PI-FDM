"""Dataset adapters for the HDF5 layouts produced by the original project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import h5py
import torch
from torch.utils.data import Dataset


class FieldDataset(Dataset):
    def __init__(self, fields: torch.Tensor):
        if fields.ndim < 3:
            raise ValueError("fields must include sample and at least two field axes")
        self.fields = fields.float()

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, index):
        return self.fields[index]


class ConditionedFieldDataset(Dataset):
    """Dataset returning a solution field and a sample-level clean condition."""

    def __init__(self, fields: torch.Tensor, g_left: torch.Tensor):
        if fields.ndim < 3:
            raise ValueError("fields must include sample and at least two field axes")
        if (g_left.ndim != 3 or g_left.shape[:2] != fields.shape[:2]
                or g_left.shape[2] != fields.shape[2]):
            raise ValueError(
                "g_left must have shape (samples, time, y) and match field samples/time/y"
            )
        self.fields = fields.float()
        self.g_left = g_left.float()

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, index):
        return {"field": self.fields[index], "g_left": self.g_left[index]}


def _normalise(values: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    minimum, maximum = values.amin(), values.amax()
    scale = (maximum - minimum).clamp_min(torch.finfo(values.dtype).eps)
    return 2 * (values - minimum) / scale - 1, {"min": minimum, "max": maximum}


def load_hdf5_field(path: str | Path, *, add_boundaries: bool = True,
                    domain: Dict[str, Sequence[float]] | None = None,
                    normalise: bool = True, value_scalers: Dict[str, torch.Tensor] | None = None
                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Load either 1D or 2D saved interior fields with explicit coordinates.

    The 2D adapter fills *all* boundaries from the file when available and
    otherwise records them as zero only when the file explicitly lacks those
    datasets.  This prevents silently treating unknown boundaries as physical
    zeros.
    """

    with h5py.File(path, "r") as handle:
        interior = torch.from_numpy(handle["u"][:]).float()
        axes = {name: torch.from_numpy(handle[name][:]).float() for name in ("t", "x", "y") if name in handle}
        boundary_names = ("bL", "bR", "bB", "bT")
        boundaries = {name: torch.from_numpy(handle[name][:]).float() for name in boundary_names if name in handle}

    missing_boundaries = []
    if add_boundaries and interior.ndim == 3:
        if "bL" not in boundaries or "bR" not in boundaries:
            raise KeyError("1D files must contain bL and bR when add_boundaries=True")
        field = torch.cat((boundaries["bL"].unsqueeze(-1), interior, boundaries["bR"].unsqueeze(-1)), dim=-1)
        if domain is None or "x" not in domain:
            raise ValueError("domain.x=[x_min,x_max] is required when adding 1D boundaries")
        axes["x"] = torch.cat((torch.tensor([domain["x"][0]]), axes["x"], torch.tensor([domain["x"][1]])))
    elif add_boundaries and interior.ndim == 4:
        b, nt, ny, nx = interior.shape
        # Corners are frozen inputs too.  Starting from zeros avoids the
        # uninitialised corner values produced by torch.empty in the archived
        # layout adapter.
        field = torch.zeros((b, nt, ny + 2, nx + 2), dtype=interior.dtype)
        field[:, :, 1:-1, 1:-1] = interior
        for name, index in (("bL", (slice(None), slice(None), slice(1, -1), 0)),
                            ("bR", (slice(None), slice(None), slice(1, -1), -1)),
                            ("bB", (slice(None), slice(None), 0, slice(1, -1))),
                            ("bT", (slice(None), slice(None), -1, slice(1, -1)))):
            if name in boundaries:
                field[index] = boundaries[name]
            else:
                # The archived 2D generator uses known homogeneous zero
                # boundaries but stores only bL.  Preserve that convention and
                # expose the omission to callers rather than failing or
                # silently inventing a non-zero condition.
                field[index] = 0.0
                missing_boundaries.append(name)
        # The packed representation assigns corners to the left/right sides;
        # bottom/top omit them consistently with extract_frozen_values().
        if "bL" in boundaries:
            field[:, :, 0, 0] = boundaries["bL"][:, :, 0]
            field[:, :, -1, 0] = boundaries["bL"][:, :, -1]
        if "bR" in boundaries:
            field[:, :, 0, -1] = boundaries["bR"][:, :, 0]
            field[:, :, -1, -1] = boundaries["bR"][:, :, -1]
        if domain is None or "x" not in domain or "y" not in domain:
            raise ValueError("domain.x and domain.y are required when adding 2D boundaries")
        axes["x"] = torch.cat((torch.tensor([domain["x"][0]]), axes["x"], torch.tensor([domain["x"][1]])))
        axes["y"] = torch.cat((torch.tensor([domain["y"][0]]), axes["y"], torch.tensor([domain["y"][1]])))
    else:
        field = interior

    if normalise and value_scalers is not None:
        minimum = torch.as_tensor(value_scalers["min"], dtype=field.dtype).detach().cpu()
        maximum = torch.as_tensor(value_scalers["max"], dtype=field.dtype).detach().cpu()
        scale = (maximum - minimum).clamp_min(torch.finfo(field.dtype).eps)
        normalised, scalers = 2 * (field - minimum) / scale - 1, {"min": minimum, "max": maximum}
    else:
        normalised, scalers = _normalise(field) if normalise else (field, {"min": field.amin(), "max": field.amax()})
    if missing_boundaries:
        scalers["missing_boundaries"] = tuple(missing_boundaries)
    return normalised, axes, scalers


def load_radiative_2d_field(
    path: str | Path, *, normalise: bool = True,
    value_scalers: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Load ``radiative_2d``'s interior field and saved left Robin history.

    This layout deliberately keeps ``gL`` separate from ``u``: ``gL`` is a
    Robin right-hand side, not a boundary value of the energy field.  The
    same scalar min/max is applied to both arrays so a checkpoint has one
    value coordinate system for reconstruction and boundary losses.
    """

    with h5py.File(path, "r") as handle:
        if "gL" not in handle:
            raise KeyError("radiative_2d data must contain the saved left Robin dataset 'gL'")
        field = torch.from_numpy(handle["u"][:]).float()
        g_left = torch.from_numpy(handle["gL"][:]).float()
        axes = {
            name: torch.from_numpy(handle[name][:]).float()
            for name in ("t", "y", "x")
        }

    if field.ndim != 4:
        raise ValueError(f"radiative_2d interior field must be 4D, got {tuple(field.shape)}")
    expected_g = (field.shape[0], field.shape[1], field.shape[2])
    if tuple(g_left.shape) != expected_g:
        raise ValueError(
            f"radiative_2d gL shape {tuple(g_left.shape)} does not match {expected_g}"
        )
    for name, size in (("t", field.shape[1]), ("y", field.shape[2]), ("x", field.shape[3])):
        if axes[name].ndim != 1 or len(axes[name]) != size:
            raise ValueError(
                f"radiative_2d axis {name!r} has shape {tuple(axes[name].shape)}, expected ({size},)"
            )
        if len(axes[name]) > 1 and not torch.all(axes[name][1:] > axes[name][:-1]):
            raise ValueError(f"radiative_2d axis {name!r} must be strictly increasing")
    if not torch.isfinite(field).all() or not torch.isfinite(g_left).all():
        raise ValueError("radiative_2d field or gL contains NaN or infinity")

    if value_scalers is None:
        minimum = torch.minimum(field.amin(), g_left.amin())
        maximum = torch.maximum(field.amax(), g_left.amax())
        scalers = {"min": minimum, "max": maximum}
    else:
        minimum = torch.as_tensor(value_scalers["min"], dtype=field.dtype).detach().cpu()
        maximum = torch.as_tensor(value_scalers["max"], dtype=field.dtype).detach().cpu()
        scalers = {"min": minimum, "max": maximum}
    if not normalise:
        return field, g_left, axes, scalers
    scale = (maximum - minimum).clamp_min(torch.finfo(field.dtype).eps)
    return (
        2.0 * (field - minimum) / scale - 1.0,
        2.0 * (g_left - minimum) / scale - 1.0,
        axes,
        scalers,
    )
