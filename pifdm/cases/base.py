"""Case protocol shared by data generation, training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class CaseData:
    """A generated dataset plus the metadata needed by the physics loss."""

    fields: np.ndarray
    axes: Dict[str, np.ndarray]
    boundaries: Dict[str, np.ndarray]
    material: Optional[np.ndarray] = None
    source: Optional[np.ndarray] = None
    metadata: Dict[str, Any] | None = None


class CaseAdapter:
    """Interface implemented by each traditional numerical PDE case."""

    name: str
    dimension: int

    def generate(self, config: dict[str, Any], output_path: str | Path) -> Path:
        raise NotImplementedError

    def load(self, path: str | Path) -> CaseData:
        raise NotImplementedError

    def validate(self, data: CaseData) -> None:
        if data.fields.ndim != self.dimension + 2:
            raise ValueError(f"{self.name}: expected fields with {self.dimension + 2} dimensions, got {data.fields.shape}")
        expected_axes = ("t", "x") if self.dimension == 1 else ("t", "y", "x")
        shape = data.fields.shape[1:]
        for axis, size in zip(expected_axes, shape):
            if axis not in data.axes or len(data.axes[axis]) != size:
                raise ValueError(f"{self.name}: axis {axis!r} does not match field shape {data.fields.shape}")
        if not np.all(np.isfinite(data.fields)):
            raise ValueError(f"{self.name}: field contains NaN or infinity")

    def evaluate(self, prediction: np.ndarray, target: np.ndarray, data: CaseData) -> dict[str, float]:
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.shape != target.shape or prediction.shape != data.fields.shape:
            raise ValueError(f"metric shapes must all match: {prediction.shape}, {target.shape}, {data.fields.shape}")
        diff = prediction - target
        denom = np.linalg.norm(target.reshape(target.shape[0], -1), axis=1).clip(min=1e-12)
        relative_l2 = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1) / denom
        return {
            "relative_l2_mean": float(np.mean(relative_l2)),
            "relative_l2_std": float(np.std(relative_l2)),
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "absolute_linf": float(np.max(np.abs(diff))),
        }


def residual_statistics(residual: np.ndarray) -> dict[str, float]:
    residual = np.asarray(residual)
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return {"pde_residual_mae": float("nan"), "pde_residual_rmse": float("nan")}
    return {
        "pde_residual_mae": float(np.mean(np.abs(finite))),
        "pde_residual_rmse": float(np.sqrt(np.mean(finite**2))),
    }


_REGISTRY: dict[str, CaseAdapter] = {}


def register_case(adapter: CaseAdapter) -> CaseAdapter:
    _REGISTRY[adapter.name] = adapter
    return adapter


def get_case_adapter(name: str) -> CaseAdapter:
    if name not in _REGISTRY:
        from . import linear_1d, radiative_1d, radiative_2d, radiative_initial_2d  # noqa: F401
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown case adapter {name!r}; expected {tuple(_REGISTRY)}") from exc
