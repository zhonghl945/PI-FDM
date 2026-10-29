"""Statistics for deterministic operators and stochastic diffusion models."""

from __future__ import annotations

import numpy as np

from pifdm.cases.base import CaseData


def evaluate_draws(adapter, draws: np.ndarray, data: CaseData):
    """Evaluate reconstruction errors for ``(draw, case, ...)`` predictions.

    PDE and boundary residuals require the continuous decoder and are computed
    separately by the automatic-differentiation diagnostics.
    """
    del adapter
    draws = np.asarray(draws)
    target = np.asarray(data.fields)
    if draws.ndim != target.ndim + 1 or draws.shape[1:] != target.shape:
        raise ValueError(f"draws {draws.shape} must be (N,{','.join(map(str, target.shape))})")
    denominator = np.linalg.norm(target.reshape(target.shape[0], -1), axis=1).clip(min=1e-12)
    relative_l2 = np.linalg.norm((draws - target[None]).reshape(draws.shape[0], draws.shape[1], -1), axis=2)
    relative_l2 = relative_l2 / denominator[None]
    rows = []
    for case_index in range(target.shape[0]):
        values = relative_l2[:, case_index]
        best = int(np.argmin(values))
        rows.append({
            "case": case_index, "draws": draws.shape[0], "best_draw": best,
            "relative_l2_mean": float(values.mean()), "relative_l2_std": float(values.std()),
            "relative_l2_min": float(values.min()),
        })
    best_indices = relative_l2.argmin(axis=0)
    best = draws[best_indices, np.arange(draws.shape[1])]
    summary = {
        "cases": int(target.shape[0]), "draws_per_condition": int(draws.shape[0]),
        "relative_l2_mean": float(relative_l2.mean()), "relative_l2_std": float(relative_l2.std()),
        "relative_l2_min": float(relative_l2.min()),
    }
    return summary, rows, best, relative_l2
