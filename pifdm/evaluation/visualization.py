"""Optional publication-neutral prediction diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("visualization requires matplotlib (install pifdm[data])") from exc
    return plt


def plot_dataset_sample(data, index, output, *, title="generated reference solution"):
    """Visualize one generated HDF5 sample without changing the saved data."""
    plt = _pyplot()
    field = np.asarray(data.fields[index])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if field.ndim == 2:
        fig, panels = plt.subplots(1, 2, figsize=(11, 4))
        extent = [data.axes["x"][0], data.axes["x"][-1],
                  data.axes["t"][0], data.axes["t"][-1]]
        image = panels[0].imshow(field, origin="lower", aspect="auto", extent=extent, cmap="magma")
        panels[0].set(title="Interior solution", xlabel="x", ylabel="t")
        fig.colorbar(image, ax=panels[0])
        for name, label in (("bL", "left condition"), ("bR", "right condition")):
            if name in data.boundaries:
                panels[1].plot(data.axes["t"], data.boundaries[name][index], label=label)
        panels[1].set(title="Saved boundary conditions", xlabel="t", ylabel="condition value")
        panels[1].legend()
        panels[1].grid(alpha=0.25)
    elif field.ndim == 3:
        indices = np.linspace(0, field.shape[0] - 1, min(3, field.shape[0]), dtype=int)
        fig, panels = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4), squeeze=False)
        extent = [data.axes["x"][0], data.axes["x"][-1],
                  data.axes["y"][0], data.axes["y"][-1]]
        for panel, time_index in zip(panels[0], indices):
            image = panel.imshow(field[time_index], origin="lower", extent=extent, cmap="magma")
            panel.set(title=f"t={data.axes['t'][time_index]:.3g}", xlabel="x", ylabel="y")
            fig.colorbar(image, ax=panel)
    else:
        raise ValueError("expected a 1D or 2D space-time sample")
    fig.suptitle(f"{title} #{index}")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prediction(target, prediction, axes, output, *, title="prediction"):
    plt = _pyplot()
    target, prediction = np.asarray(target), np.asarray(prediction)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if target.ndim == 2:
        fig, panels = plt.subplots(1, 3, figsize=(14, 4))
        extent = [axes["x"][0], axes["x"][-1], axes["t"][0], axes["t"][-1]]
        for panel, values, name in zip(panels, (target, prediction, np.abs(prediction - target)),
                                       ("Reference", "Prediction", "Absolute error")):
            image = panel.imshow(values, origin="lower", aspect="auto", extent=extent, cmap="magma")
            panel.set_title(name); panel.set_xlabel("x"); panel.set_ylabel("t")
            fig.colorbar(image, ax=panel)
    elif target.ndim == 3:
        indices = np.linspace(0, target.shape[0] - 1, min(3, target.shape[0]), dtype=int)
        fig, panels = plt.subplots(len(indices), 3, figsize=(12, 3.5 * len(indices)), squeeze=False)
        extent = [axes["x"][0], axes["x"][-1], axes["y"][0], axes["y"][-1]]
        for row, index in enumerate(indices):
            for panel, values, name in zip(panels[row],
                    (target[index], prediction[index], np.abs(prediction[index] - target[index])),
                    ("Reference", "Prediction", "Absolute error")):
                image = panel.imshow(values, origin="lower", extent=extent, cmap="magma")
                panel.set_title(f"{name}, t={axes['t'][index]:.3g}")
                fig.colorbar(image, ax=panel)
    else:
        raise ValueError("expected a 1D space-time or 2D space-time field")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_radiative_1d_prediction(target, prediction, axes, output, *, title="prediction"):
    """Plot a smooth space-time comparison for the radiative 1D case.

    The full ``(t, x)`` field is shown together with representative spatial
    profiles of the prediction.  The interpolation, contour overlay, and
    profile selection are display-only and do not alter the prediction used
    for metrics.
    """
    plt = _pyplot()
    target, prediction = np.asarray(target), np.asarray(prediction)
    time = np.asarray(axes["t"], dtype=float)
    x = np.asarray(axes["x"], dtype=float)
    expected_shape = (len(time), len(x))
    if target.shape != expected_shape or prediction.shape != expected_shape:
        raise ValueError(
            "radiative_1d prediction plots require target and prediction shaped "
            f"{expected_shape}"
        )
    if time.ndim != 1 or x.ndim != 1 or len(time) < 2 or len(x) < 2:
        raise ValueError("radiative_1d prediction plots require two non-empty axes")
    if not np.all(np.diff(time) > 0.0) or not np.all(np.diff(x) > 0.0):
        raise ValueError("radiative_1d prediction axes must be strictly increasing")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    error = np.abs(prediction - target)
    energy_values = np.concatenate((target.reshape(-1), prediction.reshape(-1)))
    finite_energy = energy_values[np.isfinite(energy_values)]
    finite_error = error[np.isfinite(error)]
    energy_min = float(finite_energy.min()) if finite_energy.size else 0.0
    energy_max = float(finite_energy.max()) if finite_energy.size else 1.0
    error_max = float(finite_error.max()) if finite_error.size else 1.0
    if energy_max <= energy_min:
        energy_max = energy_min + np.finfo(float).eps
    error_max = max(error_max, np.finfo(float).eps)

    def outer_edges(axis):
        return np.concatenate((
            [axis[0] - 0.5 * (axis[1] - axis[0])],
            0.5 * (axis[:-1] + axis[1:]),
            [axis[-1] + 0.5 * (axis[-1] - axis[-2])],
        ))

    # imshow with bicubic interpolation gives the same smooth, publication
    # oriented appearance as the 2D prediction plots.  The extent uses the
    # outer cell edges so the plotted samples remain centered on their axes.
    extent = [
        float(outer_edges(x)[0]), float(outer_edges(x)[-1]),
        float(outer_edges(time)[0]), float(outer_edges(time)[-1]),
    ]
    contour_levels = np.linspace(energy_min, energy_max, 11)[1:-1]
    profile_fractions = (0.25, 0.50, 0.75, 1.00)
    profile_indices = [
        int(np.abs(time - (time[0] + fraction * (time[-1] - time[0]))).argmin())
        for fraction in profile_fractions
    ]
    figure, panels = plt.subplots(
        1, 4, figsize=(19.0, 4.8), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.0, 1.05)},
    )
    energy_image = None
    for panel, values, label in (
        (panels[0], target, "Reference"),
        (panels[1], prediction, "Prediction"),
    ):
        energy_image = panel.imshow(
            values, origin="lower", extent=extent, aspect="auto",
            cmap="inferno", vmin=energy_min, vmax=energy_max,
            interpolation="bicubic",
        )
        if contour_levels.size:
            panel.contour(
                x, time, values, levels=contour_levels,
                colors="#f6e8c3", linewidths=0.55, alpha=0.72,
            )
        panel.set_title(label, fontsize=12, fontweight="bold")
        panel.set_xlabel("x")
        panel.set_ylabel("t")

    error_image = panels[2].imshow(
        error, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=0.0, vmax=error_max,
        interpolation="bicubic",
    )
    panels[2].set_title("Absolute error", fontsize=12, fontweight="bold")
    panels[2].set_xlabel("x")
    panels[2].set_ylabel("t")

    profile_panel = panels[3]
    profile_colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.92, len(profile_indices))
    )
    for color, index in zip(profile_colors, profile_indices):
        profile_panel.plot(
            x, target[index], color=color, linewidth=1.8,
            linestyle="-", label=f"Reference, t = {time[index]:.2f}",
        )
        profile_panel.plot(
            x, prediction[index], color=color, linewidth=1.8,
            linestyle="--", label=f"Prediction, t = {time[index]:.2f}",
        )
    profile_panel.set_title("Slices", fontsize=12, fontweight="bold")
    profile_panel.set_xlabel("x")
    profile_panel.set_ylabel("E")
    profile_panel.set_xlim(float(x[0]), float(x[-1]))
    profile_panel.set_ylim(energy_min, energy_max)
    profile_panel.grid(color="#d8d8d8", linewidth=0.65, alpha=0.8)
    profile_panel.legend(
        loc="best", fontsize=7.8, frameon=False, ncol=2,
        columnspacing=0.8, handlelength=2.0,
    )

    for panel in panels:
        panel.tick_params(labelsize=9)
    for panel in panels[:3]:
        panel.grid(color="white", linewidth=0.45, alpha=0.22)

    figure.colorbar(
        energy_image, ax=panels[:2], shrink=0.90, pad=0.018,
        label="Energy E",
    )
    figure.colorbar(
        error_image, ax=panels[2], shrink=0.90, pad=0.018,
        label="Absolute error |Prediction - Reference|",
    )
    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def _horizontal_slice(values, y, location: float) -> np.ndarray:
    """Linearly interpolate a field slice at a physical y-coordinate."""
    y = np.asarray(y, dtype=float)
    if location <= y[0]:
        return np.asarray(values[0])
    if location >= y[-1]:
        return np.asarray(values[-1])
    upper = int(np.searchsorted(y, location, side="right"))
    lower = upper - 1
    fraction = (location - y[lower]) / (y[upper] - y[lower])
    return (1.0 - fraction) * np.asarray(values[lower]) + fraction * np.asarray(values[upper])


def plot_radiative_2d_prediction(target, prediction, axes, output, *, title="prediction",
                                 slice_y: float = 0.5,
                                 energy_scale: str = "global",
                                 first_time: Optional[float] = None):
    """Plot three non-initial 2D snapshots with contours and centreline slices."""
    plt = _pyplot()
    target, prediction = np.asarray(target), np.asarray(prediction)
    time = np.asarray(axes["t"], dtype=float)
    y, x = (np.asarray(axes[name], dtype=float) for name in ("y", "x"))
    expected_shape = (len(time), len(y), len(x))
    if target.shape != expected_shape or prediction.shape != expected_shape:
        raise ValueError(
            "radiative_2d prediction plots require target and prediction shaped "
            f"{expected_shape}"
        )
    if len(time) < 4:
        raise ValueError("radiative_2d prediction plots require at least four saved time slices")
    if not y[0] <= slice_y <= y[-1]:
        raise ValueError(f"slice_y={slice_y} lies outside the saved y axis")
    if energy_scale not in ("global", "per_time"):
        raise ValueError("energy_scale must be 'global' or 'per_time'")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Skip t=0: its frozen known field does not evaluate the model prediction.
    indices = np.linspace(1, len(time) - 1, 3, dtype=int)
    if first_time is not None:
        # Use the closest saved time instead of interpolating a new snapshot.
        first_index = int(np.argmin(np.abs(time - float(first_time))))
        first_index = max(1, min(first_index, len(time) - 1))
        indices[0] = first_index
        if len(np.unique(indices)) != len(indices):
            raise ValueError("first_time collides with another selected visualization time")
    selected_target = target[indices]
    selected_prediction = prediction[indices]
    selected_error = np.abs(selected_prediction - selected_target)
    energy_min = energy_max = None
    if energy_scale == "global":
        energy_values = np.concatenate((selected_target.reshape(-1), selected_prediction.reshape(-1)))
        finite_energy = energy_values[np.isfinite(energy_values)]
        energy_min = float(finite_energy.min()) if finite_energy.size else 0.0
        energy_max = float(finite_energy.max()) if finite_energy.size else 1.0
        if energy_max <= energy_min:
            energy_max = energy_min + np.finfo(float).eps
    error_values = selected_error[np.isfinite(selected_error)]
    error_max = float(error_values.max()) if error_values.size else 1.0
    error_max = max(error_max, np.finfo(float).eps)
    contour_levels = (
        np.linspace(energy_min, energy_max, 11)[1:-1]
        if energy_scale == "global" else None
    )
    extent = image_extent_from_centers(x, y)

    figure, panels = plt.subplots(3, 4, figsize=(16.8, 10.8), constrained_layout=True)
    energy_image = error_image = None
    for row, index in enumerate(indices):
        time_label = f"t = {time[index]:.3g}"
        reference_panel, prediction_panel, error_panel, slice_panel = panels[row]
        if energy_scale == "per_time":
            row_energy = np.concatenate((target[index].reshape(-1), prediction[index].reshape(-1)))
            finite_row_energy = row_energy[np.isfinite(row_energy)]
            row_energy_min = float(finite_row_energy.min()) if finite_row_energy.size else 0.0
            row_energy_max = float(finite_row_energy.max()) if finite_row_energy.size else 1.0
            if row_energy_max <= row_energy_min:
                row_energy_max = row_energy_min + np.finfo(float).eps
            row_contour_levels = np.linspace(row_energy_min, row_energy_max, 11)[1:-1]
        else:
            row_energy_min = energy_min
            row_energy_max = energy_max
            row_contour_levels = contour_levels
        row_error_values = selected_error[row][np.isfinite(selected_error[row])]
        row_error_max = (
            max(float(row_error_values.max()), np.finfo(float).eps)
            if row_error_values.size else 1.0
        )
        for panel, values, label in (
            (reference_panel, target[index], "Reference"),
            (prediction_panel, prediction[index], "Prediction"),
        ):
            energy_image = panel.imshow(
                values, origin="lower", extent=extent, aspect="equal", cmap="inferno",
                vmin=row_energy_min, vmax=row_energy_max, interpolation="bicubic",
            )
            if row_contour_levels.size:
                panel.contour(
                    x, y, values, levels=row_contour_levels, colors="#f6e8c3",
                    linewidths=0.55, alpha=0.72,
                )
            panel.set_title(f"{label} ({time_label})")
            panel.set_xlabel("x")
            panel.set_ylabel("y")

        error_image = error_panel.imshow(
            selected_error[row], origin="lower", extent=extent, aspect="equal", cmap="viridis",
            vmin=0.0, vmax=row_error_max if energy_scale == "per_time" else error_max,
            interpolation="bicubic",
        )
        error_panel.set_title(f"Absolute error ({time_label})")
        error_panel.set_xlabel("x")
        error_panel.set_ylabel("y")

        slice_panel.plot(
            x, _horizontal_slice(target[index], y, slice_y), color="#1f4e79",
            linewidth=1.7, label="Reference",
        )
        slice_panel.plot(
            x, _horizontal_slice(prediction[index], y, slice_y), color="#c43c39",
            linewidth=1.45, linestyle="--", label="Prediction",
        )
        slice_panel.set_title(f"Slice at y = {slice_y:g} ({time_label})")
        slice_panel.set_xlabel("x")
        slice_panel.set_ylabel("E")
        slice_panel.set_xlim(float(x[0]), float(x[-1]))
        slice_panel.grid(alpha=0.24, linewidth=0.6)
        slice_panel.legend(loc="best", fontsize=8, frameon=False)

        if energy_scale == "per_time":
            figure.colorbar(
                energy_image, ax=[reference_panel, prediction_panel],
                shrink=0.90, pad=0.018, label="Energy E",
            )
            figure.colorbar(
                error_image, ax=error_panel,
                shrink=0.90, pad=0.018, label="Absolute error",
            )

    if energy_scale == "global":
        figure.colorbar(
            energy_image, ax=panels[:, :2].ravel().tolist(), shrink=0.90, pad=0.018,
            label="Energy E",
        )
    if energy_scale == "global":
        figure.colorbar(
            error_image, ax=panels[:, 2].tolist(), shrink=0.90, pad=0.018,
            label="Absolute error |Prediction - Reference|",
        )
    figure.suptitle(title, fontsize=13)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _symmetric_limit(values) -> float:
    values = np.asarray(values)
    finite = np.abs(values[np.isfinite(values)])
    return max(float(finite.max()) if finite.size else 0.0, np.finfo(float).eps)


def _robust_symmetric_scale(values, *, clip_quantile: float = 0.99,
                            linear_quantile: float = 0.90) -> tuple[float, float]:
    """Return a shared clipped range and its symmetric-log linear threshold."""
    if not 0.0 < linear_quantile < clip_quantile < 1.0:
        raise ValueError("expected 0 < linear_quantile < clip_quantile < 1")
    # Convert before taking absolute values or quantiles.  Autodiff residuals
    # can approach float32's upper range, where float32 interpolation inside
    # np.quantile may overflow despite all samples being finite.
    values = np.asarray(values, dtype=np.float64)
    finite = np.abs(values[np.isfinite(values)])
    epsilon = np.finfo(float).eps
    if not finite.size:
        return epsilon, epsilon
    limit = max(float(np.quantile(finite, clip_quantile)), epsilon)
    linear = max(float(np.quantile(finite, linear_quantile)), epsilon)
    return limit, min(linear, limit)


def _robust_positive_scale(values, *, clip_quantile: float = 0.99,
                           floor_quantile: float = 0.05) -> tuple[float, float]:
    """Return logarithmic lower and upper limits for a nonnegative residual map."""
    if not 0.0 < floor_quantile < clip_quantile < 1.0:
        raise ValueError("expected 0 < floor_quantile < clip_quantile < 1")
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values) & (values > 0.0)]
    epsilon = np.finfo(float).eps
    if not finite.size:
        return epsilon, 1.0
    lower = max(float(np.quantile(finite, floor_quantile)), epsilon)
    upper = max(float(np.quantile(finite, clip_quantile)), np.nextafter(lower, np.inf))
    return lower, upper


def image_extent_from_centers(x, y) -> list[float]:
    """Return an imshow extent whose pixel centers coincide with sampled axes."""
    x, y = (np.asarray(values, dtype=np.float64) for values in (x, y))
    if x.ndim != 1 or y.ndim != 1 or not len(x) or not len(y):
        raise ValueError("image axes must be nonempty one-dimensional arrays")

    def outer_edges(axis):
        if len(axis) == 1:
            return float(axis[0] - 0.5), float(axis[0] + 0.5)
        if not np.all(np.diff(axis) > 0.0):
            raise ValueError("image axes must be strictly increasing")
        return (
            float(axis[0] - 0.5 * (axis[1] - axis[0])),
            float(axis[-1] + 0.5 * (axis[-1] - axis[-2])),
        )

    x0, x1 = outer_edges(x)
    y0, y1 = outer_edges(y)
    return [x0, x1, y0, y1]


def _residual_panel_layout(time_count: int) -> tuple[int, int, tuple[float, float]]:
    if time_count < 1:
        raise ValueError("residual plots require at least one time slice")
    rows = (time_count + 4) // 5
    columns = (time_count + rows - 1) // rows
    return rows, columns, (max(4.0, 3.6 * columns), 3.4 * rows + 0.7)


def plot_pde_residual_panels(residual, time, y, x, output, *, title="PDE residual",
                             material_rectangles=(), clip_quantile: float = 0.99):
    """Plot signed PDE residual maps on one robust shared color scale."""
    plt = _pyplot()
    from matplotlib.colors import SymLogNorm
    from matplotlib.patches import Rectangle

    residual = np.asarray(residual)
    time, y, x = (np.asarray(values) for values in (time, y, x))
    if residual.shape != (len(time), len(y), len(x)):
        raise ValueError("PDE residual shape must be (time, y, x)")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    limit, linear_threshold = _robust_symmetric_scale(residual, clip_quantile=clip_quantile)
    color_map = plt.get_cmap("coolwarm").copy()
    color_map.set_over("#9e1b1b")
    color_map.set_under("#173d77")
    color_norm = SymLogNorm(
        linthresh=linear_threshold, linscale=0.9,
        vmin=-limit, vmax=limit, clip=False,
    )
    rows, columns, figure_size = _residual_panel_layout(len(time))
    figure, panels = plt.subplots(
        rows, columns, figsize=figure_size, constrained_layout=True, squeeze=False
    )
    active_panels = panels.flat[:len(time)]
    for panel in panels.flat[len(time):]:
        panel.set_visible(False)
    extent = image_extent_from_centers(x, y)
    image = None
    plot_values = np.ma.masked_invalid(residual)
    for index, panel in enumerate(active_panels):
        image = panel.imshow(
            plot_values[index], origin="lower", extent=extent, aspect="equal",
            cmap=color_map, norm=color_norm,
        )
        for x0, x1, y0, y1 in material_rectangles:
            panel.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False,
                edgecolor="#202020", linewidth=0.8, linestyle=(0, (3, 2)), zorder=2,
            ))
        panel.set_title(f"t={time[index]:.4g}")
        if index >= columns * (rows - 1):
            panel.set_xlabel("x")
        if index % columns == 0:
            panel.set_ylabel("y")
    figure.colorbar(
        image, ax=list(active_panels), shrink=0.86, pad=0.02, extend="both",
        label=f"PDE residual (symlog; clipped at |r|={limit:.2e})",
    )
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_pde_residual_magnitude_panels(residual, time, y, x, output, *,
                                       title="Mean absolute PDE residual",
                                       material_rectangles=(), clip_quantile: float = 0.99):
    """Plot nonnegative residual maps with a shared robust logarithmic scale."""
    plt = _pyplot()
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    residual = np.asarray(residual)
    time, y, x = (np.asarray(values) for values in (time, y, x))
    if residual.shape != (len(time), len(y), len(x)):
        raise ValueError("PDE residual shape must be (time, y, x)")
    if np.any(residual < 0.0):
        raise ValueError("PDE residual magnitude must be nonnegative")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lower, upper = _robust_positive_scale(residual, clip_quantile=clip_quantile)
    color_map = plt.get_cmap("magma").copy()
    color_map.set_over("#fcfdbf")
    color_norm = LogNorm(vmin=lower, vmax=upper, clip=False)
    rows, columns, figure_size = _residual_panel_layout(len(time))
    figure, panels = plt.subplots(
        rows, columns, figsize=figure_size, constrained_layout=True, squeeze=False
    )
    active_panels = panels.flat[:len(time)]
    for panel in panels.flat[len(time):]:
        panel.set_visible(False)
    extent = image_extent_from_centers(x, y)
    image = None
    plot_values = np.ma.masked_invalid(residual)
    for index, panel in enumerate(active_panels):
        image = panel.imshow(
            plot_values[index], origin="lower", extent=extent, aspect="equal",
            cmap=color_map, norm=color_norm,
        )
        for x0, x1, y0, y1 in material_rectangles:
            panel.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False,
                edgecolor="#e8e8e8", linewidth=0.8, linestyle=(0, (3, 2)), zorder=2,
            ))
        panel.set_title(f"t={time[index]:.4g}")
        if index >= columns * (rows - 1):
            panel.set_xlabel("x")
        if index % columns == 0:
            panel.set_ylabel("y")
    figure.colorbar(
        image, ax=list(active_panels), shrink=0.86, pad=0.02, extend="max",
        label=f"Mean |PDE residual| (log; clipped above {upper:.2e})",
    )
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_pde_residual_1d(residual, time, x, output, *, title="PDE residual",
                         clip_quantile: float = 0.99):
    """Plot a signed 1D space-time residual using one shared color scale."""
    plt = _pyplot()
    from matplotlib.colors import SymLogNorm

    residual = np.asarray(residual)
    time, x = (np.asarray(values) for values in (time, x))
    if residual.shape != (len(time), len(x)):
        raise ValueError("1D PDE residual shape must be (time, x)")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    limit, linear_threshold = _robust_symmetric_scale(
        residual, clip_quantile=clip_quantile
    )
    color_map = plt.get_cmap("coolwarm").copy()
    color_map.set_over("#9e1b1b")
    color_map.set_under("#173d77")
    norm = SymLogNorm(
        linthresh=linear_threshold, linscale=0.9,
        vmin=-limit, vmax=limit, clip=False,
    )
    figure, panel = plt.subplots(1, 1, figsize=(8.2, 4.6), constrained_layout=True)
    image = panel.imshow(
        np.ma.masked_invalid(residual), origin="lower",
        extent=image_extent_from_centers(x, time), aspect="auto",
        cmap=color_map, norm=norm,
    )
    panel.set_xlabel("x")
    panel.set_ylabel("t")
    panel.set_title(title)
    figure.colorbar(
        image, ax=panel, pad=0.02, extend="both",
        label=f"Signed PDE residual (symlog; clipped at |r|={limit:.2e})",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_pde_residual_magnitude_1d(residual, time, x, output, *,
                                   title="Mean absolute PDE residual",
                                   clip_quantile: float = 0.99):
    """Plot a nonnegative 1D space-time residual on a robust log scale."""
    plt = _pyplot()
    from matplotlib.colors import LogNorm

    residual = np.asarray(residual)
    time, x = (np.asarray(values) for values in (time, x))
    if residual.shape != (len(time), len(x)):
        raise ValueError("1D PDE residual shape must be (time, x)")
    if np.any(residual < 0.0):
        raise ValueError("PDE residual magnitude must be nonnegative")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lower, upper = _robust_positive_scale(residual, clip_quantile=clip_quantile)
    color_map = plt.get_cmap("magma").copy()
    color_map.set_over("#fcfdbf")
    norm = LogNorm(vmin=lower, vmax=upper, clip=False)
    figure, panel = plt.subplots(1, 1, figsize=(8.2, 4.6), constrained_layout=True)
    image = panel.imshow(
        np.ma.masked_invalid(residual), origin="lower",
        extent=image_extent_from_centers(x, time), aspect="auto",
        cmap=color_map, norm=norm,
    )
    panel.set_xlabel("x")
    panel.set_ylabel("t")
    panel.set_title(title)
    figure.colorbar(
        image, ax=panel, pad=0.02, extend="max",
        label=f"Mean |PDE residual| (log; clipped above {upper:.2e})",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_robin_residual_1d(residual, time, output, *, side, title=None):
    """Plot the signed Robin residual trace at a 1D boundary."""
    plt = _pyplot()
    residual = np.asarray(residual)
    time = np.asarray(time)
    if residual.ndim == 2 and residual.shape[1] == 1:
        residual = residual[:, 0]
    if residual.shape != (len(time),):
        raise ValueError("1D Robin residual shape must be (time,) or (time, 1)")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    limit = _symmetric_limit(residual)
    figure, panel = plt.subplots(1, 1, figsize=(8.2, 3.8), constrained_layout=True)
    panel.plot(time, residual, color="#b13d3d", linewidth=1.4)
    panel.axhline(0.0, color="#333333", linewidth=0.8)
    panel.set_ylim(-limit, limit)
    panel.set_xlabel("t")
    panel.set_ylabel("Robin residual")
    panel.set_title(title or f"Robin residual: {side} boundary")
    panel.grid(alpha=0.25)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_robin_residual_panels(residual, time, tangent, output, *, side, title=None):
    """Plot boundary-residual profiles along one physical boundary side."""
    plt = _pyplot()
    residual = np.asarray(residual)
    time, tangent = (np.asarray(values) for values in (time, tangent))
    if residual.shape != (len(time), len(tangent)):
        raise ValueError("Robin residual shape must be (time, tangent)")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    limit = _symmetric_limit(residual)
    tangent_label = "y" if side in ("left", "right") else "x"
    rows, columns, figure_size = _residual_panel_layout(len(time))
    figure, panels = plt.subplots(
        rows, columns, figsize=figure_size, constrained_layout=True, squeeze=False
    )
    active_panels = panels.flat[:len(time)]
    for panel in panels.flat[len(time):]:
        panel.set_visible(False)
    for index, panel in enumerate(active_panels):
        panel.plot(tangent, residual[index], color="#b13d3d", linewidth=1.25)
        panel.axhline(0.0, color="#333333", linewidth=0.8)
        panel.set_ylim(-limit, limit)
        panel.set_title(f"t={time[index]:.4g}")
        panel.grid(alpha=0.25)
        if index >= columns * (rows - 1):
            panel.set_xlabel(tangent_label)
        if index % columns == 0:
            panel.set_ylabel("Robin residual")
    figure.suptitle(title or f"Robin residual: {side}")
    figure.savefig(output, dpi=180)
    plt.close(figure)
