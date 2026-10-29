"""二维随机高斯初值、多介质辐射扩散：传统数值数据生成主文件。"""

from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pifdm.workflows.generation import generate_dataset


CASE_CONFIG = {
    "case": "radiative_initial_2d",
    "seed": 42,
    "domain": {"t": [0.0, 1.5], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    "generation": {
        "samples": 100,
        "nx": 256, "ny": 256, "nt": 5120,
        "nx_save": 62, "ny_save": 62, "nt_save": 32,
        # E0 = background + A exp(-r^2/(2 sigma^2)); all ranges are uniform.
        "initial_condition": {
            "type": "gaussian_hotspot", "background": 1e-3,
            "amplitude": [150.0, 300.0], "sigma": [0.04, 0.09],
            "center_x": [0.00, 0.15], "center_y": [0.00, 0.15],
        },
        # Four saved values are the zero right-hand sides of the Robin/Marshak operators.
        "boundary_mode": "zero",
        "workers": 1,
        "show_progress": True,
        # In a non-interactive Slurm log, print one heartbeat line at this interval.
        "progress_interval": 30.0,
        "solver": {
            # PCG uses a true relative residual; set linear_solver="sor" to compare/fallback.
            "linear_solver": "pcg", "picard_tolerance": 1e-6,
            "linear_tolerance": 1e-6, "linear_absolute_tolerance": 1e-12,
            "picard_iterations": 2000, "linear_iterations": 500,
        },
    },
    # Same hotspot distribution, independent seed: unseen ID initial conditions.
    "test_generation": {"samples": 20, "seed": 4242},
    "physics": {
        "nonnegative_solution": True,
        "use_flux_limiter": True,
        # The radiative flux limiter uses the Euclidean magnitude |grad E|.
        # Its smooth form also avoids Picard cycling around sharp hotspots.
        "limiter_norm": "l2_smooth",
        "source": {"type": "constant", "value": 0.0},
        "material": {
            "type": "rectangles", "background": 1.0,
            "regions": [
                # Squares of side 0.20 centered at (0.25,0.25) and (0.75,0.75).
                {"x": [0.15, 0.35], "y": [0.15, 0.35], "value": 2.5},
                {"x": [0.60, 0.80], "y": [0.60, 0.80], "value": 5.0},
            ],
        },
    },
}

RUN_SETTINGS = {
    "splits": ("train", "test"),
    "train_output": CASE_DIR / f"data/training_{CASE_CONFIG['generation']['samples']}.h5",
    "test_output": CASE_DIR / f"data/test_{CASE_CONFIG['test_generation']['samples']}.h5",
    "visualization_root": CASE_DIR / "data/visualizations",
    "visualization_samples": 3,
}


def main():
    for split in RUN_SETTINGS["splits"]:
        generate_dataset(
            CASE_CONFIG, split=split, output=RUN_SETTINGS[f"{split}_output"],
            visualization_dir=RUN_SETTINGS["visualization_root"] / split,
            visualization_samples=RUN_SETTINGS["visualization_samples"],
        )


if __name__ == "__main__":
    main()
