"""二维非线性、多介质辐射扩散：传统数值数据生成主文件。"""

from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from pifdm.workflows.generation import generate_dataset


CASE_CONFIG = {
    "case": "radiative_2d",
    "seed": 42,
    "domain": {"t": [0.0, 1.2], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    "generation": {
        "samples": 100,
        # Match the archived boundary-inflow experiment exactly at the data
        # level before comparing the newer model/training implementations.
        "nx": 128, "ny": 128, "nt": 1000,
        "nx_save": 64, "ny_save": 64, "nt_save": 16,
        "initial_value": 1e-3,
        "boundary_mode": "sustained",
        "seed_mode": "legacy_indexed", "sample_seed": 2026,
        "legacy_boundary_formula": True,
        "limiter_norm": "l1",
        "workers": 1,
        "show_progress": True,
        "solver": {
            # These are the archived solver settings and boundary-face
            # closure, not the newer self-consistent Robin-face path.
            "linear_solver": "sor", "picard_tolerance": 1e-6,
            "linear_tolerance": 1e-7, "linear_absolute_tolerance": 1e-12,
            "picard_iterations": 2000, "linear_iterations": 2000,
            "omega": 1.9, "legacy_boundary_extrapolation": True,
        },
    },
    # The initial E and Robin g remain frozen known conditions.  Robin g is
    # the boundary right-hand side rather than a boundary value of E, so it
    # is used by the boundary constraint but is not an E reconstruction target.
    "reconstruction_loss": {
        "include_initial_condition": True,
        "include_robin_boundary": False,
    },
    # 与训练集同一随机边界分布、不同随机种子：当前为同分布未见样本测试。
    "test_generation": {
        "samples": 20, "seed": 3000,
        "sample_seed": 3000,
    },
    "physics": {
        "nonnegative_solution": False,
        "use_flux_limiter": True,
        # Keep the current training/PINN residual definition unchanged.  The
        # archived L1 limiter is selected only in generation.limiter_norm.
        "limiter_norm": "l2_smooth",
        "source": {"type": "constant", "value": 0.0},
        "material": {
            "type": "rectangles", "background": 1.0,
            "regions": [
                {"x": [0.4, 0.6], "y": [0.0, 0.2], "value": 10.0},
                {"x": [0.4, 0.6], "y": [0.8, 1.0], "value": 10.0},
            ],
        },
    },
}

RUN_SETTINGS = {
    "splits": ("train", "test"),
    "train_output": CASE_DIR / f"data/training_{CASE_CONFIG['generation']['samples']}.h5",
    "test_output": CASE_DIR / f"data/test_{CASE_CONFIG['test_generation']['samples']}.h5",
    "visualization_root": CASE_DIR / "data/visualizations",
    "visualization_samples": 3,  # 设为 0 可关闭数据集样本图
}


def main():
    for split in RUN_SETTINGS["splits"]:
        generate_dataset(
            CASE_CONFIG, split=split, output=RUN_SETTINGS[f"{split}_output"],
            visualization_dir=RUN_SETTINGS["visualization_root"] / split,
            visualization_samples=RUN_SETTINGS["visualization_samples"],
        )


if __name__ == "__main__": main()
