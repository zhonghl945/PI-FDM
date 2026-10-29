"""一维非线性辐射扩散：传统数值数据生成主文件。"""

from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from pifdm.workflows.generation import generate_dataset


CASE_CONFIG = {
    "case": "radiative_1d",
    "seed": 42,
    "domain": {"t": [0.0, 1.0], "x": [0.0, 1.0]},
    "generation": {
        "samples": 1000, "nx": 512, "nt": 5120, "nx_save": 126, "nt_save": 64,
        "initial_value": 1e-3, "right_boundary": 0.0, "workers": 32,
        "show_progress": True,
    },
    # 与训练集同一随机边界分布、不同随机种子；真正分布外测试在 test.py 的 ood 模式。
    "test_generation": {"samples": 20, "seed": 4242},
    "physics": {
        "nonnegative_solution": True,
        "use_flux_limiter": True,
        "limiter_norm": "l2_smooth",       # l2_smooth / l1
        "source": {"type": "constant", "value": 0.0},
        "material": {"type": "constant", "value": 1.0},
    },
}

RUN_SETTINGS = {
    "splits": ("train", "test"),  # ("train",) 或 (,"test")
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
