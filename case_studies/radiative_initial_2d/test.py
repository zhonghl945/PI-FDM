"""二维随机高斯初值、多介质辐射扩散：测试与统计主文件。"""

from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_studies.radiative_initial_2d.generate import RUN_SETTINGS as DATA_PATHS
from case_studies.radiative_initial_2d.train import output_directory
from pifdm.experiments.contracts import DIFFUSION_METHODS
from pifdm.experiments.naming import validate_diffconphy_base_method

# fdm_unet / fdm_vit / c_fdm_unet / c_fdm_vit / pifdm_unet / pifdm_vit
# deeponet / pideeponet / fno / pino / diffusionpde / diffconphy
METHOD = "c_fdm_unet"
# Select a model trained with 100 / 1000 / ... samples.  Normalization and
# inverse normalization come from this checkpoint's saved training scalers.
TRAINING_SAMPLES = 100
DIFFCONPHY_SETTINGS = {
    # Select one of the six trained diffusion checkpoints above.
    "checkpoint_method": "fdm_vit",
    "guidance_weight": 1e-5,
    "guidance_huber_delta": 1.0,
}

CHECKPOINT_METHOD = (
    DIFFCONPHY_SETTINGS["checkpoint_method"] if METHOD == "diffconphy" else METHOD
)
if METHOD == "diffconphy":
    validate_diffconphy_base_method(CHECKPOINT_METHOD)
CHECKPOINT_RUN_DIR = output_directory(CHECKPOINT_METHOD, TRAINING_SAMPLES)
RUN_DIR = CHECKPOINT_RUN_DIR / "diffconphy" if METHOD == "diffconphy" else CHECKPOINT_RUN_DIR
CHECKPOINT = CHECKPOINT_RUN_DIR / "checkpoint.pt"

TEST_SETTINGS = {
    "mode": "id",  # id / ood / both；当前还没有该二维算例的 OOD 场景
    "training_samples": TRAINING_SAMPLES,
    "checkpoint": CHECKPOINT,
    "data": DATA_PATHS["test_output"],
    "output": RUN_DIR / "evaluation",
    "device": "auto",
    "batch_size": 10,
    "save_visualizations": True,
    "residual_diagnostics": {
        "enabled": True,
        "case_count": 3,
        "seed": 42,  # 只固定诊断用例选择，不固定扩散采样噪声
        "pde_time_points": 10,
        "pde_spatial_shape": [100, 100],
        "boundary_time_points": 10,
    },
    "ood_scenarios": (),
}

if METHOD in DIFFUSION_METHODS or METHOD == "diffconphy":
    TEST_SETTINGS.update({
        # sampler: "ddim" / "ddpm" / "auto"；DDPM 始终使用完整扩散步数。
        "sampler": "ddim", "sampling_timesteps": 50,
        "samples_per_condition": 10, "eta": 0.0,
    })
if METHOD == "diffconphy":
    TEST_SETTINGS.update({
        key: DIFFCONPHY_SETTINGS[key]
        for key in ("guidance_weight", "guidance_huber_delta")
    })


def main():
    from pifdm.workflows.evaluation import evaluate_checkpoint
    arguments = {
        "checkpoint_path": TEST_SETTINGS["checkpoint"], "data_path": TEST_SETTINGS["data"],
        "output_dir": TEST_SETTINGS["output"], "mode": TEST_SETTINGS["mode"],
        "device": TEST_SETTINGS["device"], "batch_size": TEST_SETTINGS["batch_size"],
        "scenarios": TEST_SETTINGS["ood_scenarios"],
        "save_visualizations": TEST_SETTINGS["save_visualizations"],
        "residual_diagnostics": TEST_SETTINGS["residual_diagnostics"],
        "expected_training_samples": TEST_SETTINGS["training_samples"],
    }
    for key in ("sampler", "sampling_timesteps", "samples_per_condition", "eta",
                "guidance_weight", "guidance_huber_delta"):
        if key in TEST_SETTINGS:
            arguments[key] = TEST_SETTINGS[key]
    evaluate_checkpoint(**arguments)


if __name__ == "__main__":
    main()
