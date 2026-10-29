"""一维非线性辐射扩散：所有可训练方法共用的训练主文件。"""

from copy import deepcopy
from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_studies.radiative_1d.generate import CASE_CONFIG, RUN_SETTINGS as DATA_PATHS

# fdm_unet / fdm_vit / c_fdm_unet / c_fdm_vit / pifdm_unet / pifdm_vit
# deeponet / pideeponet / fno / pino / diffusionpde
METHOD = "c_fdm_unet"

# 每种方法只列实际生效的参数，便于直接审查单个实验。
METHOD_SETTINGS = {
    "fdm_unet": {
        "method": "fdm_unet",
        "model": {"width": 64, "dim_mults": [1, 2, 4]},
        "diffusion": {
            "timesteps": 1000, "beta_schedule": "cosine",
        },
        "training": {
            "batch_size": 8, "steps": 100000, "learning_rate": 1e-4,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.99], "weight_decay": 0.0,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "ema_decay": 0.995, "ema_update_every": 10,
            "num_workers": 0, "log_every": 100, "save_every": 100000,
        },
    },
    "fdm_vit": {
        "method": "fdm_vit",
        "model": {
            "width": 192, "depth": 12, "heads": 6,
            "patch_size": [4, 4], "mlp_ratio": 4.0,
        },
        "diffusion": {
            "timesteps": 1000, "beta_schedule": "cosine",
        },
        "training": {
            "batch_size": 8, "steps": 100000, "learning_rate": 1e-4,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.99], "weight_decay": 0.00,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "ema_decay": 0.995, "ema_update_every": 10,
            "num_workers": 0, "log_every": 100, "save_every": 50000,
        },
    },
    "c_fdm_vit": {
        "method": "c_fdm_vit",
        "model": {
            "width": 192, "depth": 12, "heads": 6, "patch_size": [4, 4],
            "mlp_ratio": 4.0, "coordinate_projection_scale": 0.05,
            "coordinate_projection_seed": 0,
            "decoder_checkpoint": False,
        },
        "diffusion": {
            "timesteps": 1000, "beta_schedule": "cosine",
        },
        "training": {
            "batch_size": 8, "steps": 100000, "learning_rate": 1e-4,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.99], "weight_decay": 0.00,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "ema_decay": 0.995, "ema_update_every": 10,
            "num_workers": 0, "log_every": 100, "save_every": 100000,
        },
    },
    "c_fdm_unet": {
        "method": "c_fdm_unet",
        "model": {
            "width": 64, "dim_mults": [1, 2, 4],
            "coordinate_projection_scale": 0.05, "coordinate_projection_seed": 0,
            "decoder_checkpoint": False,
        },
        "diffusion": {
            "timesteps": 1000, "beta_schedule": "cosine",
        },
        "training": {
            "batch_size": 8, "steps": 300000, "learning_rate": 1e-4,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.99], "weight_decay": 0.00,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "ema_decay": 0.995, "ema_update_every": 10,
            "num_workers": 0, "log_every": 100, "save_every": 50000,
        },
    },
    "pifdm_vit": {
        "method": "pifdm_vit",
        "fuse_physics_queries": True,
        "model": {
            "width": 192, "depth": 12, "heads": 6, "patch_size": [4, 4],
            "mlp_ratio": 4.0, "coordinate_projection_scale": 0.05,
            "coordinate_projection_seed": 0,
            "decoder_checkpoint": False,
        },
        "diffusion": {
            "timesteps": 1000, "beta_schedule": "cosine",
        },
        "physics_training": {
            "lambda_phy": 1e-4, "lambda_bc": 1e-4,
            "start_step": 10000, "warmup_steps": 5000,
            "collocation_points": 2048, "boundary_points": 512,
        },
        "training": {
            "batch_size": 8, "steps": 100000, "learning_rate": 1e-4,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.99], "weight_decay": 0.00,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "ema_decay": 0.995, "ema_update_every": 10,
            "num_workers": 0, "log_every": 100, "save_every": 100000,
        },
    },
    "pifdm_unet": {
        "method": "pifdm_unet",
        "fuse_physics_queries": True,
        "model": {
            "width": 64, "dim_mults": [1, 2, 4],
            "coordinate_projection_scale": 0.05, "coordinate_projection_seed": 0,
            "decoder_checkpoint": False,
        },
        "diffusion": {
            "timesteps": 1000, "beta_schedule": "cosine",
        },
        "physics_training": {
            "lambda_phy": 1e-8, "lambda_bc": 1e-8,
            "start_step": 10000, "warmup_steps": 5000,
            "collocation_points": 2048, "boundary_points": 512,
        },
        "training": {
            "batch_size": 8, "steps": 100000, "learning_rate": 1e-4,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.99], "weight_decay": 0.00,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "ema_decay": 0.995, "ema_update_every": 10,
            "num_workers": 0, "log_every": 100, "save_every": 100000,
        },
    },
    "deeponet": {
        "method": "deeponet",
        "model": {
            "width": 128, "hidden": 64, "layers": 6,
            # Only g_L(t) varies across this dataset. Initial E and g_R are
            # fixed known conditions, so they do not enter the branch.
            "branch_encoder": "left_robin_cnn", "branch_channels": 32, "branch_layers": 4,
            "trunk_fourier_features": False, "trunk_fourier_dim": 32,
            "trunk_fourier_scale": 1.0, "trunk_fourier_seed": 0,
            # Start near the training-field mean instead of the data-range midpoint.
            "output_softplus_range_fraction": 1.0, "output_softplus_initial_value": 0.8,
        },
        "training": {
            "batch_size": 32, "steps": 100000, "learning_rate": 1e-3,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.999], "weight_decay": 0.0,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "num_workers": 0,
            "log_every": 100, "save_every": 100000,
        },
    },
    "pideeponet": {
        "method": "pideeponet",
        "model": {
            "width": 128, "hidden": 64, "layers": 6,
            "branch_encoder": "left_robin_cnn", "branch_channels": 32, "branch_layers": 4,
            "output_softplus_range_fraction": 1.0, "output_softplus_initial_value": 0.8,
        },
        "physics_training": {
            "lambda_phy": 1e-7, "lambda_bc": 1e-7,
            "start_step": 5000, "warmup_steps": 5000,
            "collocation_points": 2048, "boundary_points": 256,
        },
        "training": {
            "batch_size": 8, "steps": 100000, "learning_rate": 1e-3,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.999], "weight_decay": 0.0,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "num_workers": 0,
            "log_every": 100, "save_every": 100000,
        },
    },
    "fno": {
        "method": "fno",
        "model": {
            "width": 32, "modes": [16, 24], "layers": 4,
            "projection_width": 64,
        },
        "training": {
            "batch_size": 8, "steps": 100000, "learning_rate": 1e-3,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.999], "weight_decay": 1e-4,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "num_workers": 0,
            "log_every": 100, "save_every": 100000,
        },
    },
    "pino": {
        "method": "pino",
        "model": {
            "width": 32, "modes": [16, 24], "layers": 4,
            "projection_width": 64,
        },
        "physics_training": {
            "lambda_phy": 1e-7, "lambda_bc": 1e-7,
            "start_step": 10000, "warmup_steps": 5000,
            "point_sampling": "grid", "huber_delta": 1.0,
        },
        "training": {
            "batch_size": 8, "steps": 100000, "learning_rate": 1e-3,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.999], "weight_decay": 1e-4,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "num_workers": 0,
            "log_every": 100, "save_every": 100000,
        },
    },
    "diffusionpde": {
        "method": "diffusionpde",
        "model": {"width": 64, "dim_mults": [1, 2, 4]},
        "diffusion": {"timesteps": 1000, "beta_schedule": "cosine"},
        "physics_training": {
            "lambda_phy": 1e-7, "lambda_bc": 1e-7,
            "start_step": 10000, "warmup_steps": 5000,
            "point_sampling": "grid", "huber_delta": 1.0,
        },
        "training": {
            "batch_size": 8, "steps": 100000, "learning_rate": 1e-4,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.99], "weight_decay": 0.00,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "ema_decay": 0.995, "ema_update_every": 10,
            "num_workers": 0, "log_every": 100, "save_every": 100000,
        },
    },
}

def build_config(method=METHOD):
    config = deepcopy(CASE_CONFIG)
    config.update(deepcopy(METHOD_SETTINGS[method]))
    return config


def output_directory(method=METHOD):
    return CASE_DIR / "runs" / method


RUN_SETTINGS = {
    "data": DATA_PATHS["train_output"],
    "output": output_directory(METHOD),
    "device": "auto",
    "resume": None,
    "duration_override": None,  # 所有方法都表示 optimizer steps
}


def main():
    import torch
    from pifdm.workflows.training import train_experiment

    # All radiative_1d UNet inputs have fixed shapes; let cuDNN select the
    # fastest convolution algorithm for this experiment.
    torch.backends.cudnn.benchmark = METHOD.endswith("_unet")
    train_experiment(
        build_config(), data_path=RUN_SETTINGS["data"], output_dir=RUN_SETTINGS["output"],
        device=RUN_SETTINGS["device"], duration=RUN_SETTINGS["duration_override"],
        resume=RUN_SETTINGS["resume"],
    )


if __name__ == "__main__":
    main()
