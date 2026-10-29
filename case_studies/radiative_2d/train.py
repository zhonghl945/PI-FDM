"""二维非线性、多介质辐射扩散：所有可训练方法共用的训练主文件。"""

from copy import deepcopy
from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_studies.radiative_2d.generate import CASE_CONFIG, RUN_SETTINGS as DATA_PATHS

# fdm_unet / fdm_vit / c_fdm_unet / c_fdm_vit / pifdm_unet / pifdm_vit
# deeponet / pideeponet / fno / pino / diffusionpde
METHOD = "pifdm_unet"

# 每种方法只列实际生效的参数，便于直接审查单个实验。
METHOD_SETTINGS = {
    "fdm_unet": {
        "method": "fdm_unet",
        "material_conditioning": True,
        "use_condition_channels": False,
        "model": {"width": 64, "dim_mults": [1, 2], "padding_mode": "replicate"},
        "diffusion": {"timesteps": 1000, "beta_schedule": "cosine"},
        "training": {
            "batch_size": 4, "steps": 50000, "learning_rate": 1e-4,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.99], "weight_decay": 0.00,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "ema_decay": 0.995, "ema_update_every": 10,
            "num_workers": 0, "log_every": 100, "save_every": 100000,
        },
    },
    "fdm_vit": {
        "method": "fdm_vit",
        "material_conditioning": True,
        "use_condition_channels": False,
        "model": {
            "width": 128, "depth": 8, "heads": 4,
            "patch_size": [4, 4, 4], "mlp_ratio": 4.0,
            # Match the archived DiT's ordinary random AdaLN initialization
            # for this case only; other cases keep AdaLN-Zero by default.
            "zero_modulation_init": False,
        },
        "diffusion": {"timesteps": 1000, "beta_schedule": "cosine"},
        "training": {
            "batch_size": 4, "steps": 100000, "learning_rate": 1e-4,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.99], "weight_decay": 0.00,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": False, "ema_decay": 0.995, "ema_update_every": 10,
            "num_workers": 0, "log_every": 100, "save_every": 100000,
        },
    },
    "c_fdm_vit": {
        "method": "c_fdm_vit",
        "material_conditioning": True,
        "model": {
            "width": 128, "depth": 8, "heads": 4, "patch_size": [4, 4, 4],
            "mlp_ratio": 4.0, "coordinate_projection_scale": 0.05,
            "coordinate_projection_seed": 0, "decoder_chunk_size": 150000,
            "decoder_checkpoint": False, "extrapolate_grid_axes": True,
        },
        "diffusion": {"timesteps": 1000, "beta_schedule": "cosine"},
        "training": {
            "batch_size": 4, "steps": 100000, "learning_rate": 1e-4,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.99], "weight_decay": 0.00,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "ema_decay": 0.995, "ema_update_every": 10,
            "num_workers": 0, "log_every": 100, "save_every": 100000,
        },
    },
    "c_fdm_unet": {
        "method": "c_fdm_unet",
        "material_conditioning": True,
        "use_condition_channels": False,
        "model": {
            "width": 64, "dim_mults": [1, 2], "padding_mode": "replicate",
            "coordinate_projection_scale": 0.05, "coordinate_projection_seed": 0,
            "decoder_chunk_size": 150000, "decoder_checkpoint": False,
            "extrapolate_grid_axes": True,
        },
        "diffusion": {"timesteps": 1000, "beta_schedule": "cosine"},
        "training": {
            "batch_size": 4, "steps": 50000, "learning_rate": 1e-4,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.99], "weight_decay": 0.00,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "ema_decay": 0.995, "ema_update_every": 10,
            "num_workers": 0, "log_every": 100, "save_every": 25000,
        },
    },
    "pifdm_vit": {
        "method": "pifdm_vit",
        "fuse_physics_queries": True,
        "material_conditioning": True,
        "model": {
            "width": 128, "depth": 8, "heads": 4, "patch_size": [4, 4, 4],
            "mlp_ratio": 4.0, "coordinate_projection_scale": 0.05,
            "coordinate_projection_seed": 0, "decoder_chunk_size": 150000,
            "decoder_checkpoint": False, "extrapolate_grid_axes": True,
        },
        "diffusion": {"timesteps": 1000, "beta_schedule": "cosine"},
        "physics_training": {
            "lambda_phy": 1e-8, "lambda_bc": 1e-8,
            "start_step": 5000, "warmup_steps": 5000,
            "collocation_points": 4096, "boundary_points": 400,
            "huber_delta": 1.0,
        },
        "training": {
            "batch_size": 4, "steps": 100000, "learning_rate": 1e-4,
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
        "material_conditioning": True,
        "use_condition_channels": True,
        "model": {
            "width": 64, "dim_mults": [1, 2], "padding_mode": "replicate",
            "coordinate_projection_scale": 0.05, "coordinate_projection_seed": 0,
            "decoder_chunk_size": 150000, "decoder_checkpoint": False,
            "extrapolate_grid_axes": True,
        },
        "diffusion": {"timesteps": 1000, "beta_schedule": "cosine"},
        "physics_training": {
            "lambda_phy": 1e-7, "lambda_bc": 1e-7,
            "start_step": 10000, "warmup_steps": 5000,
            "collocation_points": 4096, "boundary_points": 400,
            "huber_delta": 1.0,
        },
        "training": {
            "batch_size": 4, "steps": 100000, "learning_rate": 1e-4,
            "minimum_learning_rate": 0.0, "scheduler": "cosine",
            "adam_betas": [0.9, 0.99], "weight_decay": 0.00,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "ema_decay": 0.995, "ema_update_every": 10,
            "num_workers": 0, "log_every": 100, "save_every": 100000,
        },
    },
    "deeponet": {
        "method": "deeponet",
        "material_conditioning": True,
        "model": {
            "width": 64, "hidden": 64, "layers": 6,
            # The only sample-varying operator input is the saved left Robin
            # history gL(t,y); the initial field is fixed across this case.
            "branch_encoder": "left_robin_2d_cnn",
            "branch_channels": 32, "branch_layers": 3,
        },
        "training": {
            "batch_size": 10, "steps": 100000, "learning_rate": 5e-4,
            "minimum_learning_rate": 1e-5, "scheduler": "cosine",
            "adam_betas": [0.9, 0.999], "weight_decay": 0.0,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": False, "num_workers": 0,
            "log_every": 100, "save_every": 100000,
        },
    },
    "pideeponet": {
        "method": "pideeponet",
        "material_conditioning": True,
        "model": {
            "width": 64, "hidden": 64, "layers": 6,
            # PI-DeepONet receives the same sample-specific gL(t,y) branch as
            # DeepONet, which also makes its Robin residual well-defined.
            "branch_encoder": "left_robin_2d_cnn",
            "branch_channels": 32, "branch_layers": 3,
        },
        "physics_training": {
            "lambda_phy": 1e-7, "lambda_bc": 1e-7,
            "start_step": 50000, "warmup_steps": 5000,
            "point_sampling": "grid_subsample",
            "boundary_point_sampling": "grid",
            "collocation_points": 10240,
        },
        "training": {
            "batch_size": 10, "steps": 100000, "learning_rate": 5e-4,
            "minimum_learning_rate": 1e-5, "scheduler": "cosine",
            "adam_betas": [0.9, 0.999], "weight_decay": 0.0,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": False, "num_workers": 0,
            "log_every": 100, "save_every": 100000,
        },
    },
    "fno": {
        "method": "fno",
        "material_conditioning": True,
        "model": {
            "width": 16, "modes": [8, 8, 8], "layers": 4,
            "projection_width": 64,
        },
        "training": {
            "batch_size": 10, "steps": 100000, "learning_rate": 5e-4,
            "minimum_learning_rate": 1e-5, "scheduler": "cosine",
            "adam_betas": [0.9, 0.999], "weight_decay": 0.0,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "num_workers": 0,
            "log_every": 100, "save_every": 100000,
        },
    },
    "pino": {
        "method": "pino",
        "material_conditioning": True,
        "model": {
            "width": 16, "modes": [8, 8, 8], "layers": 4,
            "projection_width": 64,
        },
        "physics_training": {
            "lambda_phy": 1e-7, "lambda_bc": 1e-7,
            "start_step": 5000, "warmup_steps": 5000,
            "point_sampling": "grid", "huber_delta": 1.0,
        },
        "training": {
            "batch_size": 10, "steps": 100000, "learning_rate": 5e-4,
            "minimum_learning_rate": 1e-5, "scheduler": "cosine",
            "adam_betas": [0.9, 0.999], "weight_decay": 0.0,
            "gradient_clip": 1.0, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "num_workers": 0,
            "log_every": 100, "save_every": 100000,
        },
    },
    "diffusionpde": {
        "method": "diffusionpde",
        "material_conditioning": True,
        "use_condition_channels": True,
        "model": {"width": 64, "dim_mults": [1, 2], "padding_mode": "replicate"},
        "diffusion": {"timesteps": 1000, "beta_schedule": "cosine"},
        "physics_training": {
            "lambda_phy": 1e-7, "lambda_bc": 1e-7,
            "start_step": 5000, "warmup_steps": 5000,
            "point_sampling": "grid", "huber_delta": 1.0,
        },
        "training": {
            "batch_size": 4, "steps": 100000, "learning_rate": 1e-4,
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
    from pifdm.workflows.training import train_experiment

    train_experiment(
        build_config(), data_path=RUN_SETTINGS["data"], output_dir=RUN_SETTINGS["output"],
        device=RUN_SETTINGS["device"], duration=RUN_SETTINGS["duration_override"],
        resume=RUN_SETTINGS["resume"],
    )


if __name__ == "__main__":
    main()
