"""Named method builders shared by all case-study training files."""

from __future__ import annotations

from .backbones import (ContinuousUNet3DBackbone, ContinuousUNetBackbone,
                        ContinuousViT, ContinuousViT3D, DeepONet, FNOBackbone,
                        UNet3DBackbone, UNetBackbone, ViT3DBackbone, ViTBackbone)


METHODS = ("fdm_unet", "fdm_vit", "pifdm_unet", "pifdm_vit",
           "deeponet", "pideeponet", "fno", "pino")


def frozen_condition_size(input_shape):
    if len(input_shape) == 2:
        nt, nx = input_shape
        return nx + 2 * (nt - 1)
    if len(input_shape) == 3:
        nt, ny, nx = input_shape
        return ny * nx + (nt - 1) * (2 * ny + 2 * (nx - 2))
    raise ValueError(f"unsupported field shape {input_shape}")


def build_method(name: str, *, input_shape, in_channels: int = 1, coord_dim: int = 2,
                 grid_coordinates=None, **kwargs):
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}; expected one of {METHODS}")
    is_3d = len(input_shape) == 3
    patch_size = tuple(kwargs.get("patch_size", (2, 2, 2) if is_3d else (2, 4)))
    output_options = {
        "output_floor": kwargs.get("output_floor"),
        "output_softplus_beta": kwargs.get("output_softplus_beta", 10.0),
        "output_softplus_scale": kwargs.get("output_softplus_scale"),
        "output_softplus_initial_bias": kwargs.get("output_softplus_initial_bias"),
    }
    continuous_output_options = {
        **output_options,
        "output_softplus_scale": kwargs.get("output_softplus_scale"),
        "output_softplus_initial_bias": kwargs.get("output_softplus_initial_bias"),
    }
    if name in ("fdm_vit", "pifdm_vit") and any(size % patch for size, patch in zip(input_shape, patch_size)):
        raise ValueError(f"input_shape {tuple(input_shape)} must be divisible by patch_size {patch_size}")
    if name == "fdm_unet":
        if is_3d:
            return UNet3DBackbone(in_channels=in_channels, width=kwargs.get("width", 32),
                                  depth=kwargs.get("depth", 3), dim_mults=kwargs.get("dim_mults"),
                                  padding_mode=kwargs.get("padding_mode", "zeros"),
                                  **output_options)
        return UNetBackbone(in_channels=in_channels, width=kwargs.get("width", 64),
                            depth=kwargs.get("depth", 3), dim_mults=kwargs.get("dim_mults"),
                            padding_mode=kwargs.get("padding_mode", "zeros"),
                            **output_options)
    if name == "pifdm_unet":
        continuous_unet = ContinuousUNet3DBackbone if is_3d else ContinuousUNetBackbone
        return continuous_unet(
            input_shape=input_shape, in_channels=in_channels,
            width=kwargs.get("width", 32 if is_3d else 64),
            depth=kwargs.get("depth", 3), dim_mults=kwargs.get("dim_mults"),
            coordinate_projection_scale=kwargs.get("coordinate_projection_scale", 0.05),
            coordinate_projection_seed=kwargs.get("coordinate_projection_seed", 0),
            decoder_chunk_size=kwargs.get("decoder_chunk_size", 16384),
            decoder_checkpoint=kwargs.get("decoder_checkpoint", True),
            extrapolate_grid_axes=kwargs.get("extrapolate_grid_axes", False),
            padding_mode=kwargs.get("padding_mode", "zeros"),
            material_function=kwargs.get("material_function"),
            grid_coordinates=grid_coordinates, **continuous_output_options,
        )
    if name in ("fno", "pino"):
        return FNOBackbone(
            3 if is_3d else 2, in_channels=in_channels,
            width=kwargs.get("width", 64), modes=kwargs.get("modes"),
            layers=kwargs.get("layers", 4),
            projection_width=kwargs.get("projection_width", 128),
            grid_coordinates=grid_coordinates,
            **output_options,
        )
    if name == "pifdm_vit":
        if is_3d:
            return ContinuousViT3D(input_shape=input_shape, in_channels=in_channels,
                                   width=kwargs.get("width", 128), depth=kwargs.get("depth", 4),
                                   heads=kwargs.get("heads", 4), patch_size=patch_size,
                                   mlp_ratio=kwargs.get("mlp_ratio", 4.0), dropout=kwargs.get("dropout", 0.0),
                                   coordinate_projection_scale=kwargs.get("coordinate_projection_scale", 0.05),
                                   coordinate_projection_seed=kwargs.get("coordinate_projection_seed", 0),
                                   decoder_chunk_size=kwargs.get("decoder_chunk_size", 16384),
                                   decoder_checkpoint=kwargs.get("decoder_checkpoint", True),
                                   extrapolate_grid_axes=kwargs.get("extrapolate_grid_axes", False),
                                   material_function=kwargs.get("material_function"),
                                   grid_coordinates=grid_coordinates, **continuous_output_options)
        return ContinuousViT(input_shape=input_shape, in_channels=in_channels,
                             width=kwargs.get("width", 128), depth=kwargs.get("depth", 4),
                             heads=kwargs.get("heads", 4), patch_size=patch_size,
                             mlp_ratio=kwargs.get("mlp_ratio", 4.0), dropout=kwargs.get("dropout", 0.0),
                             coordinate_projection_scale=kwargs.get("coordinate_projection_scale", 0.05),
                             coordinate_projection_seed=kwargs.get("coordinate_projection_seed", 0),
                             decoder_chunk_size=kwargs.get("decoder_chunk_size", 16384),
                             decoder_checkpoint=kwargs.get("decoder_checkpoint", True),
                             extrapolate_grid_axes=kwargs.get("extrapolate_grid_axes", False),
                             material_function=kwargs.get("material_function"),
                             grid_coordinates=grid_coordinates, **continuous_output_options)
    if name == "fdm_vit":
        if is_3d:
            return ViT3DBackbone(input_shape=input_shape, in_channels=in_channels,
                                 width=kwargs.get("width", 128), depth=kwargs.get("depth", 4),
                                 heads=kwargs.get("heads", 4), patch_size=patch_size,
                                 mlp_ratio=kwargs.get("mlp_ratio", 4.0), dropout=kwargs.get("dropout", 0.0),
                                 zero_modulation_init=kwargs.get("zero_modulation_init", True),
                                 **output_options)
        return ViTBackbone(input_shape=input_shape, in_channels=in_channels,
                           width=kwargs.get("width", 128), depth=kwargs.get("depth", 4),
                           heads=kwargs.get("heads", 4), patch_size=patch_size,
                           mlp_ratio=kwargs.get("mlp_ratio", 4.0), dropout=kwargs.get("dropout", 0.0),
                           zero_modulation_init=kwargs.get("zero_modulation_init", True),
                           **output_options)
    branch_encoder = kwargs.get("branch_encoder", "frozen_mlp")
    if branch_encoder == "frozen_mlp":
        branch_dim = kwargs.get("branch_dim", frozen_condition_size(input_shape))
    elif branch_encoder == "left_robin_mlp":
        if len(input_shape) != 2:
            raise ValueError("left_robin_mlp is only defined for 1D space-time fields")
        branch_dim = kwargs.get("branch_dim", int(input_shape[0]))
    elif branch_encoder == "left_robin_cnn":
        if len(input_shape) != 2:
            raise ValueError("left_robin_cnn is only defined for 1D space-time fields")
        branch_dim = None
    elif branch_encoder == "left_robin_2d_mlp":
        if len(input_shape) != 3:
            raise ValueError("left_robin_2d_mlp is only defined for 2D space-time fields")
        branch_dim = kwargs.get("branch_dim", int(input_shape[0] * input_shape[1]))
    elif branch_encoder == "left_robin_2d_cnn":
        if len(input_shape) != 3:
            raise ValueError("left_robin_2d_cnn is only defined for 2D space-time fields")
        branch_dim = None
    else:
        branch_dim = None
    return DeepONet(branch_dim=branch_dim, coord_dim=coord_dim, input_shape=input_shape,
                    grid_coordinates=grid_coordinates, width=kwargs.get("width", 128),
                    hidden=kwargs.get("hidden", 100), layers=kwargs.get("layers", 6),
                    material_function=kwargs.get("material_function"), branch_encoder=branch_encoder,
                    branch_channels=kwargs.get("branch_channels", 16),
                    branch_layers=kwargs.get("branch_layers", 3),
                    trunk_fourier_features=kwargs.get("trunk_fourier_features", False),
                    trunk_fourier_dim=kwargs.get("trunk_fourier_dim", 32),
                    trunk_fourier_scale=kwargs.get("trunk_fourier_scale", 1.0),
                    trunk_fourier_seed=kwargs.get("trunk_fourier_seed", 0),
                    **output_options)
