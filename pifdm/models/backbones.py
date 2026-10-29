"""Production backbones shared by all configured PDE cases."""

from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from .continuous_decoder import ContinuousDecoder


def _positive_output(values: torch.Tensor, floor: float | None, beta: float,
                     scale: float | None = None) -> torch.Tensor:
    if floor is None:
        return values
    if scale is not None:
        return float(floor) + float(scale) * F.softplus(values, beta=float(beta))
    return float(floor) + F.softplus(
        values - float(floor), beta=float(beta)
    )


def _register_grid_coordinates(module: nn.Module, coordinates, input_shape) -> None:
    module._grid_axis_count = 0
    if coordinates is None:
        return
    if len(coordinates) != len(input_shape):
        raise ValueError("grid_coordinates must match input_shape")
    for index, (axis, size) in enumerate(zip(coordinates, input_shape)):
        values = torch.as_tensor(axis, dtype=torch.float32)
        if values.ndim != 1 or len(values) != int(size):
            raise ValueError("each grid coordinate axis must match input_shape")
        module.register_buffer(f"_grid_axis_{index}", values, persistent=False)
    module._grid_axis_count = len(coordinates)


def _grid_coordinates(module: nn.Module, reference: torch.Tensor):
    if not module._grid_axis_count:
        return None
    return tuple(
        getattr(module, f"_grid_axis_{index}").to(device=reference.device, dtype=reference.dtype)
        for index in range(module._grid_axis_count)
    )


def extract_frozen_values(field: torch.Tensor) -> torch.Tensor:
    """Flatten unique initial/boundary condition slots."""
    if field.ndim == 3:
        return torch.cat((field[:, 0], field[:, 1:, 0], field[:, 1:, -1]), dim=1)
    if field.ndim == 4:
        initial = field[:, 0].flatten(1)
        later = field[:, 1:]
        return torch.cat((initial, later[..., 0].flatten(1), later[..., -1].flatten(1),
                          later[:, :, 0, 1:-1].flatten(1), later[:, :, -1, 1:-1].flatten(1)), dim=1)
    raise ValueError(f"expected (B,Nt,Nx) or (B,Nt,Ny,Nx), got {tuple(field.shape)}")


def extract_initial_field(field: torch.Tensor) -> torch.Tensor:
    """Return the interior initial field as a channel-first CNN input."""
    if field.ndim == 3:
        return field[:, :1, 1:-1]
    if field.ndim == 4:
        return field[:, :1, 1:-1, 1:-1]
    raise ValueError(f"expected (B,Nt,Nx) or (B,Nt,Ny,Nx), got {tuple(field.shape)}")


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.proj = nn.Sequential(nn.Linear(frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        frequency = torch.exp(-math.log(10000) * torch.arange(half, device=timestep.device) / max(half, 1))
        phase = timestep.float()[:, None] * frequency[None]
        embedding = torch.cat((phase.cos(), phase.sin()), dim=-1)
        return self.proj(F.pad(embedding, (0, self.frequency_dim - embedding.shape[-1])))


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlockND(nn.Module):
    def __init__(self, dimension: int, in_channels: int, out_channels: int, time_dim: int,
                 padding_mode: str = "zeros"):
        super().__init__()
        convolution = nn.Conv2d if dimension == 2 else nn.Conv3d
        self.dimension = dimension
        self.block1 = nn.Sequential(nn.GroupNorm(_group_count(in_channels), in_channels), nn.SiLU(),
                                    convolution(in_channels, out_channels, 3, padding=1,
                                                padding_mode=padding_mode))
        self.block2 = nn.Sequential(nn.GroupNorm(_group_count(out_channels), out_channels), nn.SiLU(),
                                    convolution(out_channels, out_channels, 3, padding=1,
                                                padding_mode=padding_mode))
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.skip = (convolution(in_channels, out_channels, 1, padding_mode=padding_mode)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, values, time_embedding):
        hidden = self.block1(values)
        shape = (hidden.shape[0], hidden.shape[1]) + (1,) * self.dimension
        hidden = hidden + self.time(time_embedding).reshape(shape)
        return self.block2(hidden) + self.skip(values)


class UNetND(nn.Module):
    """Multi-resolution UNet matching the archived FDM capacity pattern."""
    def __init__(self, dimension: int, in_channels=1, width=64, depth=3, dim_mults=None,
                 output_floor: float | None = None, output_softplus_beta: float = 10.0,
                 output_softplus_scale: float | None = None,
                 output_softplus_initial_bias: float | None = None,
                 padding_mode: str = "zeros"):
        super().__init__()
        convolution = nn.Conv2d if dimension == 2 else nn.Conv3d
        transpose = nn.ConvTranspose2d if dimension == 2 else nn.ConvTranspose3d
        if padding_mode not in ("zeros", "reflect", "replicate", "circular"):
            raise ValueError(
                "padding_mode must be one of 'zeros', 'reflect', 'replicate', or 'circular'"
            )
        self.padding_mode = padding_mode
        dim_mults = tuple(dim_mults or (1, 2, 4, 8)[:max(2, depth)])
        channels = [width * multiplier for multiplier in dim_mults]
        self.feature_dim = channels[0]
        self.output_floor = output_floor
        self.output_softplus_beta = float(output_softplus_beta)
        self.output_softplus_scale = (
            None if output_softplus_scale is None else float(output_softplus_scale)
        )
        time_dim = width * 4
        self.time = TimestepEmbedding(time_dim)
        self.input = convolution(in_channels, channels[0], 7, padding=3,
                                  padding_mode=padding_mode)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for index, channel in enumerate(channels):
            self.down_blocks.append(nn.ModuleList((
                ResBlockND(dimension, channel, channel, time_dim, padding_mode=padding_mode),
                ResBlockND(dimension, channel, channel, time_dim, padding_mode=padding_mode),
            )))
            if index + 1 < len(channels):
                self.downsample.append(convolution(
                    channel, channels[index + 1], 4, stride=2, padding=1,
                    padding_mode=padding_mode,
                ))
        self.middle = nn.ModuleList((
            ResBlockND(dimension, channels[-1], channels[-1], time_dim,
                       padding_mode=padding_mode),
            ResBlockND(dimension, channels[-1], channels[-1], time_dim,
                       padding_mode=padding_mode),
        ))
        self.upsample = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for index in range(len(channels) - 1, 0, -1):
            self.upsample.append(transpose(channels[index], channels[index - 1], 4, stride=2, padding=1))
            self.up_blocks.append(nn.ModuleList((
                ResBlockND(dimension, 2 * channels[index - 1], channels[index - 1], time_dim,
                           padding_mode=padding_mode),
                ResBlockND(dimension, channels[index - 1], channels[index - 1], time_dim,
                           padding_mode=padding_mode),
            )))
        self.output = nn.Sequential(nn.GroupNorm(_group_count(channels[0]), channels[0]), nn.SiLU(),
                                    convolution(channels[0], 1, 1, padding_mode=padding_mode))
        if output_softplus_initial_bias is not None:
            nn.init.constant_(self.output[-1].bias, float(output_softplus_initial_bias))

    def encode_features(self, values, timestep):
        """Return the full-resolution multi-scale UNet feature lattice."""
        time = self.time(timestep)
        hidden = self.input(values)
        skips = []
        for index, blocks in enumerate(self.down_blocks):
            hidden = blocks[0](hidden, time)
            hidden = blocks[1](hidden, time)
            skips.append(hidden)
            if index < len(self.downsample):
                hidden = self.downsample[index](hidden)
        hidden = self.middle[1](self.middle[0](hidden, time), time)
        for upsample, blocks in zip(self.upsample, self.up_blocks):
            hidden = upsample(hidden)
            skip = skips.pop(-2)
            if hidden.shape[2:] != skip.shape[2:]:
                hidden = F.interpolate(hidden, size=skip.shape[2:], mode="nearest")
            hidden = blocks[0](torch.cat((hidden, skip), dim=1), time)
            hidden = blocks[1](hidden, time)
        return hidden

    def forward(self, values, timestep):
        hidden = self.encode_features(values, timestep)
        return _positive_output(
            self.output(hidden), self.output_floor, self.output_softplus_beta,
            self.output_softplus_scale,
        )


class UNetBackbone(UNetND):
    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)


class UNet3DBackbone(UNetND):
    def __init__(self, **kwargs):
        super().__init__(3, **kwargs)


class ContinuousUNetND(UNetND):
    """UNet feature extractor followed by a C2 continuous coordinate decoder."""

    def __init__(self, dimension: int, *, input_shape, in_channels=1, width=64,
                 depth=3, dim_mults=None, coordinate_projection_scale=0.05,
                 coordinate_projection_seed=0, decoder_chunk_size=16384,
                 decoder_checkpoint=True, grid_coordinates=None,
                 extrapolate_grid_axes=False,
                 material_function=None, output_floor: float | None = None,
                 output_softplus_beta: float = 10.0,
                 output_softplus_scale: float | None = None,
                 output_softplus_initial_bias: float | None = None,
                 padding_mode: str = "zeros"):
        super().__init__(
            dimension, in_channels=in_channels, width=width, depth=depth,
            dim_mults=dim_mults, output_floor=output_floor,
            output_softplus_beta=output_softplus_beta,
            padding_mode=padding_mode,
        )
        del self.output
        self.dimension = int(dimension)
        self.input_shape = tuple(input_shape)
        self.continuous = True
        _register_grid_coordinates(self, grid_coordinates, self.input_shape)
        self.decoder = ContinuousDecoder(
            self.feature_dim, coord_dim=self.dimension,
            coordinate_projection_scale=coordinate_projection_scale,
            coordinate_projection_seed=coordinate_projection_seed,
            chunk_size=decoder_chunk_size,
            checkpoint_decoding=decoder_checkpoint, grid_axes=grid_coordinates,
            extrapolate_grid_axes=extrapolate_grid_axes,
            material_function=material_function, output_floor=output_floor,
            output_softplus_beta=output_softplus_beta,
            output_softplus_scale=output_softplus_scale,
            output_softplus_initial_bias=output_softplus_initial_bias,
        )

    def encode(self, values, timestep):
        return self.encode_features(values, timestep)

    def decode(self, latent, *, coords=None, output_shape=None):
        if coords is not None:
            return self.decoder(latent, coords)
        if output_shape is None:
            raise ValueError("output_shape is required for regular-grid decoding")
        return self.decoder.forward_grid(
            latent, tuple(output_shape), axes=_grid_coordinates(self, latent)
        )

    def forward(self, values, timestep, coords=None):
        latent = self.encode(values, timestep)
        return self.decode(
            latent, coords=coords,
            output_shape=values.shape[-self.dimension:] if coords is None else None,
        )


class ContinuousUNetBackbone(ContinuousUNetND):
    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)


class ContinuousUNet3DBackbone(ContinuousUNetND):
    def __init__(self, **kwargs):
        super().__init__(3, **kwargs)


class AdaLNBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, width), nn.Dropout(dropout))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, tokens, condition):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).chunk(6, dim=1)
        normalized = self.norm1(tokens) * (1 + scale_a[:, None]) + shift_a[:, None]
        tokens = tokens + gate_a[:, None] * self.attention(normalized, normalized, normalized, need_weights=False)[0]
        normalized = self.norm2(tokens) * (1 + scale_m[:, None]) + shift_m[:, None]
        return tokens + gate_m[:, None] * self.mlp(normalized)


class PatchDiT(nn.Module):
    """AdaLN-conditioned 2D/3D diffusion transformer."""
    def __init__(self, dimension, input_shape, in_channels=1, width=128, depth=4, heads=4,
                 patch_size=None, mlp_ratio=4.0, dropout=0.0, continuous=False,
                 coordinate_projection_scale=0.05, coordinate_projection_seed=0,
                 decoder_chunk_size=16384, decoder_checkpoint=True, grid_coordinates=None,
                 extrapolate_grid_axes=False, material_function=None,
                 output_floor: float | None = None,
                 output_softplus_beta: float = 10.0,
                 output_softplus_scale: float | None = None,
                 output_softplus_initial_bias: float | None = None,
                 zero_modulation_init: bool = True):
        super().__init__()
        convolution = nn.Conv2d if dimension == 2 else nn.Conv3d
        transpose = nn.ConvTranspose2d if dimension == 2 else nn.ConvTranspose3d
        self.dimension = dimension
        self.input_shape = tuple(input_shape)
        self.output_floor = output_floor
        self.output_softplus_beta = float(output_softplus_beta)
        self.output_softplus_scale = (
            None if output_softplus_scale is None else float(output_softplus_scale)
        )
        self.zero_modulation_init = bool(zero_modulation_init)
        _register_grid_coordinates(self, grid_coordinates, self.input_shape)
        self.patch_size = tuple(patch_size or ((2, 4) if dimension == 2 else (2, 2, 2)))
        self.grid_shape = tuple(size // patch for size, patch in zip(input_shape, self.patch_size))
        self.embed = convolution(in_channels, width, self.patch_size, stride=self.patch_size)
        token_count = math.prod(self.grid_shape)
        self.position = nn.Parameter(torch.zeros(1, token_count, width))
        self.time = TimestepEmbedding(width)
        self.blocks = nn.ModuleList([AdaLNBlock(width, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.continuous = continuous
        if continuous:
            self.decoder = ContinuousDecoder(
                width, coord_dim=dimension,
                coordinate_projection_scale=coordinate_projection_scale,
                coordinate_projection_seed=coordinate_projection_seed,
                chunk_size=decoder_chunk_size, checkpoint_decoding=decoder_checkpoint,
                grid_axes=grid_coordinates, extrapolate_grid_axes=extrapolate_grid_axes,
                material_function=material_function,
                output_floor=output_floor,
                output_softplus_beta=output_softplus_beta,
                output_softplus_scale=output_softplus_scale,
                output_softplus_initial_bias=output_softplus_initial_bias,
            )
        else:
            self.head_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            self.head_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.head = transpose(width, 1, self.patch_size, stride=self.patch_size)
        self._initialize()
        if output_softplus_initial_bias is not None and not continuous:
            nn.init.constant_(self.head.bias, float(output_softplus_initial_bias))

    def _initialize(self):
        nn.init.normal_(self.position, std=0.02)
        if self.zero_modulation_init:
            for block in self.blocks:
                nn.init.zeros_(block.modulation[-1].weight)
                nn.init.zeros_(block.modulation[-1].bias)

    def encode(self, values, timestep):
        tokens = self.embed(values).flatten(2).transpose(1, 2) + self.position
        condition = self.time(timestep)
        for block in self.blocks:
            tokens = block(tokens, condition)
        return tokens.transpose(1, 2).reshape(values.shape[0], -1, *self.grid_shape)

    def decode(self, latent, *, coords=None, output_shape=None):
        if not self.continuous:
            raise TypeError("decode is only available for continuous backbones")
        if coords is not None:
            return self.decoder(latent, coords)
        if output_shape is None:
            raise ValueError("output_shape is required for regular-grid decoding")
        return self.decoder.forward_grid(
            latent, tuple(output_shape), axes=_grid_coordinates(self, latent)
        )

    def forward(self, values, timestep, coords=None):
        latent = self.encode(values, timestep)
        if self.continuous:
            return self.decode(
                latent, coords=coords,
                output_shape=values.shape[-self.dimension:] if coords is None else None,
            )
        tokens = latent.flatten(2).transpose(1, 2)
        shift, scale = self.head_modulation(self.time(timestep)).chunk(2, dim=1)
        tokens = self.head_norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        latent = tokens.transpose(1, 2).reshape(values.shape[0], -1, *self.grid_shape)
        output = self.head(latent)
        slices = (slice(None), slice(None), *(slice(0, size) for size in values.shape[-self.dimension:]))
        return _positive_output(
            output[slices], self.output_floor, self.output_softplus_beta,
            self.output_softplus_scale,
        )


class ViTBackbone(PatchDiT):
    def __init__(self, **kwargs): super().__init__(2, continuous=False, **kwargs)


class ViT3DBackbone(PatchDiT):
    def __init__(self, **kwargs): super().__init__(3, continuous=False, **kwargs)


class ContinuousViT(PatchDiT):
    def __init__(self, **kwargs): super().__init__(2, continuous=True, **kwargs)


class ContinuousViT3D(PatchDiT):
    def __init__(self, **kwargs): super().__init__(3, continuous=True, **kwargs)


class SpectralConvND(nn.Module):
    """Low-mode real Fourier convolution for two- or three-dimensional grids."""

    def __init__(self, dimension: int, in_channels: int, out_channels: int, modes):
        super().__init__()
        if dimension not in (2, 3):
            raise ValueError("spectral convolution supports 2D or 3D grids")
        self.dimension = dimension
        self.modes = tuple(int(mode) for mode in modes)
        if len(self.modes) != dimension or any(mode <= 0 for mode in self.modes):
            raise ValueError("one positive Fourier mode count is required per axis")
        scale = 1.0 / max(1, in_channels * out_channels)
        weight_shape = (in_channels, out_channels, *self.modes)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.view_as_real(
                scale * torch.randn(*weight_shape, dtype=torch.cfloat)
            ))
            for _ in range(2 ** (dimension - 1))
        ])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        input_dtype = values.dtype
        spatial_shape = values.shape[-self.dimension:]
        # CUDA does not implement the batched complex multiplication used by
        # einsum for ComplexHalf. Keep only the spectral path in full precision
        # while allowing the surrounding FNO convolutions to remain under AMP.
        with torch.autocast(device_type=values.device.type, enabled=False):
            spectral_values = (
                values.float()
                if values.dtype in (torch.float16, torch.bfloat16)
                else values
            )
            transformed = torch.fft.rfftn(
                spectral_values, dim=tuple(range(-self.dimension, 0))
            )
            output = transformed.new_zeros(
                values.shape[0], self.weights[0].shape[1], *transformed.shape[2:]
            )
            actual_modes = tuple(
                min(configured, max(1, size // 2))
                for configured, size in zip(self.modes[:-1], spatial_shape[:-1])
            ) + (min(self.modes[-1], transformed.shape[-1]),)
            weight_slice = (
                slice(None), slice(None),
                *(slice(0, mode) for mode in actual_modes),
            )
            for corner, weight in enumerate(self.weights):
                input_slices = [slice(None), slice(None)]
                output_slices = [slice(None), slice(None)]
                for axis, mode in enumerate(actual_modes[:-1]):
                    negative = bool(corner & (1 << axis))
                    frequency_slice = slice(-mode, None) if negative else slice(0, mode)
                    input_slices.append(frequency_slice)
                    output_slices.append(frequency_slice)
                input_slices.append(slice(0, actual_modes[-1]))
                output_slices.append(slice(0, actual_modes[-1]))
                output[tuple(output_slices)] = torch.einsum(
                    "bi...,io...->bo...",
                    transformed[tuple(input_slices)],
                    torch.view_as_complex(weight[weight_slice].contiguous()).to(
                        dtype=transformed.dtype
                    ),
                )
            result = torch.fft.irfftn(
                output, s=spatial_shape, dim=tuple(range(-self.dimension, 0))
            )
        return result.to(dtype=input_dtype)


class FNOBackbone(nn.Module):
    """Grid-to-grid FNO with explicit condition, mask and coordinate inputs."""

    def __init__(self, dimension: int, in_channels: int, width: int = 64,
                 modes=None, layers: int = 4, projection_width: int = 128,
                 grid_coordinates=None,
                 output_floor: float | None = None, output_softplus_beta: float = 10.0,
                 output_softplus_scale: float | None = None,
                 output_softplus_initial_bias: float | None = None):
        super().__init__()
        if layers < 1:
            raise ValueError("FNO needs at least one Fourier layer")
        modes = tuple(modes or ((16, 16) if dimension == 2 else (12, 12, 12)))
        convolution = nn.Conv2d if dimension == 2 else nn.Conv3d
        self.dimension = dimension
        self.output_floor = output_floor
        self.output_softplus_beta = float(output_softplus_beta)
        self.output_softplus_scale = (
            None if output_softplus_scale is None else float(output_softplus_scale)
        )
        _register_grid_coordinates(self, grid_coordinates, tuple(
            len(axis) for axis in grid_coordinates
        ) if grid_coordinates is not None else ())
        self.lift = convolution(in_channels + dimension, width, 1)
        self.spectral_layers = nn.ModuleList([
            SpectralConvND(dimension, width, width, modes) for _ in range(layers)
        ])
        self.local_layers = nn.ModuleList([
            convolution(width, width, 1) for _ in range(layers)
        ])
        self.projection = nn.Sequential(
            convolution(width, projection_width, 1), nn.GELU(),
            convolution(projection_width, 1, 1),
        )
        if output_softplus_initial_bias is not None:
            nn.init.constant_(self.projection[-1].bias, float(output_softplus_initial_bias))

    def _coordinates(self, values: torch.Tensor) -> torch.Tensor:
        axes = _grid_coordinates(self, values)
        if axes is None:
            axes = tuple(
                torch.linspace(-1.0, 1.0, size, device=values.device, dtype=values.dtype)
                for size in values.shape[-self.dimension:]
            )
        if tuple(len(axis) for axis in axes) != tuple(values.shape[-self.dimension:]):
            raise ValueError("FNO grid coordinates do not match the input grid")
        mesh = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
        return mesh.unsqueeze(0).expand(values.shape[0], -1, *mesh.shape[1:])

    def forward(self, values: torch.Tensor, timestep=None, coords=None) -> torch.Tensor:
        if coords is not None:
            raise ValueError("FNO accepts regular grids, not point-query coordinates")
        if values.ndim != self.dimension + 2:
            raise ValueError(f"FNO expected {self.dimension + 2} dimensions, got {values.shape}")
        hidden = self.lift(torch.cat((values, self._coordinates(values)), dim=1))
        for index, (spectral, local) in enumerate(zip(self.spectral_layers, self.local_layers)):
            hidden = spectral(hidden) + local(hidden)
            if index + 1 < len(self.spectral_layers):
                hidden = F.gelu(hidden)
        return _positive_output(
            self.projection(hidden), self.output_floor, self.output_softplus_beta,
            self.output_softplus_scale,
        )


def _deep_mlp(input_dim, output_dim, hidden, layers):
    modules = []
    for index in range(layers):
        modules.extend((nn.Linear(input_dim if index == 0 else hidden, hidden), nn.Tanh()))
    modules.append(nn.Linear(hidden, output_dim))
    return nn.Sequential(*modules)


class FixedFourierFeatures(nn.Module):
    """Map coordinates to fixed sinusoidal features without trainable frequencies."""
    def __init__(self, input_dim, frequency_count=32, scale=1.0, seed=0):
        super().__init__()
        input_dim = int(input_dim)
        frequency_count = int(frequency_count)
        scale = float(scale)
        if input_dim <= 0:
            raise ValueError("Fourier features require a positive input dimension")
        if frequency_count <= 0:
            raise ValueError("Fourier features require a positive frequency count")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Fourier feature scale must be finite and positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        frequencies = torch.randn(
            input_dim, frequency_count, generator=generator
        ) * scale
        self.register_buffer("frequencies", frequencies)
        self.output_dim = input_dim + 2 * frequency_count

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * torch.matmul(
            coordinates, self.frequencies.to(device=coordinates.device, dtype=coordinates.dtype)
        )
        return torch.cat((coordinates, phase.sin(), phase.cos()), dim=-1)


class InitialConditionCNNBranch(nn.Module):
    """Encode an initial spatial field without consuming packed boundary slots."""
    def __init__(self, spatial_dim, spatial_shape, output_dim, channels=16, layers=3):
        super().__init__()
        if spatial_dim not in (1, 2):
            raise ValueError(f"CNN branch supports one or two spatial dimensions, got {spatial_dim}")
        if channels <= 0 or layers <= 0:
            raise ValueError("CNN branch channels and layers must be positive")
        convolution = nn.Conv1d if spatial_dim == 1 else nn.Conv2d
        normalization = nn.GroupNorm
        modules = []
        input_channels = 1
        output_channels = int(channels)
        reduced_shape = tuple(int(size) for size in spatial_shape)
        for level in range(int(layers)):
            output_channels = int(channels) * (2 ** min(level, 2))
            groups = min(8, output_channels)
            while output_channels % groups:
                groups -= 1
            modules.extend((
                convolution(input_channels, output_channels, kernel_size=3, stride=2, padding=1),
                normalization(groups, output_channels),
                nn.GELU(),
            ))
            reduced_shape = tuple((size + 1) // 2 for size in reduced_shape)
            input_channels = output_channels
        self.features = nn.Sequential(*modules)
        self.projection = nn.Linear(output_channels * math.prod(reduced_shape), output_dim)

    def forward(self, initial: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(initial).flatten(1))


class LeftRobinCNNBranch(nn.Module):
    """Encode the left Robin history g_L(t) for a one-dimensional problem."""
    def __init__(self, time_steps, output_dim, channels=16, layers=3):
        super().__init__()
        if time_steps <= 0:
            raise ValueError("left Robin branch requires a positive number of time steps")
        if channels <= 0 or layers <= 0:
            raise ValueError("CNN branch channels and layers must be positive")
        modules = []
        input_channels = 1
        output_channels = int(channels)
        reduced_steps = int(time_steps)
        for level in range(int(layers)):
            output_channels = int(channels) * (2 ** min(level, 2))
            groups = min(8, output_channels)
            while output_channels % groups:
                groups -= 1
            modules.extend((
                nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(groups, output_channels),
                nn.GELU(),
            ))
            reduced_steps = (reduced_steps + 1) // 2
            input_channels = output_channels
        self.features = nn.Sequential(*modules)
        self.projection = nn.Linear(output_channels * reduced_steps, output_dim)

    def forward(self, left_robin: torch.Tensor) -> torch.Tensor:
        if left_robin.ndim != 2:
            raise ValueError("left Robin CNN branch expects shape (batch, time)")
        return self.projection(self.features(left_robin.unsqueeze(1)).flatten(1))


class LeftRobin2DCNNBranch(nn.Module):
    """Encode a 2D left Robin history ``g_L(t, y)`` with a CNN."""
    def __init__(self, time_steps, y_steps, output_dim, channels=16, layers=3):
        super().__init__()
        if time_steps <= 0 or y_steps <= 0:
            raise ValueError("2D left Robin branch requires positive time and y sizes")
        if channels <= 0 or layers <= 0:
            raise ValueError("CNN branch channels and layers must be positive")
        modules = []
        input_channels = 1
        output_channels = int(channels)
        reduced_shape = (int(time_steps), int(y_steps))
        for level in range(int(layers)):
            output_channels = int(channels) * (2 ** min(level, 2))
            groups = min(8, output_channels)
            while output_channels % groups:
                groups -= 1
            modules.extend((
                nn.Conv2d(input_channels, output_channels, kernel_size=3,
                          stride=2, padding=1),
                nn.GroupNorm(groups, output_channels),
                nn.GELU(),
            ))
            reduced_shape = tuple((size + 1) // 2 for size in reduced_shape)
            input_channels = output_channels
        self.features = nn.Sequential(*modules)
        self.projection = nn.Linear(output_channels * math.prod(reduced_shape), output_dim)

    def forward(self, left_robin: torch.Tensor) -> torch.Tensor:
        if left_robin.ndim != 3:
            raise ValueError("2D left Robin CNN branch expects shape (batch, time, y)")
        return self.projection(self.features(left_robin.unsqueeze(1)).flatten(1))


class DeepONet(nn.Module):
    """DeepONet with direct coordinate/material trunk inputs."""
    def __init__(self, branch_dim, coord_dim, width=128, hidden=100, layers=6,
                 output_channels=1, input_shape=None, grid_coordinates=None,
                 branch_encoder="frozen_mlp", branch_channels=16, branch_layers=3,
                 material_function=None, trunk_fourier_features=False,
                 trunk_fourier_dim=32, trunk_fourier_scale=1.0, trunk_fourier_seed=0,
                 output_floor: float | None = None, output_softplus_beta: float = 10.0,
                 output_softplus_scale: float | None = None,
                 output_softplus_initial_bias: float | None = None):
        super().__init__()
        if input_shape is None:
            raise ValueError("DeepONet requires input_shape")
        self.input_shape = tuple(input_shape)
        _register_grid_coordinates(self, grid_coordinates, self.input_shape)
        self.width = width
        self.output_floor = output_floor
        self.output_softplus_beta = float(output_softplus_beta)
        self.output_softplus_scale = (None if output_softplus_scale is None
                                      else float(output_softplus_scale))
        self.material_function = material_function
        self.material_feature_dim = 1 if material_function is not None else 0
        self.branch_encoder = str(branch_encoder)
        if self.branch_encoder in ("frozen_mlp", "left_robin_mlp"):
            if branch_dim is None:
                raise ValueError(f"{self.branch_encoder} branch requires branch_dim")
            self.branch = _deep_mlp(branch_dim, width, hidden, layers)
        elif self.branch_encoder == "initial_cnn":
            self.branch = InitialConditionCNNBranch(
                len(self.input_shape) - 1, tuple(size - 2 for size in self.input_shape[1:]), width,
                channels=branch_channels, layers=branch_layers,
            )
        elif self.branch_encoder == "left_robin_cnn":
            if len(self.input_shape) != 2:
                raise ValueError("left_robin_cnn is only defined for 1D space-time fields")
            self.branch = LeftRobinCNNBranch(
                self.input_shape[0], width, channels=branch_channels, layers=branch_layers,
            )
        elif self.branch_encoder in ("left_robin_2d_mlp", "left_robin_2d_cnn"):
            if len(self.input_shape) != 3:
                raise ValueError(
                    f"{self.branch_encoder} is only defined for 2D space-time fields"
                )
            if self.branch_encoder == "left_robin_2d_mlp":
                self.branch = _deep_mlp(
                    int(self.input_shape[0] * self.input_shape[1]), width, hidden, layers
                )
            else:
                self.branch = LeftRobin2DCNNBranch(
                    self.input_shape[0], self.input_shape[1], width,
                    channels=branch_channels, layers=branch_layers,
                )
        else:
            raise ValueError(
                "branch_encoder must be 'frozen_mlp', 'left_robin_mlp', 'left_robin_cnn', "
                "'left_robin_2d_mlp', 'left_robin_2d_cnn', or 'initial_cnn'"
            )
        self.trunk_fourier = None
        trunk_coord_dim = int(coord_dim)
        if trunk_fourier_features:
            self.trunk_fourier = FixedFourierFeatures(
                coord_dim, frequency_count=trunk_fourier_dim,
                scale=trunk_fourier_scale, seed=trunk_fourier_seed,
            )
            trunk_coord_dim = self.trunk_fourier.output_dim
        self.trunk = _deep_mlp(trunk_coord_dim + self.material_feature_dim, width, hidden, layers)
        self.bias = nn.Parameter(torch.zeros(output_channels))
        self.apply(self._initialize)
        if output_softplus_initial_bias is not None:
            nn.init.constant_(self.bias, float(output_softplus_initial_bias))

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def _packed_field(self, branch_input: torch.Tensor) -> torch.Tensor | None:
        expected_dimensions = len(self.input_shape) + 1
        if branch_input.ndim == expected_dimensions + 1:
            if branch_input.shape[1] != 1:
                raise ValueError("DeepONet packed branch input must have one channel")
            branch_input = branch_input[:, 0]
        if branch_input.ndim != expected_dimensions:
            return None
        if tuple(branch_input.shape[1:]) != self.input_shape:
            raise ValueError(
                f"DeepONet packed branch input has shape {tuple(branch_input.shape[1:])}, "
                f"expected {self.input_shape}"
            )
        return branch_input

    def _branch_features(self, branch_input: torch.Tensor, g_left=None) -> torch.Tensor:
        if self.branch_encoder in ("left_robin_2d_mlp", "left_robin_2d_cnn"):
            if g_left is None:
                raise ValueError(
                    f"{self.branch_encoder} requires the separate g_left condition"
                )
            if g_left.ndim == 4 and g_left.shape[1] == 1:
                g_left = g_left[:, 0]
            expected = (branch_input.shape[0], self.input_shape[0], self.input_shape[1])
            if g_left.ndim != 3 or tuple(g_left.shape) != expected:
                raise ValueError(
                    f"g_left must have shape {expected}, got {tuple(g_left.shape)}"
                )
            branch_values = (
                g_left.reshape(g_left.shape[0], -1)
                if self.branch_encoder == "left_robin_2d_mlp" else g_left
            )
            return self.branch(branch_values)
        field = self._packed_field(branch_input)
        if self.branch_encoder == "initial_cnn":
            if field is None:
                raise ValueError("initial_cnn branch requires a packed space-time condition")
            return self.branch(extract_initial_field(field))
        if field is not None:
            if self.branch_encoder in ("left_robin_mlp", "left_robin_cnn"):
                if field.ndim != 3:
                    raise ValueError(f"{self.branch_encoder} branch supports only 1D space-time fields")
                # The left packed boundary slot stores the Robin right-hand
                # side g_L(t); constant initial/right conditions are omitted.
                branch_input = field[:, :, 0]
            else:
                branch_input = extract_frozen_values(field)
        if branch_input.ndim != 2:
            raise ValueError(
                f"{self.branch_encoder} branch requires packed conditions or flattened branch values"
            )
        return self.branch(branch_input)

    def _material_features(self, coords: torch.Tensor) -> torch.Tensor:
        if self.material_function is None:
            return coords.new_empty((*coords.shape[:-1], 0))
        value = self.material_function(coords)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=coords.device, dtype=coords.dtype)
        else:
            value = value.to(device=coords.device, dtype=coords.dtype)
        if value.ndim == 0:
            value = torch.ones(
                (*coords.shape[:-1], 1), device=coords.device, dtype=coords.dtype
            ) * value
        elif value.ndim == coords.ndim - 1:
            value = value.unsqueeze(-1)
        expected = (*coords.shape[:-1], 1)
        if value.shape[0] == 1 and coords.shape[0] != 1:
            value = value.expand(coords.shape[0], -1, -1)
        if value.shape != expected:
            raise ValueError(
                f"material feature has shape {tuple(value.shape)}, expected {expected}"
            )
        return value

    def forward(self, branch_input, timestep=None, coords=None, g_left=None):
        if coords is None:
            field = self._packed_field(branch_input)
            if field is None:
                raise ValueError("coords are required when branch_input is already flattened")
            shape = self.input_shape
            axes = _grid_coordinates(self, field)
            if axes is None:
                axes = [torch.linspace(-1, 1, size, device=field.device) for size in shape]
            coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, -1, len(shape))
            coords = coords.expand(field.shape[0], -1, -1)
        branch = self._branch_features(branch_input, g_left=g_left)[:, None]
        trunk_coords = coords if self.trunk_fourier is None else self.trunk_fourier(coords)
        trunk_input = torch.cat((trunk_coords, self._material_features(coords)), dim=-1)
        trunk = self.trunk(trunk_input)
        output = (branch * trunk).sum(dim=-1, keepdim=True) / math.sqrt(self.width) + self.bias
        return _positive_output(
            output, self.output_floor, self.output_softplus_beta,
            self.output_softplus_scale,
        )
