"""Canonical experiment-family names shared by training and evaluation."""

from __future__ import annotations

CFDM_METHODS = ("c_fdm_unet", "c_fdm_vit")
PIFDM_METHODS = ("pifdm_unet", "pifdm_vit")
CONTINUOUS_DIFFUSION_METHODS = (*CFDM_METHODS, *PIFDM_METHODS)
DIFFCONPHY_BASE_METHODS = (
    "fdm_unet", "fdm_vit", *CONTINUOUS_DIFFUSION_METHODS,
)
_LEGACY_METHOD_ALIASES = {
    "pi_deeponet": "pideeponet",
}


def canonical_method_name(method: str) -> str:
    """Map serialized pre-refactor method names to their canonical names."""
    return _LEGACY_METHOD_ALIASES.get(method, method)


def validate_diffconphy_base_method(method: str) -> None:
    if method not in DIFFCONPHY_BASE_METHODS:
        raise ValueError(
            f"DiffConPhy checkpoint_method must be one of {DIFFCONPHY_BASE_METHODS}, "
            f"got {method!r}"
        )
