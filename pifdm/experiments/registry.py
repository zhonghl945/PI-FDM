from dataclasses import dataclass
from typing import Callable, Dict


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    dimension: int
    equation: str
    data_generator: str
    residual: str
    boundary_slots: str


CASES: Dict[str, Case] = {
    "linear_1d": Case("linear_1d", "one-dimensional linear diffusion", 1,
                       "u_t - D u_xx = f", "pifdm.cases.linear_1d", "linear_diffusion", "mixed"),
    "radiative_1d": Case("radiative_1d", "one-dimensional nonlinear radiative diffusion", 1,
                          "E_t + div(-D grad E) - S = 0", "pifdm.cases.radiative_1d",
                          "radiative_diffusion", "operator"),
    "radiative_2d": Case("radiative_2d", "two-dimensional heterogeneous radiative diffusion", 2,
                          "E_t + div(-D grad E) - S = 0", "pifdm.cases.radiative_2d",
                          "radiative_diffusion", "operator"),
    "radiative_initial_2d": Case(
        "radiative_initial_2d", "initial-driven two-dimensional heterogeneous radiative diffusion", 2,
        "E_t + div(-D grad E) - S = 0", "pifdm.cases.radiative_initial_2d",
        "radiative_diffusion", "operator",
    ),
}


def get_case(name: str) -> Case:
    try:
        return CASES[name]
    except KeyError as exc:
        raise ValueError(f"unknown PDE case {name!r}; expected {tuple(CASES)}") from exc
