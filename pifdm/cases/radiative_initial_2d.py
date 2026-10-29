"""Initial-condition-driven variant of the heterogeneous 2D radiative case."""

from .base import register_case
from .radiative_2d import Radiative2DCase


class RadiativeInitial2DCase(Radiative2DCase):
    name = "radiative_initial_2d"


register_case(RadiativeInitial2DCase())
