"""Data APIs, lazily imported so numerical generation does not need PyTorch."""

__all__ = [
    "FieldDataset", "ConditionedFieldDataset", "load_hdf5_field",
    "load_radiative_2d_field",
]


def __getattr__(name):
    if name in __all__:
        from .datasets import (
            ConditionedFieldDataset, FieldDataset, load_hdf5_field,
            load_radiative_2d_field,
        )
        return {
            "FieldDataset": FieldDataset,
            "ConditionedFieldDataset": ConditionedFieldDataset,
            "load_hdf5_field": load_hdf5_field,
            "load_radiative_2d_field": load_radiative_2d_field,
        }[name]
    raise AttributeError(name)
