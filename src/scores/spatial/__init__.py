"""
Import the functions from the implementations into the public API
"""

import importlib

from scores.spatial.fss_impl import fss_2d, fss_2d_binary, fss_2d_single_field

__all__ = ["fss_2d", "fss_2d_binary", "fss_2d_single_field", "pfss_for_ensemble"]


def __getattr__(name: str):
    # Loaded lazily to avoid a circular import (scores.probability.pfss_impl imports from scores.spatial)
    if name == "pfss_for_ensemble":
        return importlib.import_module("scores.probability.pfss_impl").pfss_for_ensemble
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
