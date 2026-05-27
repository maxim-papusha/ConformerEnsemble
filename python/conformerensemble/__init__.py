from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._api import (
    ConfDiffOptions,
    ConfFilterOptions,
    ConformerEnsemble,
    GeoConfDiffOptions,
    Geometry,
    backend_name,
)

try:
    __version__ = version("ConformerEnsemble")
except PackageNotFoundError:
    __version__ = "0.1.0a0"

__all__ = [
    "ConfDiffOptions",
    "ConfFilterOptions",
    "ConformerEnsemble",
    "GeoConfDiffOptions",
    "Geometry",
    "backend_name",
    "__version__",
]
