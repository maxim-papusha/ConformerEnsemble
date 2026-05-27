from __future__ import annotations

import numpy as np
from conformerensemble import ConformerEnsemble, Geometry, backend_name
from conformerensemblepy import (
    ConformerEnsemble as LegacyConformerEnsemble,
)
from conformerensemblepy import (
    Geometry as LegacyGeometry,
)
from conformerensemblepy import (
    backend_name as legacy_backend_name,
)
from conformerensemblers import ConformerEnsemble as RSConformerEnsemble
from conformerensemblers import Geometry as RSGeometry


def make_geometry(offset: float = 0.0) -> Geometry:
    return Geometry(
        atom_types=["C", "H"],
        coords=[[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]],
    )


def test_backend_imports() -> None:
    assert backend_name() == "rust"
    assert legacy_backend_name() == "rust"


def test_canonical_types_point_to_rs() -> None:
    assert Geometry is RSGeometry
    assert ConformerEnsemble is RSConformerEnsemble


def test_legacy_package_reexports_canonical_types() -> None:
    assert LegacyGeometry is Geometry
    assert LegacyConformerEnsemble is ConformerEnsemble


def test_sort_descending() -> None:
    ensemble = ConformerEnsemble(
        geos=[make_geometry(0.0), make_geometry(2.0), make_geometry(4.0)],
        energies=[0.1, 0.3, 0.2],
    )

    ensemble.sort(descending=True)

    assert ensemble.energies == [0.3, 0.2, 0.1]


def test_boltzmann_weights_sum_to_one() -> None:
    ensemble = ConformerEnsemble(
        geos=[make_geometry(0.0), make_geometry(2.0)],
        energies=[0.0, 0.001],
    )

    weights = ensemble.get_boltzmann_weight(slice(None), temperature=298.15)

    assert np.isclose(np.sum(weights), 1.0)
    assert weights[0] > weights[1]


def test_geometry_xyz_string() -> None:
    xyz_output = make_geometry().xyz_str(comment="draft")

    assert xyz_output.startswith("2\ndraft\n")
    assert "C 0.00000000 0.00000000 0.00000000" in xyz_output
