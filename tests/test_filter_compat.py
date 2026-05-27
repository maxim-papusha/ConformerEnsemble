"""chemtrayzer-compatible tests for the .filter() interface.

These tests mirror chemtrayzer's `ConfFilterOptions` / `ConformerEnsemble.filter`
behaviour to ensure the Rust-backed wrapper is a drop-in replacement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conformerensemble import (
    ConfDiffOptions,
    ConfFilterOptions,
    ConformerEnsemble,
    GeoConfDiffOptions,
    Geometry,
)


def _two_atom(offset: float) -> Geometry:
    return Geometry(
        atom_types=["C", "H"],
        coords=[[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]],
    )


def _h2o(displacement: float = 0.0) -> Geometry:
    return Geometry(
        atom_types=["O", "H", "H"],
        coords=[
            [0.0 + displacement, 0.0, 0.0],
            [0.9572, 0.0, 0.0 + displacement],
            [-0.2399872, 0.92662721, 0.0],
        ],
    )


# ---------------------------------------------------------------------------
# Options classes — chemtrayzer-compatible field names and inheritance


def test_geo_conf_diff_options_defaults():
    o = GeoConfDiffOptions()
    assert o.rmsd_threshold == 0.125
    assert o.rot_threshold == 1.0
    assert o.mass_weighted is False
    assert o.rigid_transform is True


def test_conf_diff_options_inherits_geo_fields():
    o = ConfDiffOptions()
    assert o.rmsd_threshold == 0.125
    assert o.energy_threshold == pytest.approx(0.00038)
    assert o.mirror_check is False
    assert isinstance(o, GeoConfDiffOptions)


def test_conf_filter_options_full_chain():
    o = ConfFilterOptions()
    # GeoConfDiffOptions fields
    assert o.rmsd_threshold == 0.125
    assert o.rot_threshold == 1.0
    assert o.mass_weighted is False
    assert o.rigid_transform is True
    # ConfDiffOptions fields
    assert o.energy_threshold == pytest.approx(0.00038)
    assert o.mirror_check is False
    # ConfFilterOptions fields
    assert o.temperature == 1500.0
    assert o.cum_boltzmann_threshold == 1.0
    assert math.isinf(o.energy_window)
    assert isinstance(o, ConfDiffOptions)
    assert isinstance(o, GeoConfDiffOptions)


def test_conf_filter_options_kwargs_set_all_levels():
    o = ConfFilterOptions(
        rmsd_threshold=0.5,
        rot_threshold=2.5,
        energy_threshold=1e-3,
        mirror_check=True,
        temperature=300.0,
        cum_boltzmann_threshold=0.9,
        energy_window=0.01,
    )
    assert o.rmsd_threshold == 0.5
    assert o.rot_threshold == 2.5
    assert o.energy_threshold == pytest.approx(1e-3)
    assert o.mirror_check is True
    assert o.temperature == 300.0
    assert o.cum_boltzmann_threshold == 0.9
    assert o.energy_window == pytest.approx(0.01)


def test_conf_filter_options_fields_are_writable():
    o = ConfFilterOptions()
    o.rmsd_threshold = 0.4
    o.energy_threshold = 5e-4
    o.temperature = 400.0
    o.energy_window = 0.05
    assert o.rmsd_threshold == 0.4
    assert o.energy_threshold == pytest.approx(5e-4)
    assert o.temperature == 400.0
    assert o.energy_window == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# .filter() behaviour


def test_filter_with_default_options_returns_new_ensemble():
    ensemble = ConformerEnsemble(
        geos=[_two_atom(0.0), _two_atom(0.0), _two_atom(5.0)],
        energies=[0.0, 0.0, 0.1],
    )

    filtered = ensemble.filter()

    # default returns a copy (inplace=False)
    assert filtered is not ensemble
    # 3 input → 2 conformers (first two are geometrically identical and within
    # the default energy_threshold)
    assert len(filtered) == 2
    # original ensemble untouched
    assert len(ensemble) == 3


def test_filter_inplace_mutates_self_and_returns_none():
    ensemble = ConformerEnsemble(
        geos=[_two_atom(0.0), _two_atom(0.0), _two_atom(5.0)],
        energies=[0.0, 0.0, 0.1],
    )

    result = ensemble.filter(inplace=True)

    assert result is None
    assert len(ensemble) == 2


def test_filter_sorts_ascending_before_filtering():
    # Provide energies in descending order
    ensemble = ConformerEnsemble(
        geos=[_two_atom(5.0), _two_atom(0.0)],
        energies=[0.2, 0.0],
    )

    filtered = ensemble.filter()
    energies = list(filtered.energies)
    assert energies == sorted(energies)
    assert energies[0] == 0.0


def test_filter_energy_window_drops_high_energy_conformers():
    ensemble = ConformerEnsemble(
        geos=[_two_atom(0.0), _two_atom(5.0), _two_atom(10.0)],
        energies=[0.0, 0.005, 0.5],
    )
    # Default options: rmsd_threshold=0.125 → all 3 geometries are different
    opts = ConfFilterOptions(energy_window=0.01)

    filtered = ensemble.filter(opts)

    # only the 0.0 and 0.005 conformers survive (0.5 - 0.0 > 0.01)
    assert len(filtered) == 2
    assert list(filtered.energies) == pytest.approx([0.0, 0.005])


def test_filter_deduplicates_identical_geometries():
    ensemble = ConformerEnsemble(
        geos=[_two_atom(0.0)] * 4,
        energies=[0.0, 1e-6, 2e-6, 3e-6],
    )

    filtered = ensemble.filter()

    assert len(filtered) == 1
    assert filtered.energies[0] == 0.0


def test_filter_keeps_geometrically_distinct_conformers():
    ensemble = ConformerEnsemble(
        geos=[_two_atom(0.0), _two_atom(2.0), _two_atom(5.0)],
        energies=[0.0, 0.01, 0.02],
    )

    filtered = ensemble.filter()

    assert len(filtered) == 3


def test_filter_options_argument_is_not_mutated():
    opts = ConfFilterOptions(rmsd_threshold=0.2, energy_window=0.05)
    ensemble = ConformerEnsemble(
        geos=[_two_atom(0.0), _two_atom(5.0)],
        energies=[0.0, 0.001],
    )

    ensemble.filter(opts)

    assert opts.rmsd_threshold == 0.2
    assert opts.energy_window == pytest.approx(0.05)


def test_filter_empty_ensemble_returns_empty():
    ensemble = ConformerEnsemble()
    filtered = ensemble.filter()
    assert len(filtered) == 0


def test_filter_single_conformer_returns_single():
    ensemble = ConformerEnsemble(
        geos=[_h2o()],
        energies=[-76.4],
    )
    filtered = ensemble.filter()
    assert len(filtered) == 1
    assert filtered.energies[0] == pytest.approx(-76.4)


def test_filter_cum_boltzmann_threshold_truncates_population():
    # Two well-separated geometries with a large energy gap so the lower-energy
    # conformer carries essentially all Boltzmann weight at room temperature.
    ensemble = ConformerEnsemble(
        geos=[_two_atom(0.0), _two_atom(5.0)],
        energies=[0.0, 0.05],  # ~131 kJ/mol gap
    )

    opts = ConfFilterOptions(
        temperature=298.15,
        cum_boltzmann_threshold=0.5,
    )

    filtered = ensemble.filter(opts)

    # At 298 K, the 0.05 Hartree-higher conformer is suppressed by exp(-50)
    # → only the ground state is needed to exceed the 0.5 cumulative weight.
    assert len(filtered) == 1
    assert filtered.energies[0] == 0.0


# ---------------------------------------------------------------------------
# Boltzmann weights & __getitem__ semantics


def test_boltzmann_weight_int_index_returns_scalar():
    ensemble = ConformerEnsemble(
        geos=[_two_atom(0.0), _two_atom(2.0)],
        energies=[0.0, 0.001],
    )
    w0 = np.asarray(ensemble.get_boltzmann_weight(0, temperature=298.15))
    w1 = np.asarray(ensemble.get_boltzmann_weight(1, temperature=298.15))
    assert float(w0.sum()) > float(w1.sum())


def test_getitem_int_returns_geometry_energy_tuple():
    ensemble = ConformerEnsemble(
        geos=[_two_atom(0.0), _two_atom(2.0)],
        energies=[0.1, 0.2],
    )
    geo, energy = ensemble[0]
    assert isinstance(geo, Geometry)
    assert energy == pytest.approx(0.1)


def test_getitem_slice_returns_sub_ensemble():
    ensemble = ConformerEnsemble(
        geos=[_two_atom(0.0), _two_atom(2.0), _two_atom(4.0)],
        energies=[0.1, 0.2, 0.3],
    )
    sub = ensemble[1:]
    assert isinstance(sub, ConformerEnsemble)
    assert len(sub) == 2
    assert list(sub.energies) == pytest.approx([0.2, 0.3])


def test_xyz_round_trip_preserves_atom_symbols_and_energies():
    ensemble = ConformerEnsemble(
        geos=[_h2o(0.0), _h2o(0.5)],
        energies=[-76.41, -76.40],
    )
    text = ensemble.xyz_str()
    restored = ConformerEnsemble.from_xyz_str(text)
    assert len(restored) == 2
    assert list(restored.energies) == pytest.approx([-76.41, -76.40])
    assert restored.geometry(0).atom_symbols() == ["O", "H", "H"]
