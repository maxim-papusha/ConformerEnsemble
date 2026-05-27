use ndarray::{Array1, Array2, ArrayView2};
use std::fs;
use std::path::Path;

use crate::error::ConformerError;
use crate::geometry::{into_fortran_layout, AtomType, Geometry};
use crate::{BOLTZMANN_CONSTANT_J_PER_K, HARTREE_ENERGY_J};
use crate::xyz_io::{
    parse_energy_from_comment, parse_xyz_blocks, write_xyz_block,
};

/// A collection of conformers sharing a single set of atom types.
///
/// Storing the atom-type vector once and a `Vec<Array2<f64>>` of per-conformer
/// coordinate matrices avoids duplicating the chemistry across geometries.
///
/// **Memory layout**: each per-conformer coordinate matrix is stored in
/// **column-major (Fortran) order** with shape `(n_atoms, 3)`. Its backing
/// buffer is therefore `[x0..x_{n-1} | y0..y_{n-1} | z0..z_{n-1}]` —
/// directly the SoA streams the QCP RMSD kernel consumes, so the hot path
/// skips any AoS→SoA scratch copy. Conformers themselves are independent
/// allocations (one `Array2` per conformer); they need not be contiguous
/// with each other.
///
/// Coordinates passed to constructors in row-major (C) order are silently
/// re-laid into column-major order on insert.
///
/// Energies are stored in atomic units (Hartree) as `f64` by convention; this
/// precision matters for [`Self::boltzmann_weights`].
#[derive(Debug, Clone, PartialEq)]
pub struct ConformerEnsemble {
    atom_types: Vec<AtomType>,
    coords: Vec<Array2<f64>>,
    energies: Vec<f64>,
}

impl ConformerEnsemble {
    /// Builds a new ensemble after validating lengths and per-conformer
    /// coordinate shapes.
    pub fn new(
        atom_types: Vec<AtomType>,
        coords: Vec<Array2<f64>>,
        energies: Vec<f64>,
    ) -> Result<Self, ConformerError> {
        if coords.len() != energies.len() {
            return Err(ConformerError::GeometryEnergyMismatch {
                n_geos: coords.len(),
                n_energies: energies.len(),
            });
        }

        let n_atoms = atom_types.len();
        let coords: Vec<Array2<f64>> = coords
            .into_iter()
            .map(|c| {
                validate_conformer_shape(&c, n_atoms)?;
                Ok(into_fortran_layout(c))
            })
            .collect::<Result<_, ConformerError>>()?;

        Ok(Self {
            atom_types,
            coords,
            energies,
        })
    }

    /// Builds an empty ensemble with the given atom types and no conformers.
    pub fn with_atom_types(atom_types: Vec<AtomType>) -> Self {
        Self {
            atom_types,
            coords: Vec::new(),
            energies: Vec::new(),
        }
    }

    /// Number of atoms per conformer.
    pub fn n_atoms(&self) -> usize {
        self.atom_types.len()
    }

    /// Number of conformers stored in the ensemble.
    pub fn n_conformers(&self) -> usize {
        self.coords.len()
    }

    /// Returns true when this ensemble has no conformers.
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Atom types shared across the ensemble.
    pub fn atom_types(&self) -> &[AtomType] {
        &self.atom_types
    }

    /// Borrowed slice of all per-conformer coordinate matrices.
    pub fn coords(&self) -> &[Array2<f64>] {
        &self.coords
    }

    /// Read-only view of the coordinate matrix for a single conformer.
    pub fn conformer_coords(
        &self,
        index: usize,
    ) -> Option<ArrayView2<'_, f64>> {
        self.coords.get(index).map(|array| array.view())
    }

    /// Energies in current order.
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// Appends a coordinate matrix / energy pair, validating shape.
    ///
    /// Coordinates given in row-major (C) order are re-laid into
    /// column-major (Fortran) order on insert.
    pub fn push(
        &mut self,
        coords: Array2<f64>,
        energy: f64,
    ) -> Result<(), ConformerError> {
        validate_conformer_shape(&coords, self.n_atoms())?;
        self.coords.push(into_fortran_layout(coords));
        self.energies.push(energy);
        Ok(())
    }

    /// Appends a [`Geometry`], requiring its atom types to match the ensemble.
    pub fn push_geometry(
        &mut self,
        geometry: Geometry,
        energy: f64,
    ) -> Result<(), ConformerError> {
        if geometry.atom_types() != self.atom_types.as_slice() {
            return Err(ConformerError::InconsistentAtomTypes);
        }
        self.coords.push(geometry.into_coords());
        self.energies.push(energy);
        Ok(())
    }

    /// Constructs a [`Geometry`] for the conformer at `index`, cloning its
    /// coordinates and atom types.
    pub fn geometry(&self, index: usize) -> Option<Geometry> {
        let coords = self.coords.get(index)?.clone();
        // Shape is validated on insert, so `new` cannot fail here.
        Geometry::new(self.atom_types.clone(), coords).ok()
    }

    /// Sorts the ensemble in place by energy. With `descending = true`,
    /// the highest-energy conformer becomes the first entry; otherwise the
    /// lowest-energy conformer comes first. NaN energies are ordered using
    /// [`f64::total_cmp`].
    pub fn sort_by_energy(&mut self, descending: bool) {
        let mut order: Vec<usize> = (0..self.energies.len()).collect();
        order.sort_by(|&left, &right| {
            self.energies[left].total_cmp(&self.energies[right])
        });
        if descending {
            order.reverse();
        }

        let coords: Vec<Array2<f64>> =
            order.iter().map(|&i| self.coords[i].clone()).collect();
        let energies: Vec<f64> =
            order.iter().map(|&i| self.energies[i]).collect();

        self.coords = coords;
        self.energies = energies;
    }

    /// Returns Boltzmann weights for the stored energies at the given
    /// temperature. Energies are interpreted as Hartree; temperature is in
    /// Kelvin. The returned weights sum to one.
    pub fn boltzmann_weights(
        &self,
        temperature_k: f64,
    ) -> Result<Array1<f64>, ConformerError> {
        if !temperature_k.is_finite() || temperature_k <= 0.0 {
            return Err(ConformerError::InvalidTemperature(temperature_k));
        }

        if self.energies.is_empty() {
            return Ok(Array1::zeros(0));
        }

        let min_energy =
            self.energies.iter().copied().fold(f64::INFINITY, f64::min);

        let kt = BOLTZMANN_CONSTANT_J_PER_K * temperature_k;
        let factors: Array1<f64> = self
            .energies
            .iter()
            .map(|energy| {
                let delta_j = (energy - min_energy) * HARTREE_ENERGY_J;
                (-delta_j / kt).exp()
            })
            .collect();

        let sum: f64 = factors.sum();
        if !sum.is_finite() || sum == 0.0 {
            return Err(ConformerError::BoltzmannNormalizationFailed);
        }

        Ok(factors / sum)
    }

    /// Serializes the ensemble to a multi-block XYZ string.
    ///
    /// Each comment line stores `energy=<value>` for robust round-trips.
    pub fn to_xyz_str(&self) -> String {
        let mut output = String::new();
        for (coords, energy) in self.coords.iter().zip(self.energies.iter()) {
            write_xyz_block(
                &mut output,
                &self.atom_types,
                coords.view(),
                &format!("energy={:.16}", energy),
            );
        }
        output
    }

    /// Writes the ensemble to a multi-block XYZ file.
    pub fn to_xyz_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConformerError> {
        fs::write(path, self.to_xyz_str())
            .map_err(|err| ConformerError::IoError(err.to_string()))
    }

    /// Parses an ensemble from a multi-block XYZ string.
    ///
    /// Energies are read from comment lines when present as `energy=<value>`,
    /// `Energy=<value>`, `E=<value>`, `e=<value>`, or when the comment itself
    /// is a bare float. Missing energies default to `0.0`.
    pub fn from_xyz_str(input: &str) -> Result<Self, ConformerError> {
        let blocks = parse_xyz_blocks(input)?;

        let atom_types = blocks[0].atom_types.clone();
        let mut coords = Vec::with_capacity(blocks.len());
        let mut energies = Vec::with_capacity(blocks.len());

        for block in blocks {
            if block.atom_types != atom_types {
                return Err(ConformerError::InconsistentAtomTypes);
            }
            energies.push(parse_energy_from_comment(&block.comment).unwrap_or(0.0));
            coords.push(block.coords);
        }

        Self::new(atom_types, coords, energies)
    }

    /// Parses an ensemble from a multi-block XYZ file.
    pub fn from_xyz_file<P: AsRef<Path>>(path: P) -> Result<Self, ConformerError> {
        let input = fs::read_to_string(path)
            .map_err(|err| ConformerError::IoError(err.to_string()))?;
        Self::from_xyz_str(&input)
    }
}

fn validate_conformer_shape(
    coords: &Array2<f64>,
    expected_n_atoms: usize,
) -> Result<(), ConformerError> {
    let (rows, cols) = coords.dim();
    if cols != 3 {
        return Err(ConformerError::InvalidCoordShape { rows, cols });
    }
    if rows != expected_n_atoms {
        return Err(ConformerError::AtomCountMismatch {
            expected: expected_n_atoms,
            found: rows,
        });
    }
    Ok(())
}
