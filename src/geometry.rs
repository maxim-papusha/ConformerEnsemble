use ndarray::{Array2, ArrayView2, ShapeBuilder};
use std::fs;
use std::path::Path;

use crate::error::ConformerError;
use crate::rmsd::{qcp_rmsd, QcpRmsdWorkspace};
use crate::xyz_io::{parse_xyz_blocks, write_xyz_block};

/// Returns `true` when `coords` is a column-major contiguous `(n, 3)` array.
///
/// A Fortran-ordered `(n, 3)` array has memory layout `[x0..x_{n-1} |
/// y0..y_{n-1} | z0..z_{n-1}]` — exactly the SoA stream layout the QCP
/// kernel consumes, so `as_slice_memory_order()` returns the SoA buffer
/// with no further work.
#[inline]
pub(crate) fn is_fortran_layout(coords: &Array2<f64>) -> bool {
    let n = coords.nrows() as isize;
    coords.strides() == [1, n]
}

/// Re-lays `coords` as a column-major `(n_atoms, 3)` array if it is not
/// already in that layout. Column-major matches the SoA layout used by the
/// QCP RMSD kernel, allowing the hot path to skip the AoS→SoA scratch copy.
pub(crate) fn into_fortran_layout(coords: Array2<f64>) -> Array2<f64> {
    if is_fortran_layout(&coords) {
        return coords;
    }

    let (n, _) = coords.dim();
    let mut buf = Vec::with_capacity(3 * n);
    for col in 0..3 {
        for row in 0..n {
            buf.push(coords[[row, col]]);
        }
    }
    Array2::from_shape_vec((n, 3).f(), buf)
        .expect("3*n elements fit a column-major (n,3) array")
}

/// Allocates a column-major `(n_atoms, 3)` zero array.
#[inline]
pub(crate) fn zeros_fortran(n_atoms: usize) -> Array2<f64> {
    Array2::<f64>::zeros((n_atoms, 3).f())
}

/// Atomic number used to identify an atom type.
///
/// Using `u8` covers the entire periodic table (1..=118) and maps directly
/// onto NumPy `uint8` arrays for efficient Python interop.
pub type AtomType = u8;

/// Molecular geometry: an ordered list of atomic numbers paired with a
/// coordinate matrix of shape `(n_atoms, 3)`.
///
/// Coordinates are stored in an [`ndarray::Array2<f64>`] in **column-major
/// (Fortran) order** so that the backing buffer is the SoA layout
/// `[x | y | z]` directly. This lets the QCP RMSD kernel consume the data
/// without an AoS\u2192SoA scratch copy, and maps directly onto a Python-side
/// `np.asfortranarray((n_atoms, 3))` zero-copy view. Coordinates supplied
/// in row-major (C) order are silently re-laid on insert.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    atom_types: Vec<AtomType>,
    coords: Array2<f64>,
}

impl Geometry {
    /// Creates a new geometry, validating shape and atom count.
    pub fn new(
        atom_types: Vec<AtomType>,
        coords: Array2<f64>,
    ) -> Result<Self, ConformerError> {
        let (rows, cols) = coords.dim();

        if cols != 3 {
            return Err(ConformerError::InvalidCoordShape { rows, cols });
        }

        if atom_types.len() != rows {
            return Err(ConformerError::AtomCountMismatch {
                expected: atom_types.len(),
                found: rows,
            });
        }

        Ok(Self {
            atom_types,
            coords: into_fortran_layout(coords),
        })
    }

    /// Creates an empty geometry with zero atoms.
    pub fn empty() -> Self {
        Self {
            atom_types: Vec::new(),
            coords: zeros_fortran(0),
        }
    }

    /// Returns the number of atoms in this geometry.
    pub fn n_atoms(&self) -> usize {
        self.atom_types.len()
    }

    /// Returns true when this geometry has no atoms.
    pub fn is_empty(&self) -> bool {
        self.atom_types.is_empty()
    }

    /// Returns the atomic numbers in declaration order.
    pub fn atom_types(&self) -> &[AtomType] {
        &self.atom_types
    }

    /// Returns a read-only view over the `(n_atoms, 3)` coordinate matrix.
    pub fn coords(&self) -> ArrayView2<'_, f64> {
        self.coords.view()
    }

    /// Returns ownership of the coordinate matrix.
    pub fn into_coords(self) -> Array2<f64> {
        self.coords
    }

    /// Returns the geometry mirrored across the `y-z` plane (sign of the
    /// `x` coordinate flipped).
    pub fn mirror_geometry(&self) -> Self {
        let mut mirrored = self.coords.clone();
        mirrored.column_mut(0).mapv_inplace(|x| -x);
        Self {
            atom_types: self.atom_types.clone(),
            coords: mirrored,
        }
    }

    /// Returns a standard XYZ-format string for this geometry. `comment`
    /// becomes the second line; an empty string is written when `None`.
    ///
    /// Atom labels are written as atomic numbers; downstream code may map
    /// them to element symbols if required.
    pub fn xyz_str(&self, comment: Option<&str>) -> String {
        let mut output = String::new();
        write_xyz_block(
            &mut output,
            &self.atom_types,
            self.coords(),
            comment.unwrap_or(""),
        );

        output
    }

    /// Writes this geometry as a single-block XYZ file.
    pub fn to_xyz_file<P: AsRef<Path>>(
        &self,
        path: P,
        comment: Option<&str>,
    ) -> Result<(), ConformerError> {
        fs::write(path, self.xyz_str(comment))
            .map_err(|err| ConformerError::IoError(err.to_string()))
    }

    /// Parses exactly one geometry from an XYZ string.
    pub fn from_xyz_str(input: &str) -> Result<Self, ConformerError> {
        let mut geometries = Self::from_xyz_str_many(input)?;
        if geometries.len() != 1 {
            return Err(ConformerError::ExpectedSingleGeometry {
                found: geometries.len(),
            });
        }
        Ok(geometries.remove(0))
    }

    /// Parses one or more geometries from a multi-block XYZ string.
    pub fn from_xyz_str_many(input: &str) -> Result<Vec<Self>, ConformerError> {
        let blocks = parse_xyz_blocks(input)?;
        let mut out = Vec::with_capacity(blocks.len());
        for block in blocks {
            out.push(Self::new(block.atom_types, block.coords)?);
        }
        Ok(out)
    }

    /// Parses exactly one geometry from an XYZ file.
    pub fn from_xyz_file<P: AsRef<Path>>(path: P) -> Result<Self, ConformerError> {
        let input = fs::read_to_string(path)
            .map_err(|err| ConformerError::IoError(err.to_string()))?;
        Self::from_xyz_str(&input)
    }

    /// Parses one or more geometries from a multi-block XYZ file.
    pub fn from_xyz_file_many<P: AsRef<Path>>(
        path: P,
    ) -> Result<Vec<Self>, ConformerError> {
        let input = fs::read_to_string(path)
            .map_err(|err| ConformerError::IoError(err.to_string()))?;
        Self::from_xyz_str_many(&input)
    }

    /// Writes multiple geometries to a single multi-block XYZ file.
    pub fn to_xyz_file_many<P: AsRef<Path>>(
        geometries: &[Self],
        path: P,
    ) -> Result<(), ConformerError> {
        let mut output = String::new();
        for geometry in geometries {
            write_xyz_block(
                &mut output,
                &geometry.atom_types,
                geometry.coords(),
                "",
            );
        }
        fs::write(path, output).map_err(|err| ConformerError::IoError(err.to_string()))
    }

    /// Computes optimal rigid-body RMSD to another geometry using the
    /// Theobald/QCP quaternion formulation.
    pub fn qcp_rmsd(&self, other: &Self) -> Result<f64, ConformerError> {
        if self.atom_types != other.atom_types {
            return Err(ConformerError::InconsistentAtomTypes);
        }
        qcp_rmsd(self.coords(), other.coords())
    }

    /// Computes optimal rigid-body RMSD using a reusable QCP workspace.
    pub fn qcp_rmsd_with_workspace(
        &self,
        other: &Self,
        workspace: &mut QcpRmsdWorkspace,
    ) -> Result<f64, ConformerError> {
        if self.atom_types != other.atom_types {
            return Err(ConformerError::InconsistentAtomTypes);
        }
        workspace.rmsd(self.coords(), other.coords())
    }
}
