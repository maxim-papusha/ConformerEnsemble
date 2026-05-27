use thiserror::Error;

/// Errors that can be produced by this crate.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ConformerError {
    /// Filesystem I/O error.
    #[error("I/O error: {0}")]
    IoError(String),

    /// Generic XYZ parsing error.
    #[error("xyz parse error: {0}")]
    XyzParse(String),

    /// A routine expected a single geometry block but found a different
    /// count.
    #[error("expected exactly one geometry in xyz input, found {found}")]
    ExpectedSingleGeometry { found: usize },

    /// The coordinate array did not have shape `(n_atoms, 3)`.
    #[error("coords must have shape (n_atoms, 3), got ({rows}, {cols})")]
    InvalidCoordShape { rows: usize, cols: usize },

    /// Two inputs that should describe the same atom count disagree.
    #[error("atom counts do not match: expected {expected}, found {found}")]
    AtomCountMismatch { expected: usize, found: usize },

    /// The number of geometries did not match the number of energies.
    #[error(
        "number of geometries ({n_geos}) does not match number of \
         energies ({n_energies})"
    )]
    GeometryEnergyMismatch { n_geos: usize, n_energies: usize },

    /// Geometries in the ensemble disagree on atom composition.
    #[error("all geometries in an ensemble must share the same atom types")]
    InconsistentAtomTypes,

    /// Workspace and coordinate shape disagree.
    #[error(
        "workspace expects {workspace_n_atoms} atoms, got coordinates with {coords_n_atoms} atoms"
    )]
    WorkspaceAtomCountMismatch {
        workspace_n_atoms: usize,
        coords_n_atoms: usize,
    },

    /// A prepared-reference RMSD call was made before loading a reference.
    #[error("workspace reference coordinates have not been prepared")]
    WorkspaceReferenceNotPrepared,

    /// The temperature passed to a thermodynamic routine was not valid.
    #[error("temperature must be a finite positive number, got {0}")]
    InvalidTemperature(f64),

    /// Normalization of Boltzmann factors failed (e.g. all factors zero or
    /// non-finite).
    #[error("could not normalize Boltzmann factors")]
    BoltzmannNormalizationFailed,

    /// An atomic number is not in the supported range (1..=118).
    #[error("unknown atom type (atomic number {0}); must be in 1..=118")]
    UnknownAtomType(u8),

    /// All three inertia eigenvalues are zero (single atom). Rotational
    /// constants are not defined in this case.
    #[error("rotational constants are not defined for a single atom")]
    SingleAtomRotationalConstants,

    /// The inertia tensor has an unphysical eigenvalue pattern (e.g. two
    /// zero eigenvalues with one nonzero).
    #[error("inertia tensor has an unphysical eigenvalue pattern")]
    DegenerateInertiaTensor,

    /// A permutation passed to a filter routine has the wrong length.
    #[error(
        "permutation has length {permutation_len}, expected {expected_len}"
    )]
    InvalidPermutationLength {
        permutation_len: usize,
        expected_len: usize,
    },

    /// A permutation contains an out-of-range index.
    #[error(
        "permutation contains out-of-range index {index} (must be < {n_atoms})"
    )]
    InvalidPermutationIndex { index: usize, n_atoms: usize },

    /// A permutation maps atoms of incompatible types onto each other.
    #[error(
        "permutation maps atom {self_index} (type {self_type}) onto atom \
         {other_index} (type {other_type})"
    )]
    PermutationAtomTypeMismatch {
        self_index: usize,
        self_type: u8,
        other_index: usize,
        other_type: u8,
    },

    /// `is_different_conformer` was invoked with a finite `energy_threshold`
    /// but at least one energy was not supplied.
    #[error(
        "energies must be supplied when energy_threshold is finite"
    )]
    MissingEnergyForThreshold,
}
