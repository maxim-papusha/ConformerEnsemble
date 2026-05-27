#![cfg_attr(feature = "portable-simd", feature(portable_simd))]

//! Core data structures for conformer ensemble manipulation.
//!
//! This crate provides pure-Rust types for representing molecular geometries
//! and ensembles of conformers. Atomic coordinates are stored in
//! [`ndarray::Array2<f64>`] of shape `(n_atoms, 3)` in **column-major
//! (Fortran) order**. The backing buffer of each per-conformer array is
//! therefore `[x0..x_{n-1} | y0..y_{n-1} | z0..z_{n-1}]`, which matches the
//! SoA stream layout the QCP RMSD kernel consumes — so the hot path skips
//! any AoS→SoA scratch copy. On the Python side this maps directly onto an
//! `np.asfortranarray((n_atoms, 3))` `float64` view (zero-copy through
//! crates such as [`numpy`](https://docs.rs/numpy)).
//! The public interface intentionally stays in `(n_atoms, 3)` array form;
//! the internal SoA streams are an optimization detail, not a caller-facing API.
//!
//! Coordinates passed to constructors in row-major (C) order are silently
//! re-laid into column-major order on insert.
//!
//! # Example
//!
//! ```
//! use conformerensemblers::{ConformerEnsemble, Geometry};
//! use ndarray::array;
//!
//! // Atomic numbers: C = 6, H = 1.
//! let geo = Geometry::new(
//!     vec![6, 1],
//!     array![[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]],
//! )
//! .unwrap();
//!
//! let ensemble = ConformerEnsemble::new(
//!     geo.atom_types().to_vec(),
//!     vec![geo.into_coords()],
//!     vec![0.0],
//! )
//! .unwrap();
//! assert_eq!(ensemble.n_conformers(), 1);
//! ```

mod drmsd;
mod elements;
mod ensemble;
mod error;
mod filter;
mod geometry;
mod inertia;
mod rmsd;
mod xyz_io;

#[cfg(feature = "portable-simd")]
mod rmsd_simd;
#[cfg(feature = "portable-simd")]
mod drmsd_simd;

/// Per-(scalar, lane-width) QCP SIMD kernels exposed for benchmarking.
/// Stable Rust users will not see this module.
#[cfg(feature = "portable-simd")]
pub mod simd_kernels {
    pub use crate::rmsd_simd::*;
}

#[cfg(feature = "python")]
mod python;

pub use drmsd::{drmsd, pairwise_distances, DrmsdWorkspace};
pub use elements::{atomic_weight, atomic_weights};
pub use ensemble::ConformerEnsemble;
pub use error::ConformerError;
pub use filter::{
    filter_ensemble, is_different_conformer, is_different_geometry,
    min_rmsd_over_mappings, rmsd_permuted, validate_permutation,
    ConfDiffOptions, ConfFilterOptions, ValidatedMappings,
};
pub use geometry::{AtomType, Geometry};
pub use inertia::{
    moment_of_inertia_eigvals, rotational_constants_hz, RotationalConstants,
};
pub use rmsd::{
    qcp_rmsd, QcpFloat, QcpRmsdWorkspace, QcpRmsdWorkspaceF32,
    QcpRmsdWorkspaceT,
};

/// Boltzmann constant in J/K.
pub const BOLTZMANN_CONSTANT_J_PER_K: f64 = 1.380_649e-23;

/// Hartree energy in joules. Used to convert atomic-unit energies to SI for
/// Boltzmann weighting.
pub const HARTREE_ENERGY_J: f64 = 4.359_744_722_207_1e-18;
