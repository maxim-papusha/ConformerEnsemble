//! Moment of inertia and rotational constants.
//!
//! Coordinates are assumed to be in Ångström and masses in unified atomic
//! mass units (u). Rotational constants are returned in Hz, matching the
//! Python `Geometry.rotational_constants()` convention.

use ndarray::ArrayView2;

use crate::error::ConformerError;

/// Reduced Planck constant `ℏ` [J·s].
const HBAR_J_S: f64 = 1.054_571_817e-34;
/// Atomic mass constant [kg].
const AMU_KG: f64 = 1.660_539_066_60e-27;
/// Ångström in metres.
const ANGSTROM_M: f64 = 1.0e-10;

/// Result of [`rotational_constants_hz`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RotationalConstants {
    /// Nonlinear molecule: three rotational constants in ascending order [Hz].
    Nonlinear([f64; 3]),
    /// Linear molecule: a single rotational constant [Hz].
    Linear(f64),
}

impl RotationalConstants {
    /// Returns the rotational constants as a slice in ascending order.
    pub fn as_slice(&self) -> &[f64] {
        match self {
            Self::Nonlinear(v) => v.as_slice(),
            Self::Linear(v) => std::slice::from_ref(v),
        }
    }
}

/// Computes the centre of mass over `coords` weighted by `masses`.
#[inline]
fn center_of_mass(coords: ArrayView2<'_, f64>, masses: &[f64]) -> [f64; 3] {
    let mut total = 0.0;
    let mut com = [0.0; 3];
    for (i, &m) in masses.iter().enumerate() {
        total += m;
        com[0] += m * coords[[i, 0]];
        com[1] += m * coords[[i, 1]];
        com[2] += m * coords[[i, 2]];
    }
    if total > 0.0 {
        com[0] /= total;
        com[1] /= total;
        com[2] /= total;
    }
    com
}

/// Returns the inertia-tensor eigenvalues in ascending order, in units of
/// `[Ångström]² · u`.
///
/// Mirrors the Python `Geometry.moment_of_inertia()`: builds the inertia
/// tensor about the centre of mass and diagonalises it.
pub fn moment_of_inertia_eigvals(
    coords: ArrayView2<'_, f64>,
    masses: &[f64],
) -> Result<[f64; 3], ConformerError> {
    let n = masses.len();
    if coords.nrows() != n {
        return Err(ConformerError::AtomCountMismatch {
            expected: n,
            found: coords.nrows(),
        });
    }

    let com = center_of_mass(coords, masses);

    let (mut ixx, mut iyy, mut izz) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut ixy, mut ixz, mut iyz) = (0.0_f64, 0.0_f64, 0.0_f64);

    for (i, &m) in masses.iter().enumerate() {
        let x = coords[[i, 0]] - com[0];
        let y = coords[[i, 1]] - com[1];
        let z = coords[[i, 2]] - com[2];
        ixx += m * (y * y + z * z);
        iyy += m * (x * x + z * z);
        izz += m * (x * x + y * y);
        ixy -= m * x * y;
        ixz -= m * x * z;
        iyz -= m * y * z;
    }

    let mut tensor = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]];
    let mut eigvals = jacobi_eigvals_sym3(&mut tensor);
    eigvals.sort_by(|a, b| a.total_cmp(b));
    Ok(eigvals)
}

/// Rotational constants in Hz, classifying the molecule as linear or
/// nonlinear (mirrors Python `Geometry.rotational_constants()`).
///
/// # Errors
/// Returns [`ConformerError::SingleAtomRotationalConstants`] if all three
/// inertia eigenvalues are essentially zero (i.e. a single atom).
/// Returns [`ConformerError::AtomCountMismatch`] if `coords` and `masses`
/// disagree on atom count.
pub fn rotational_constants_hz(
    coords: ArrayView2<'_, f64>,
    masses: &[f64],
) -> Result<RotationalConstants, ConformerError> {
    let eigvals = moment_of_inertia_eigvals(coords, masses)?;
    let n_zero = eigvals.iter().filter(|&&v| v.abs() < 1.0e-10).count();

    let factor = |moi: f64| HBAR_J_S / (4.0 * std::f64::consts::PI * moi * ANGSTROM_M * ANGSTROM_M * AMU_KG);

    match n_zero {
        0 => Ok(RotationalConstants::Nonlinear([
            factor(eigvals[0]),
            factor(eigvals[1]),
            factor(eigvals[2]),
        ])),
        1 => {
            // One eigenvalue is effectively zero; the other two should
            // be equal. Average them to match Python.
            let avg = (eigvals[0] + eigvals[1] + eigvals[2]) * 0.5;
            Ok(RotationalConstants::Linear(factor(avg)))
        }
        3 => Err(ConformerError::SingleAtomRotationalConstants),
        _ => Err(ConformerError::DegenerateInertiaTensor),
    }
}

/// Jacobi eigenvalue solver for a 3×3 real symmetric matrix.
///
/// Performs cyclic Jacobi sweeps until off-diagonal magnitudes drop below
/// a tight tolerance. Returns the three eigenvalues (unsorted).
fn jacobi_eigvals_sym3(a: &mut [[f64; 3]; 3]) -> [f64; 3] {
    const MAX_SWEEPS: usize = 50;
    const TOL: f64 = 1.0e-14;

    for _ in 0..MAX_SWEEPS {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < TOL {
            break;
        }
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() < TOL {
                continue;
            }
            let app = a[p][p];
            let aqq = a[q][q];
            let theta = (aqq - app) / (2.0 * apq);
            let t = if theta >= 0.0 {
                1.0 / (theta + (1.0 + theta * theta).sqrt())
            } else {
                1.0 / (theta - (1.0 + theta * theta).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            let r = 3 - p - q; // remaining index
            let arp = a[r][p];
            let arq = a[r][q];
            a[r][p] = c * arp - s * arq;
            a[p][r] = a[r][p];
            a[r][q] = s * arp + c * arq;
            a[q][r] = a[r][q];
        }
    }

    [a[0][0], a[1][1], a[2][2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn diagonal_already() {
        let mut m = [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]];
        let mut e = jacobi_eigvals_sym3(&mut m);
        e.sort_by(|a, b| a.total_cmp(b));
        assert!((e[0] - 1.0).abs() < 1e-12);
        assert!((e[1] - 2.0).abs() < 1e-12);
        assert!((e[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn known_symmetric() {
        // Eigenvalues of [[4,1,0],[1,4,0],[0,0,2]] are 5, 3, 2.
        let mut m = [[4.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 2.0]];
        let mut e = jacobi_eigvals_sym3(&mut m);
        e.sort_by(|a, b| a.total_cmp(b));
        assert!((e[0] - 2.0).abs() < 1e-12);
        assert!((e[1] - 3.0).abs() < 1e-12);
        assert!((e[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn linear_h2() {
        // Two H atoms 0.74 Å apart along x.
        let coords = Array2::from_shape_vec(
            (2, 3),
            vec![0.0, 0.0, 0.0, 0.74, 0.0, 0.0],
        )
        .unwrap();
        let masses = vec![1.008, 1.008];
        let r = rotational_constants_hz(coords.view(), &masses).unwrap();
        match r {
            RotationalConstants::Linear(_) => {}
            other => panic!("expected linear, got {other:?}"),
        }
    }
}
