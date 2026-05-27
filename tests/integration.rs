use approx::assert_relative_eq;
use conformerensemblers::{
    atomic_weight, drmsd, filter_ensemble, is_different_conformer,
    min_rmsd_over_mappings, pairwise_distances, qcp_rmsd, rmsd_permuted,
    rotational_constants_hz, validate_permutation, ConfDiffOptions,
    ConfFilterOptions, ConformerEnsemble, ConformerError, DrmsdWorkspace,
    Geometry, QcpRmsdWorkspace, RotationalConstants, ValidatedMappings,
};
use ndarray::{array, s, Array2, ShapeBuilder};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

// Atomic numbers: C=6, H=1, O=8.
const METHANE_ATOM_TYPES: [u8; 5] = [6, 1, 1, 1, 1];

fn methane_coords() -> Array2<f64> {
    array![
        [0.0_f64, 0.0, 0.0],
        [0.629, 0.629, 0.629],
        [-0.629, -0.629, 0.629],
        [-0.629, 0.629, -0.629],
        [0.629, -0.629, -0.629],
    ]
}

fn methane() -> Geometry {
    Geometry::new(METHANE_ATOM_TYPES.to_vec(), methane_coords()).unwrap()
}

fn temp_xyz_path(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "conformerensemblers_{}_{}_{}.xyz",
        tag,
        std::process::id(),
        nanos
    ))
}

/// Pure rotation about z. The QCP kernel assumes inputs are pre-centered
/// to a common origin, so this helper intentionally adds **no** translation.
fn rotate_translate_z(coords: &Array2<f64>) -> Array2<f64> {
    let c: f64 = 0.5;
    let s: f64 = (1.0_f64 - c * c).sqrt();

    let mut out = Array2::<f64>::zeros((coords.nrows(), 3));
    for (i, row) in coords.rows().into_iter().enumerate() {
        let x = row[0];
        let y = row[1];
        let z = row[2];
        out[[i, 0]] = c * x - s * y;
        out[[i, 1]] = s * x + c * y;
        out[[i, 2]] = z;
    }
    out
}

#[test]
fn geometry_validates_shape() {
    let bad = Geometry::new(vec![6], array![[0.0_f64, 0.0]]);
    assert!(matches!(
        bad,
        Err(ConformerError::InvalidCoordShape { rows: 1, cols: 2 })
    ));
}

#[test]
fn geometry_validates_atom_count() {
    let bad = Geometry::new(vec![6, 1], array![[0.0_f64, 0.0, 0.0]]);
    assert!(matches!(
        bad,
        Err(ConformerError::AtomCountMismatch {
            expected: 2,
            found: 1
        })
    ));
}

#[test]
fn mirror_flips_x_axis() {
    let geo = methane();
    let mirrored = geo.mirror_geometry();
    for (orig, mirr) in
        geo.coords().rows().into_iter().zip(mirrored.coords().rows())
    {
        assert_relative_eq!(mirr[0], -orig[0]);
        assert_relative_eq!(mirr[1], orig[1]);
        assert_relative_eq!(mirr[2], orig[2]);
    }
}

#[test]
fn xyz_str_writes_element_symbols() {
    let xyz = methane().xyz_str(Some("methane"));
    let mut lines = xyz.lines();
    assert_eq!(lines.next(), Some("5"));
    assert_eq!(lines.next(), Some("methane"));
    assert!(lines.next().unwrap().starts_with("C "));
    // Remaining four lines are the methane hydrogens.
    for _ in 0..4 {
        assert!(lines.next().unwrap().starts_with("H "));
    }
}

#[test]
fn ensemble_rejects_mismatched_lengths() {
    let result = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![methane_coords()],
        vec![],
    );
    assert!(matches!(
        result,
        Err(ConformerError::GeometryEnergyMismatch {
            n_geos: 1,
            n_energies: 0
        })
    ));
}

#[test]
fn ensemble_rejects_bad_conformer_shape() {
    let result = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![array![[0.0_f64, 0.0, 0.0]]],
        vec![0.0],
    );
    assert!(matches!(
        result,
        Err(ConformerError::AtomCountMismatch {
            expected: 5,
            found: 1
        })
    ));
}

#[test]
fn push_geometry_requires_matching_atom_types() {
    let mut ensemble =
        ConformerEnsemble::with_atom_types(METHANE_ATOM_TYPES.to_vec());
    let water = Geometry::new(
        vec![8, 1, 1],
        array![
            [0.0_f64, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
    )
    .unwrap();
    assert_eq!(
        ensemble.push_geometry(water, 0.0),
        Err(ConformerError::InconsistentAtomTypes)
    );
}

#[test]
fn sort_descending_orders_high_to_low() {
    let ensemble = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![methane_coords(), methane_coords(), methane_coords()],
        vec![0.1, -0.2, 0.5],
    )
    .unwrap();
    let mut sorted = ensemble;
    sorted.sort_by_energy(true);
    assert_eq!(sorted.energies(), &[0.5, 0.1, -0.2]);
}

#[test]
fn boltzmann_weights_normalize_to_one() {
    let ensemble = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![methane_coords(), methane_coords(), methane_coords()],
        vec![0.0, 1.0e-3, 2.0e-3],
    )
    .unwrap();
    let weights = ensemble.boltzmann_weights(298.15).unwrap();
    assert_relative_eq!(weights.sum(), 1.0, epsilon = 1e-12);
    assert!(weights[0] > weights[1]);
    assert!(weights[1] > weights[2]);
}

#[test]
fn boltzmann_rejects_bad_temperature() {
    let ensemble = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![methane_coords()],
        vec![0.0],
    )
    .unwrap();
    assert!(matches!(
        ensemble.boltzmann_weights(-1.0),
        Err(ConformerError::InvalidTemperature(_))
    ));
}

#[test]
fn geometry_round_trips_through_ensemble() {
    let ensemble = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![methane_coords()],
        vec![0.0],
    )
    .unwrap();
    let geo = ensemble.geometry(0).unwrap();
    assert_eq!(geo.atom_types(), METHANE_ATOM_TYPES.as_slice());
    assert_eq!(geo.coords().to_owned(), methane_coords());
}

#[test]
fn qcp_rmsd_is_rigid_transform_invariant() {
    let base = methane_coords();
    let moved = rotate_translate_z(&base);
    let rmsd = qcp_rmsd(base.view(), moved.view()).unwrap();
    assert_relative_eq!(rmsd, 0.0, epsilon = 1e-5);
}

#[test]
fn qcp_workspace_can_be_reused() {
    let base = methane_coords();
    let moved = rotate_translate_z(&base);

    let mut workspace = QcpRmsdWorkspace::new(base.nrows());

    let rmsd1 = workspace.rmsd(base.view(), moved.view()).unwrap();
    assert_relative_eq!(rmsd1, 0.0, epsilon = 1e-5);

    let mut perturbed = moved.clone();
    perturbed[[0, 0]] += 0.02;
    let rmsd2 = workspace.rmsd(base.view(), perturbed.view()).unwrap();
    assert!(rmsd2 > 0.0);
}

#[test]
fn qcp_workspace_scalar_and_simd_paths_agree() {
    let base = methane_coords();
    let mut moved = rotate_translate_z(&base);
    moved[[0, 0]] += 0.013;

    let mut scalar = QcpRmsdWorkspace::with_simd(base.nrows(), false);
    let mut simd = QcpRmsdWorkspace::with_simd(base.nrows(), true);

    let scalar_rmsd = scalar.rmsd(base.view(), moved.view()).unwrap();
    let simd_rmsd = simd.rmsd(base.view(), moved.view()).unwrap();
    assert_relative_eq!(scalar_rmsd, simd_rmsd, epsilon = 1e-5);
}

#[test]
fn qcp_workspace_prepared_reference_matches_direct_path() {
    let base = methane_coords();
    let mut moved = rotate_translate_z(&base);
    moved[[1, 2]] += 0.009;

    let mut workspace = QcpRmsdWorkspace::new(base.nrows());
    let direct = workspace.rmsd(base.view(), moved.view()).unwrap();

    workspace.prepare_reference(base.view()).unwrap();
    let prepared = workspace.rmsd_prepared(moved.view()).unwrap();

    assert_relative_eq!(direct, prepared, epsilon = 1e-5);
}

#[test]
fn qcp_workspace_prepared_scalar_and_simd_paths_agree() {
    let base = methane_coords();
    let mut moved = rotate_translate_z(&base);
    moved[[2, 1]] -= 0.011;

    let mut scalar = QcpRmsdWorkspace::with_simd(base.nrows(), false);
    let mut simd = QcpRmsdWorkspace::with_simd(base.nrows(), true);

    scalar.prepare_reference(base.view()).unwrap();
    simd.prepare_reference(base.view()).unwrap();

    let scalar_rmsd = scalar.rmsd_prepared(moved.view()).unwrap();
    let simd_rmsd = simd.rmsd_prepared(moved.view()).unwrap();
    assert_relative_eq!(scalar_rmsd, simd_rmsd, epsilon = 1e-5);
}

#[test]
fn qcp_workspace_prepared_reference_must_be_loaded() {
    let coords = methane_coords();
    let mut workspace = QcpRmsdWorkspace::new(coords.nrows());
    let err = workspace.rmsd_prepared(coords.view()).unwrap_err();
    assert_eq!(err, ConformerError::WorkspaceReferenceNotPrepared);
}

#[test]
fn qcp_workspace_validates_atom_count() {
    let mut workspace = QcpRmsdWorkspace::new(4);
    let coords = methane_coords();
    let err = workspace.rmsd(coords.view(), coords.view()).unwrap_err();
    assert!(matches!(
        err,
        ConformerError::WorkspaceAtomCountMismatch {
            workspace_n_atoms: 4,
            coords_n_atoms: 5
        }
    ));
}

#[test]
fn qcp_workspace_fortran_path_matches_c_order_path() {
    let base = methane_coords();
    let moved = rotate_translate_z(&base);

    fn to_fortran(a: &Array2<f64>) -> Array2<f64> {
        let n = a.nrows();
        let mut buf = Vec::with_capacity(3 * n);
        for col in 0..3 {
            for row in 0..n {
                buf.push(a[[row, col]]);
            }
        }
        Array2::from_shape_vec((n, 3).f(), buf).unwrap()
    }

    let base_fortran = to_fortran(&base);
    let moved_fortran = to_fortran(&moved);

    for use_simd in [false, true] {
        let mut c_order = QcpRmsdWorkspace::with_simd(base.nrows(), use_simd);
        let mut fortran = QcpRmsdWorkspace::with_simd(base.nrows(), use_simd);

        c_order.prepare_reference(base.view()).unwrap();
        fortran.prepare_reference(base_fortran.view()).unwrap();

        let c_order_rmsd = c_order.rmsd_prepared(moved.view()).unwrap();
        let fortran_rmsd = fortran.rmsd_prepared(moved_fortran.view()).unwrap();

        // SIMD (F-order) and scalar (C-order) reductions sum the
        // partial products in different orders, so RMSD agreement is
        // bounded by floating-point associativity, not by the algebraic
        // identity.
        assert_relative_eq!(c_order_rmsd, fortran_rmsd, epsilon = 1e-7);
    }
}

#[test]
fn geometry_qcp_methods_use_same_kernel() {
    let base = methane();
    let moved = Geometry::new(
        METHANE_ATOM_TYPES.to_vec(),
        rotate_translate_z(&methane_coords()),
    )
    .unwrap();

    let mut workspace = QcpRmsdWorkspace::new(base.n_atoms());
    let a = base.qcp_rmsd(&moved).unwrap();
    let b = base
        .qcp_rmsd_with_workspace(&moved, &mut workspace)
        .unwrap();

    assert_relative_eq!(a, b, epsilon = 1e-5);
    assert_relative_eq!(a, 0.0, epsilon = 1e-5);
}

#[test]
fn qcp_supports_non_contiguous_views() {
    let base = methane_coords();
    let moved = rotate_translate_z(&base);

    let left = base.slice(s![..;-1, ..]);
    let right = moved.slice(s![..;-1, ..]);

    let rmsd = qcp_rmsd(left, right).unwrap();
    assert_relative_eq!(rmsd, 0.0, epsilon = 1e-5);
}

#[test]
fn qcp_reports_atom_count_mismatch() {
    let left = array![[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let right = array![
        [0.0_f64, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];

    let err = qcp_rmsd(left.view(), right.view()).unwrap_err();
    assert!(matches!(
        err,
        ConformerError::AtomCountMismatch {
            expected: 2,
            found: 3
        }
    ));
}

#[test]
fn geometry_qcp_rejects_atom_type_mismatch() {
    let methane = methane();
    let water = Geometry::new(
        vec![8, 1, 1],
        array![
            [0.0_f64, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
    )
    .unwrap();

    let err = methane.qcp_rmsd(&water).unwrap_err();
    assert_eq!(err, ConformerError::InconsistentAtomTypes);
}

#[test]
fn geometry_from_xyz_str_accepts_symbols() {
    let xyz = "2\ncomment\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n";
    let geo = Geometry::from_xyz_str(xyz).unwrap();

    assert_eq!(geo.atom_types(), &[6, 1]);
    assert_relative_eq!(geo.coords()[[1, 0]], 1.0_f64, epsilon = 1e-10);
}

#[test]
fn geometry_from_xyz_str_accepts_lower_upper_and_numeric_labels() {
    let xyz = "3\ncomment\ncl 0.0 0.0 0.0\nBR 1.0 0.0 0.0\n8 0.0 1.0 0.0\n";
    let geo = Geometry::from_xyz_str(xyz).unwrap();

    assert_eq!(geo.atom_types(), &[17, 35, 8]);
}

#[test]
fn geometry_from_xyz_str_rejects_invalid_numeric_labels() {
    let xyz = "1\ncomment\n0 0.0 0.0 0.0\n";
    let err = Geometry::from_xyz_str(xyz).unwrap_err();
    assert!(matches!(err, ConformerError::XyzParse(_)));
}

#[test]
fn geometry_multi_xyz_file_roundtrip() {
    let g1 = methane();
    let g2 = g1.mirror_geometry();
    let path = temp_xyz_path("geom_many");

    Geometry::to_xyz_file_many(&[g1.clone(), g2.clone()], &path).unwrap();
    let loaded = Geometry::from_xyz_file_many(&path).unwrap();
    let _ = fs::remove_file(&path);

    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].atom_types(), g1.atom_types());
    assert_eq!(loaded[1].coords().to_owned(), g2.coords().to_owned());
}

#[test]
fn geometry_from_xyz_file_requires_single_block() {
    let g1 = methane();
    let g2 = g1.mirror_geometry();
    let path = temp_xyz_path("geom_single");

    Geometry::to_xyz_file_many(&[g1, g2], &path).unwrap();
    let err = Geometry::from_xyz_file(&path).unwrap_err();
    let _ = fs::remove_file(&path);

    assert!(matches!(
        err,
        ConformerError::ExpectedSingleGeometry { found: 2 }
    ));
}

#[test]
fn ensemble_xyz_roundtrip_preserves_energies() {
    let ensemble = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![methane_coords(), rotate_translate_z(&methane_coords())],
        vec![0.125, -0.250],
    )
    .unwrap();

    let xyz = ensemble.to_xyz_str();
    let loaded = ConformerEnsemble::from_xyz_str(&xyz).unwrap();

    assert_eq!(loaded.atom_types(), ensemble.atom_types());
    assert_eq!(loaded.n_conformers(), 2);
    assert_relative_eq!(loaded.energies()[0], 0.125, epsilon = 1e-12);
    assert_relative_eq!(loaded.energies()[1], -0.250, epsilon = 1e-12);
}

#[test]
fn ensemble_from_xyz_defaults_missing_energy_to_zero() {
    let xyz = "2\nframe_a\n6 0.0 0.0 0.0\n1 1.0 0.0 0.0\n2\nframe_b\n6 0.0 0.0 0.0\n1 1.2 0.0 0.0\n";
    let ensemble = ConformerEnsemble::from_xyz_str(xyz).unwrap();

    assert_eq!(ensemble.n_conformers(), 2);
    assert_eq!(ensemble.energies(), &[0.0, 0.0]);
}

#[test]
fn ensemble_xyz_file_roundtrip() {
    let ensemble = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![methane_coords(), rotate_translate_z(&methane_coords())],
        vec![1.0, 2.0],
    )
    .unwrap();
    let path = temp_xyz_path("ensemble_file");

    ensemble.to_xyz_file(&path).unwrap();
    let loaded = ConformerEnsemble::from_xyz_file(&path).unwrap();
    let _ = fs::remove_file(&path);

    assert_eq!(loaded.atom_types(), ensemble.atom_types());
    assert_eq!(loaded.n_conformers(), ensemble.n_conformers());
    assert_eq!(loaded.energies().len(), ensemble.energies().len());
    assert_relative_eq!(loaded.energies()[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(loaded.energies()[1], 2.0, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// dRMSD / pairwise distances
// ---------------------------------------------------------------------------

fn translate(coords: &Array2<f64>, t: [f64; 3]) -> Array2<f64> {
    let mut out = coords.clone();
    for mut row in out.rows_mut() {
        row[0] += t[0];
        row[1] += t[1];
        row[2] += t[2];
    }
    out
}

fn to_fortran_layout(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut buf = Vec::with_capacity(3 * n);
    for col in 0..3 {
        for row in 0..n {
            buf.push(a[[row, col]]);
        }
    }
    Array2::from_shape_vec((n, 3).f(), buf).unwrap()
}

fn naive_pairwise_distances(coords: &Array2<f64>) -> Vec<f64> {
    let n = coords.nrows();
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[[j, 0]] - coords[[i, 0]];
            let dy = coords[[j, 1]] - coords[[i, 1]];
            let dz = coords[[j, 2]] - coords[[i, 2]];
            out.push((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
    out
}

#[test]
fn pairwise_distances_matches_naive_on_methane() {
    let coords = methane_coords();
    let got = pairwise_distances(coords.view()).unwrap();
    let want = naive_pairwise_distances(&coords);
    assert_eq!(got.len(), want.len());
    assert_eq!(got.len(), 5 * 4 / 2);
    for (a, b) in got.iter().zip(want.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-12);
    }
}

#[test]
fn pairwise_distances_simd_matches_scalar_large() {
    // Pseudo-random but deterministic geometry, larger than SIMD_MIN_ROW_LEN
    // so multiple SIMD chunks plus a scalar tail are exercised per row.
    let n = 73;
    let mut buf = Vec::with_capacity(3 * n);
    for i in 0..n {
        let x = (i as f64 * 0.137).sin();
        let y = (i as f64 * 0.241).cos();
        let z = (i as f64 * 0.359).sin() * 1.7;
        buf.extend_from_slice(&[x, y, z]);
    }
    let coords = Array2::from_shape_vec((n, 3), buf).unwrap();

    let got = pairwise_distances(coords.view()).unwrap();
    let want = naive_pairwise_distances(&coords);
    assert_eq!(got.len(), want.len());
    for (a, b) in got.iter().zip(want.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-12);
    }
}

#[test]
fn drmsd_self_is_zero() {
    let coords = methane_coords();
    let value = drmsd(coords.view(), coords.view()).unwrap();
    assert_relative_eq!(value, 0.0, epsilon = 1e-12);
}

#[test]
fn drmsd_is_rigid_transform_invariant() {
    let base = methane_coords();
    let rotated = rotate_translate_z(&base);
    let moved = translate(&rotated, [3.5, -1.25, 7.0]);

    let value = drmsd(base.view(), moved.view()).unwrap();
    assert_relative_eq!(value, 0.0, epsilon = 1e-10);
}

#[test]
fn drmsd_detects_perturbation() {
    let base = methane_coords();
    let mut perturbed = base.clone();
    perturbed[[1, 0]] += 0.10;
    perturbed[[2, 1]] -= 0.07;

    let value = drmsd(base.view(), perturbed.view()).unwrap();
    assert!(value > 1e-3, "expected dRMSD > 1e-3, got {value}");
}

#[test]
fn drmsd_simd_matches_scalar() {
    let n = 64;
    let mut buf_a = Vec::with_capacity(3 * n);
    let mut buf_b = Vec::with_capacity(3 * n);
    for i in 0..n {
        let t = i as f64;
        buf_a.extend_from_slice(&[
            (t * 0.11).sin(),
            (t * 0.19).cos() * 1.3,
            (t * 0.27).sin() * 0.7,
        ]);
        buf_b.extend_from_slice(&[
            (t * 0.11).sin() + 0.01 * (t * 0.5).cos(),
            (t * 0.19).cos() * 1.3 - 0.02 * (t * 0.3).sin(),
            (t * 0.27).sin() * 0.7 + 0.015 * (t * 0.7).cos(),
        ]);
    }
    let left = Array2::from_shape_vec((n, 3), buf_a).unwrap();
    let right = Array2::from_shape_vec((n, 3), buf_b).unwrap();

    let mut simd_ws = DrmsdWorkspace::with_simd(n, true);
    let mut scalar_ws = DrmsdWorkspace::with_simd(n, false);

    let v_simd = simd_ws.drmsd(left.view(), right.view()).unwrap();
    let v_scalar = scalar_ws.drmsd(left.view(), right.view()).unwrap();

    assert_relative_eq!(v_simd, v_scalar, epsilon = 1e-12);
    assert!(v_simd > 0.0);
}

#[test]
fn drmsd_workspace_fortran_path_matches_c_order_path() {
    let n = 48;
    let mut buf_a = Vec::with_capacity(3 * n);
    let mut buf_b = Vec::with_capacity(3 * n);
    for i in 0..n {
        let t = i as f64;
        buf_a.extend_from_slice(&[
            (t * 0.13).sin() * 2.0,
            (t * 0.23).cos(),
            (t * 0.31).sin(),
        ]);
        buf_b.extend_from_slice(&[
            (t * 0.13).sin() * 2.0 + 0.03,
            (t * 0.23).cos() - 0.02,
            (t * 0.31).sin() + 0.05,
        ]);
    }
    let left = Array2::from_shape_vec((n, 3), buf_a).unwrap();
    let right = Array2::from_shape_vec((n, 3), buf_b).unwrap();
    let left_f = to_fortran_layout(&left);
    let right_f = to_fortran_layout(&right);

    for use_simd in [false, true] {
        let mut c_order = DrmsdWorkspace::with_simd(n, use_simd);
        let mut fortran = DrmsdWorkspace::with_simd(n, use_simd);

        c_order.prepare_reference(left.view()).unwrap();
        fortran.prepare_reference(left_f.view()).unwrap();

        // Reference distances must match across layouts.
        for (a, b) in c_order
            .reference_distances()
            .iter()
            .zip(fortran.reference_distances().iter())
        {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }

        let v_c = c_order.drmsd_prepared(right.view()).unwrap();
        let v_f = fortran.drmsd_prepared(right_f.view()).unwrap();
        assert_relative_eq!(v_c, v_f, epsilon = 1e-12);
    }
}

#[test]
fn drmsd_prepared_matches_one_shot() {
    let base = methane_coords();
    let other = translate(&rotate_translate_z(&base), [0.5, 0.5, 0.0]);
    let mut ws = DrmsdWorkspace::new(base.nrows());

    let one_shot = drmsd(base.view(), other.view()).unwrap();
    ws.prepare_reference(base.view()).unwrap();
    let prepared = ws.drmsd_prepared(other.view()).unwrap();

    assert_relative_eq!(one_shot, prepared, epsilon = 1e-12);
}

#[test]
fn drmsd_workspace_reports_atom_count_mismatch() {
    let mut ws = DrmsdWorkspace::new(4);
    let coords = methane_coords(); // 5 atoms
    let err = ws.prepare_reference(coords.view()).unwrap_err();
    assert!(matches!(
        err,
        ConformerError::WorkspaceAtomCountMismatch {
            workspace_n_atoms: 4,
            coords_n_atoms: 5,
        }
    ));
}

#[test]
fn drmsd_workspace_requires_prepared_reference() {
    let mut ws = DrmsdWorkspace::new(5);
    let coords = methane_coords();
    let err = ws.drmsd_prepared(coords.view()).unwrap_err();
    assert!(matches!(
        err,
        ConformerError::WorkspaceReferenceNotPrepared
    ));
}

#[test]
fn drmsd_reports_atom_count_mismatch() {
    let left = array![[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let right = array![
        [0.0_f64, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let err = drmsd(left.view(), right.view()).unwrap_err();
    assert!(matches!(
        err,
        ConformerError::AtomCountMismatch {
            expected: 2,
            found: 3,
        }
    ));
}

// ===========================================================================
// Filter / inertia / element-weight tests
// ===========================================================================

fn perturb(coords: &Array2<f64>, amplitude: f64) -> Array2<f64> {
    let mut out = coords.clone();
    let mut seed = 0xC0FFEE_u64;
    for v in out.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = ((seed >> 33) as f64) / (u32::MAX as f64) - 0.5;
        *v += amplitude * r;
    }
    out
}

#[test]
fn atomic_weight_known_values() {
    assert_relative_eq!(atomic_weight(1).unwrap(), 1.008, epsilon = 1e-9);
    assert_relative_eq!(atomic_weight(6).unwrap(), 12.011, epsilon = 1e-9);
    assert_relative_eq!(atomic_weight(8).unwrap(), 15.999, epsilon = 1e-9);
    assert!(matches!(atomic_weight(0), Err(ConformerError::UnknownAtomType(0))));
    assert!(matches!(atomic_weight(200), Err(ConformerError::UnknownAtomType(200))));
}

#[test]
fn rotational_constants_methane_invariant_under_rotation() {
    let m = methane();
    let masses: Vec<f64> = m
        .atom_types()
        .iter()
        .map(|&z| atomic_weight(z).unwrap())
        .collect();
    let rotated = rotate_translate_z(&methane_coords());

    let c1 = rotational_constants_hz(m.coords(), &masses).unwrap();
    let c2 = rotational_constants_hz(rotated.view(), &masses).unwrap();
    match (c1, c2) {
        (RotationalConstants::Nonlinear(a), RotationalConstants::Nonlinear(b)) => {
            for (x, y) in a.iter().zip(b.iter()) {
                assert_relative_eq!(*x, *y, max_relative = 1e-6);
            }
        }
        _ => panic!("methane should be classified as nonlinear"),
    }
}

#[test]
fn rotational_constants_linear_h2() {
    let coords = Array2::from_shape_vec(
        (2, 3),
        vec![0.0, 0.0, 0.0, 0.74, 0.0, 0.0],
    )
    .unwrap();
    let masses = vec![1.008, 1.008];
    let r = rotational_constants_hz(coords.view(), &masses).unwrap();
    assert!(matches!(r, RotationalConstants::Linear(_)));
}

#[test]
fn rotational_constants_single_atom_errors() {
    let coords = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
    let masses = vec![1.008];
    let err = rotational_constants_hz(coords.view(), &masses).unwrap_err();
    assert!(matches!(err, ConformerError::SingleAtomRotationalConstants));
}

#[test]
fn rmsd_permuted_identity_matches_qcp() {
    let a = methane_coords();
    let b = rotate_translate_z(&methane_coords());
    let r_filter = rmsd_permuted(a.view(), b.view(), None, None, true).unwrap();
    let r_qcp = qcp_rmsd(a.view(), b.view()).unwrap();
    assert_relative_eq!(r_filter, r_qcp, max_relative = 1e-6);
    assert!(r_filter < 1e-6);
}

#[test]
fn rmsd_permuted_swap_h_atoms_recovers_zero() {
    // Swap two equivalent hydrogens in methane. With a Kabsch alignment
    // and the matching permutation, RMSD must drop back to ~0.
    let a = methane_coords();
    let mut b = a.clone();
    let row1 = b.row(1).to_owned();
    let row2 = b.row(2).to_owned();
    b.row_mut(1).assign(&row2);
    b.row_mut(2).assign(&row1);

    // No mapping -> nonzero RMSD.
    let r_id = rmsd_permuted(a.view(), b.view(), None, None, true).unwrap();
    assert!(r_id > 0.1);

    // Mapping that undoes the swap: 0->0, 1->2, 2->1, 3->3, 4->4.
    let mapping = vec![0, 2, 1, 3, 4];
    let r_perm =
        rmsd_permuted(a.view(), b.view(), None, Some(&mapping), true).unwrap();
    assert!(r_perm < 1e-6);
}

#[test]
fn rmsd_permuted_no_rigid_no_centering() {
    // Translate b by (1, 0, 0). Without rigid_transform, the RMSD must
    // equal exactly 1 ? (no centering is applied).
    let a = methane_coords();
    let mut b = a.clone();
    for i in 0..b.nrows() {
        b[[i, 0]] += 1.0;
    }
    let r = rmsd_permuted(a.view(), b.view(), None, None, false).unwrap();
    assert_relative_eq!(r, 1.0, max_relative = 1e-12);
}

#[test]
fn rmsd_permuted_mass_weighted_runs() {
    let a = methane_coords();
    let b = perturb(&a, 0.01);
    let masses: Vec<f64> = METHANE_ATOM_TYPES
        .iter()
        .map(|&z| atomic_weight(z).unwrap())
        .collect();
    let r_unw = rmsd_permuted(a.view(), b.view(), None, None, true).unwrap();
    let r_mw =
        rmsd_permuted(a.view(), b.view(), Some(&masses), None, true).unwrap();
    // Both should be small; mass-weighting changes the value but it stays
    // bounded by the largest per-atom displacement.
    assert!(r_unw < 0.05);
    assert!(r_mw < 0.05);
    assert!(r_mw > 0.0);
}

#[test]
fn min_rmsd_over_mappings_picks_best() {
    let a = methane_coords();
    let mut b = a.clone();
    let row1 = b.row(1).to_owned();
    let row2 = b.row(2).to_owned();
    b.row_mut(1).assign(&row2);
    b.row_mut(2).assign(&row1);

    let identity = vec![0, 1, 2, 3, 4];
    let swap = vec![0, 2, 1, 3, 4];
    let mappings = ValidatedMappings::new(
        vec![identity, swap],
        &METHANE_ATOM_TYPES,
    )
    .unwrap();

    let best =
        min_rmsd_over_mappings(a.view(), b.view(), None, &mappings, true)
            .unwrap();
    assert!(best < 1e-6);
}

#[test]
fn validate_permutation_catches_bad_inputs() {
    let types = vec![6_u8, 1, 1, 1, 1];
    // Wrong length.
    let bad_len = vec![0_usize, 1, 2];
    assert!(matches!(
        validate_permutation(&bad_len, &types, &types),
        Err(ConformerError::InvalidPermutationLength { .. })
    ));
    // Out-of-range index.
    let bad_idx = vec![0_usize, 1, 2, 3, 99];
    assert!(matches!(
        validate_permutation(&bad_idx, &types, &types),
        Err(ConformerError::InvalidPermutationIndex { index: 99, .. })
    ));
    // Atom type mismatch (swap C with an H slot).
    let bad_type = vec![1_usize, 0, 2, 3, 4];
    assert!(matches!(
        validate_permutation(&bad_type, &types, &types),
        Err(ConformerError::PermutationAtomTypeMismatch { .. })
    ));
}

#[test]
fn is_different_conformer_energy_branch() {
    let m = methane();
    let opts = ConfDiffOptions {
        energy_threshold: 1e-4,
        rot_threshold: f64::INFINITY,
        rmsd_threshold: f64::INFINITY,
        ..ConfDiffOptions::default()
    };
    let diff = is_different_conformer(
        m.coords(),
        m.coords(),
        m.atom_types(),
        Some(0.0),
        Some(1.0),
        &ValidatedMappings::empty(),
        &opts,
    )
    .unwrap();
    assert!(diff);

    let same = is_different_conformer(
        m.coords(),
        m.coords(),
        m.atom_types(),
        Some(0.0),
        Some(1e-5),
        &ValidatedMappings::empty(),
        &opts,
    )
    .unwrap();
    assert!(!same);
}

#[test]
fn is_different_conformer_requires_energies() {
    let m = methane();
    let opts = ConfDiffOptions::default();
    let err = is_different_conformer(
        m.coords(),
        m.coords(),
        m.atom_types(),
        None,
        None,
        &ValidatedMappings::empty(),
        &opts,
    )
    .unwrap_err();
    assert!(matches!(err, ConformerError::MissingEnergyForThreshold));
}

#[test]
fn is_different_conformer_rmsd_only() {
    let a = methane();
    let b_coords = perturb(&methane_coords(), 0.001);
    let opts = ConfDiffOptions {
        energy_threshold: f64::INFINITY,
        rot_threshold: f64::INFINITY,
        rmsd_threshold: 0.01,
        ..ConfDiffOptions::default()
    };
    let diff = is_different_conformer(
        a.coords(),
        b_coords.view(),
        a.atom_types(),
        None,
        None,
        &ValidatedMappings::empty(),
        &opts,
    )
    .unwrap();
    assert!(!diff);

    let b_far = perturb(&methane_coords(), 0.5);
    let diff_far = is_different_conformer(
        a.coords(),
        b_far.view(),
        a.atom_types(),
        None,
        None,
        &ValidatedMappings::empty(),
        &opts,
    )
    .unwrap();
    assert!(diff_far);
}

#[test]
fn filter_ensemble_dedupes_duplicates() {
    // Three conformers: c0 (low E), c1 = c0 + tiny noise + slightly higher E
    // (duplicate), c2 = c0 + large perturbation + higher E (distinct).
    let c0 = methane_coords();
    let c1 = perturb(&c0, 1e-4);
    let c2 = perturb(&c0, 0.5);
    let ens = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![c0.clone(), c1, c2],
        vec![0.0, 1e-6, 1e-3],
    )
    .unwrap();

    let opts = ConfFilterOptions::default();
    let filtered = filter_ensemble(&ens, &opts, &ValidatedMappings::empty()).unwrap();
    assert_eq!(filtered.n_conformers(), 2);
    // Lowest-energy survivor first.
    assert!(filtered.energies()[0] <= filtered.energies()[1]);
}

#[test]
fn filter_ensemble_energy_window() {
    let c0 = methane_coords();
    let c1 = perturb(&c0, 0.5);
    let c2 = perturb(&c0, 0.5);
    let ens = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![c0, c1, c2],
        vec![0.0, 1e-4, 1.0],
    )
    .unwrap();

    let opts = ConfFilterOptions {
        energy_window: 1e-2,
        ..ConfFilterOptions::default()
    };
    let filtered = filter_ensemble(&ens, &opts, &ValidatedMappings::empty()).unwrap();
    assert_eq!(filtered.n_conformers(), 2);
}

#[test]
fn filter_ensemble_keeps_lowest_only_when_all_similar() {
    let c0 = methane_coords();
    let c1 = perturb(&c0, 1e-5);
    let c2 = perturb(&c0, 1e-5);
    let ens = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![c0, c1, c2],
        vec![0.0, 1e-7, 2e-7],
    )
    .unwrap();
    let opts = ConfFilterOptions::default();
    let filtered = filter_ensemble(&ens, &opts, &ValidatedMappings::empty()).unwrap();
    assert_eq!(filtered.n_conformers(), 1);
    assert_eq!(filtered.energies()[0], 0.0);
}

#[test]
fn filter_ensemble_boltzmann_truncates() {
    // Three well-separated conformers; with a tiny temperature only the
    // lowest survives the cumulative-Boltzmann cutoff at 0.99.
    let c0 = methane_coords();
    let c1 = perturb(&c0, 0.5);
    let c2 = perturb(&c0, 0.5);
    let ens = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![c0, c1, c2],
        vec![0.0, 1.0, 2.0],
    )
    .unwrap();

    let opts = ConfFilterOptions {
        temperature_k: 1.0,
        cum_boltzmann_threshold: 0.99,
        ..ConfFilterOptions::default()
    };
    let filtered = filter_ensemble(&ens, &opts, &ValidatedMappings::empty()).unwrap();
    assert_eq!(filtered.n_conformers(), 1);
}

#[test]
fn filter_ensemble_uses_mappings() {
    // Build two coords where c1 is c0 with two equivalent hydrogens
    // swapped. Without a permutation they look different; with the swap
    // mapping they are identical and one is filtered out.
    let c0 = methane_coords();
    let mut c1 = c0.clone();
    let r1 = c1.row(1).to_owned();
    let r2 = c1.row(2).to_owned();
    c1.row_mut(1).assign(&r2);
    c1.row_mut(2).assign(&r1);

    let ens = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![c0, c1],
        vec![0.0, 1e-6],
    )
    .unwrap();
    let opts = ConfFilterOptions::default();

    // Without mapping: both survive (Kabsch can't undo the relabelling).
    let no_map = filter_ensemble(&ens, &opts, &ValidatedMappings::empty()).unwrap();
    assert_eq!(no_map.n_conformers(), 2);

    // With identity + swap mapping: c1 collapses onto c0 and is removed.
    let swap = vec![0_usize, 2, 1, 3, 4];
    let identity = vec![0_usize, 1, 2, 3, 4];
    let mappings = ValidatedMappings::new(
        vec![identity, swap],
        &METHANE_ATOM_TYPES,
    )
    .unwrap();
    let with_map = filter_ensemble(&ens, &opts, &mappings).unwrap();
    assert_eq!(with_map.n_conformers(), 1);
}

#[test]
fn filter_ensemble_empty_passes_through() {
    let ens = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![],
        vec![],
    )
    .unwrap();
    let opts = ConfFilterOptions::default();
    let filtered = filter_ensemble(&ens, &opts, &ValidatedMappings::empty()).unwrap();
    assert_eq!(filtered.n_conformers(), 0);
}

#[test]
fn filter_ensemble_mirror_check_collapses_mirror_image() {
    // Methane is achiral; the mirror image of any conformer should
    // collapse onto it when `mirror_check` is enabled. Without it, the
    // mirrored geometry survives because RMSD between a frame and its
    // mirror after Kabsch alignment is non-zero for an asymmetric
    // perturbation.
    let c0 = methane_coords();
    let mut c1 = perturb(&c0, 0.2);
    // Mirror c1 across the x-axis.
    for mut row in c1.rows_mut() {
        row[0] = -row[0];
    }

    let ens = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![c0.clone(), c1.clone()],
        vec![0.0, 1e-6],
    )
    .unwrap();

    // Default (mirror_check=false): both conformers survive.
    let opts_off = ConfFilterOptions::default();
    let n_off = filter_ensemble(&ens, &opts_off, &ValidatedMappings::empty())
        .unwrap()
        .n_conformers();
    assert_eq!(n_off, 2);

    // With mirror_check=true and is_chiral=false: the mirrored copy is
    // recognised as a duplicate of the kept conformer.
    let opts_on = ConfFilterOptions {
        diff: ConfDiffOptions {
            mirror_check: true,
            ..ConfDiffOptions::default()
        },
        is_chiral: false,
        ..ConfFilterOptions::default()
    };
    let n_on = filter_ensemble(&ens, &opts_on, &ValidatedMappings::empty())
        .unwrap()
        .n_conformers();
    assert_eq!(n_on, 1);
}

#[test]
fn ensemble_xyz_str_uses_element_symbols() {
    let geo = methane();
    let ens = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![geo.coords().to_owned()],
        vec![-40.5],
    )
    .unwrap();
    let xyz = ens.to_xyz_str();
    let mut lines = xyz.lines();
    assert_eq!(lines.next(), Some("5"));
    let comment = lines.next().unwrap();
    assert!(comment.starts_with("energy="));
    // First atom is carbon (atomic number 6); written as the symbol.
    assert!(lines.next().unwrap().starts_with("C "));
}

#[test]
fn ensemble_xyz_roundtrip_with_symbol_writer() {
    // Round-trip an ensemble through the writer + parser to ensure that
    // switching the writer to element symbols hasn't broken the parser.
    let c0 = methane_coords();
    let c1 = perturb(&c0, 0.4);
    let ens = ConformerEnsemble::new(
        METHANE_ATOM_TYPES.to_vec(),
        vec![c0, c1],
        vec![0.0, 1e-3],
    )
    .unwrap();

    let written = ens.to_xyz_str();
    let parsed = ConformerEnsemble::from_xyz_str(&written).unwrap();
    assert_eq!(parsed.n_conformers(), 2);
    assert_eq!(parsed.atom_types(), ens.atom_types());
    assert_relative_eq!(parsed.energies()[0], 0.0);
    assert_relative_eq!(parsed.energies()[1], 1e-3);
}

#[test]
fn ensemble_from_crest_style_xyz_str() {
    // CREST emits multi-block XYZ where the comment line is a bare
    // float (the energy in Hartree). The parser must accept it.
    let crest = "\
3
       -76.41234567
O 0.0000 0.0000 0.0000
H 0.0000 0.0000 0.9580
H 0.9266 0.0000 -0.2390
3
       -76.41200000
O 0.0500 0.0000 0.0000
H 0.0000 0.0000 0.9580
H 0.9266 0.0000 -0.2390
3
       -76.40000000
O 0.5000 0.0000 0.0000
H 0.0000 0.0000 0.9580
H 0.9266 0.0000 -0.2390
";
    let ens = ConformerEnsemble::from_xyz_str(crest).unwrap();
    assert_eq!(ens.n_conformers(), 3);
    assert_eq!(ens.atom_types(), &[8u8, 1, 1]);
    assert_relative_eq!(ens.energies()[0], -76.41234567);
    assert_relative_eq!(ens.energies()[1], -76.41200000);
    assert_relative_eq!(ens.energies()[2], -76.40000000);
}

