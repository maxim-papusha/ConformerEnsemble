//! PyO3 bindings exposing the public Rust API as the
//! `conformerensemblers` Python extension module.
//!
//! Built only when the `python` feature is enabled. The bindings deliberately
//! stay close to the chemtrayzer `core.coords` surface (constructor names,
//! `Geometry` / `ConformerEnsemble`, `.filter`, `ConfDiffOptions` /
//! `ConfFilterOptions` field names) so the Python wrapper can be a thin
//! re-export rather than a separate translation layer.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use ndarray::Array2;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySequence, PySlice, PyTuple, PyType};

use crate::xyz_io::atom_symbol;
use crate::{
    atomic_weights, filter_ensemble as rs_filter,
    is_different_conformer as rs_is_diff, qcp_rmsd as rs_qcp_rmsd,
    rotational_constants_hz, ConfDiffOptions as RsDiff,
    ConfFilterOptions as RsFilter, ConformerEnsemble as RsEnsemble,
    ConformerError, Geometry as RsGeometry, QcpRmsdWorkspaceF32,
    QcpRmsdWorkspaceT, RotationalConstants, ValidatedMappings,
};

// ---------------------------------------------------------------------------
// helpers

fn rs_err_to_py(err: ConformerError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Maps an element symbol (case-insensitive) to its atomic number.
/// Mirrors the parser in `xyz_io.rs` but operates on Python strings.
fn symbol_to_atomic_number(token: &str) -> Option<u8> {
    if token.is_empty() || !token.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let mut canonical = String::with_capacity(token.len());
    let mut chars = token.chars();
    canonical.push(chars.next()?.to_ascii_uppercase());
    for c in chars {
        canonical.push(c.to_ascii_lowercase());
    }
    for n in 1u8..=118 {
        if atom_symbol(n) == canonical.as_str() {
            return Some(n);
        }
    }
    None
}

/// Converts a Python sequence (`list`/`tuple`/`np.ndarray`) of element
/// symbols *or* atomic numbers into the internal `Vec<u8>`.
fn pyseq_to_atom_types(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let seq = obj.downcast::<PySequence>().map_err(|_| {
        PyTypeError::new_err(
            "atom_types must be a sequence of element symbols or atomic numbers",
        )
    })?;
    let n = seq.len()? as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let item = seq.get_item(i)?;
        if let Ok(s) = item.extract::<String>() {
            let z = symbol_to_atomic_number(&s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown element symbol '{s}' at index {i}"
                ))
            })?;
            out.push(z);
        } else if let Ok(z) = item.extract::<i64>() {
            if !(1..=118).contains(&z) {
                return Err(PyValueError::new_err(format!(
                    "atomic number {z} at index {i} out of range 1..=118"
                )));
            }
            out.push(z as u8);
        } else {
            return Err(PyTypeError::new_err(format!(
                "atom_types[{i}] must be str or int, got {}",
                item.get_type().name()?
            )));
        }
    }
    Ok(out)
}

/// Converts a Python coordinate input (numpy array of shape `(n, 3)` or a
/// list of `[x, y, z]` triples) into an owned `Array2<f64>`.
fn pyany_to_coords(obj: &Bound<'_, PyAny>) -> PyResult<Array2<f64>> {
    // Fast path: numpy array.
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        return Ok(arr.as_array().to_owned());
    }
    // Fallback: cast through numpy.asarray for arbitrary nested sequences.
    let py = obj.py();
    let np = py.import_bound("numpy")?;
    let arr = np
        .getattr("asarray")?
        .call1((obj, "float64"))?
        .extract::<PyReadonlyArray2<f64>>()?;
    Ok(arr.as_array().to_owned())
}

fn coords_to_pyarray<'py>(
    py: Python<'py>,
    coords: ndarray::ArrayView2<'_, f64>,
) -> Bound<'py, PyArray2<f64>> {
    coords.to_owned().into_pyarray_bound(py)
}

// ---------------------------------------------------------------------------
// Options pyclasses
//
// These mirror chemtrayzer.core.coords.ConfDiffOptions /
// ConfFilterOptions / GeoConfDiffOptions: flat fields, not the nested Rust
// `ConfFilterOptions { diff: ConfDiffOptions, ... }` shape.

/// Subset of ConfDiffOptions excluding energy-dependent fields. Mirrors
/// chemtrayzer's `GeoConfDiffOptions`.
#[pyclass(name = "GeoConfDiffOptions", module = "conformerensemblers", subclass)]
#[derive(Clone, Debug)]
pub struct PyGeoDiff {
    #[pyo3(get, set)] pub rmsd_threshold: f64,
    #[pyo3(get, set)] pub rot_threshold: f64,
    #[pyo3(get, set)] pub mass_weighted: bool,
    #[pyo3(get, set)] pub rigid_transform: bool,
}

#[pymethods]
impl PyGeoDiff {
    #[new]
    #[pyo3(signature = (
        rmsd_threshold = 0.125,
        rot_threshold = 1.0,
        mass_weighted = false,
        rigid_transform = true,
    ))]
    fn new(
        rmsd_threshold: f64,
        rot_threshold: f64,
        mass_weighted: bool,
        rigid_transform: bool,
    ) -> Self {
        Self { rmsd_threshold, rot_threshold, mass_weighted, rigid_transform }
    }

    fn __repr__(&self) -> String {
        format!(
            "GeoConfDiffOptions(rmsd_threshold={}, rot_threshold={}, mass_weighted={}, rigid_transform={})",
            self.rmsd_threshold, self.rot_threshold, self.mass_weighted, self.rigid_transform,
        )
    }
}

#[pyclass(
    name = "ConfDiffOptions",
    module = "conformerensemblers",
    extends = PyGeoDiff,
    subclass,
)]
#[derive(Clone, Debug)]
pub struct PyDiff {
    #[pyo3(get, set)] pub energy_threshold: f64,
    #[pyo3(get, set)] pub mirror_check: bool,
}

#[pymethods]
impl PyDiff {
    #[new]
    #[pyo3(signature = (
        rmsd_threshold = 0.125,
        rot_threshold = 1.0,
        mass_weighted = false,
        rigid_transform = true,
        energy_threshold = 0.00038,
        mirror_check = false,
    ))]
    fn new(
        rmsd_threshold: f64,
        rot_threshold: f64,
        mass_weighted: bool,
        rigid_transform: bool,
        energy_threshold: f64,
        mirror_check: bool,
    ) -> (Self, PyGeoDiff) {
        (
            Self { energy_threshold, mirror_check },
            PyGeoDiff {
                rmsd_threshold,
                rot_threshold,
                mass_weighted,
                rigid_transform,
            },
        )
    }

    fn __repr__(slf: PyRef<'_, Self>) -> String {
        let base = slf.as_super();
        format!(
            "ConfDiffOptions(rmsd_threshold={}, rot_threshold={}, mass_weighted={}, rigid_transform={}, energy_threshold={}, mirror_check={})",
            base.rmsd_threshold, base.rot_threshold, base.mass_weighted, base.rigid_transform,
            slf.energy_threshold, slf.mirror_check,
        )
    }
}

#[pyclass(
    name = "ConfFilterOptions",
    module = "conformerensemblers",
    extends = PyDiff,
)]
#[derive(Clone, Debug)]
pub struct PyFilter {
    #[pyo3(get, set)] pub temperature: f64,
    #[pyo3(get, set)] pub cum_boltzmann_threshold: f64,
    #[pyo3(get, set)] pub energy_window: f64,
    /// `is_chiral` mirrors the Python `_is_chiral` ensemble flag; the
    /// Rust filter only honours `mirror_check` when this is false.
    #[pyo3(get, set)] pub is_chiral: bool,
}

#[pymethods]
impl PyFilter {
    #[new]
    #[pyo3(signature = (
        rmsd_threshold = 0.125,
        rot_threshold = 1.0,
        mass_weighted = false,
        rigid_transform = true,
        energy_threshold = 0.00038,
        mirror_check = false,
        temperature = 1500.0,
        cum_boltzmann_threshold = 1.0,
        energy_window = f64::INFINITY,
        is_chiral = false,
    ))]
    fn new(
        rmsd_threshold: f64,
        rot_threshold: f64,
        mass_weighted: bool,
        rigid_transform: bool,
        energy_threshold: f64,
        mirror_check: bool,
        temperature: f64,
        cum_boltzmann_threshold: f64,
        energy_window: f64,
        is_chiral: bool,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyGeoDiff {
            rmsd_threshold,
            rot_threshold,
            mass_weighted,
            rigid_transform,
        })
        .add_subclass(PyDiff { energy_threshold, mirror_check })
        .add_subclass(Self {
            temperature,
            cum_boltzmann_threshold,
            energy_window,
            is_chiral,
        })
    }
}

fn pyfilter_to_rs(opts: &Bound<'_, PyFilter>) -> RsFilter {
    let py_filter = opts.borrow();
    // Walk the 3-level inheritance chain (PyFilter -> PyDiff -> PyGeoDiff).
    let temperature = py_filter.temperature;
    let cum_boltzmann_threshold = py_filter.cum_boltzmann_threshold;
    let energy_window = py_filter.energy_window;
    let is_chiral = py_filter.is_chiral;
    let py_diff: PyRef<'_, PyDiff> = PyRef::into_super(py_filter);
    let energy_threshold = py_diff.energy_threshold;
    let mirror_check = py_diff.mirror_check;
    let py_geo: PyRef<'_, PyGeoDiff> = PyRef::into_super(py_diff);
    RsFilter {
        diff: RsDiff {
            rmsd_threshold: py_geo.rmsd_threshold,
            rot_threshold: py_geo.rot_threshold,
            energy_threshold,
            mass_weighted: py_geo.mass_weighted,
            rigid_transform: py_geo.rigid_transform,
            mirror_check,
        },
        temperature_k: temperature,
        cum_boltzmann_threshold,
        energy_window,
        is_chiral,
    }
}

// ---------------------------------------------------------------------------
// Geometry

#[pyclass(name = "Geometry", module = "conformerensemblers")]
#[derive(Clone, Debug)]
pub struct PyGeometry {
    pub(crate) inner: RsGeometry,
}

#[pymethods]
impl PyGeometry {
    #[new]
    #[pyo3(signature = (atom_types, coords))]
    fn new(
        atom_types: &Bound<'_, PyAny>,
        coords: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let atom_types = pyseq_to_atom_types(atom_types)?;
        let coords = pyany_to_coords(coords)?;
        Ok(Self {
            inner: RsGeometry::new(atom_types, coords).map_err(rs_err_to_py)?,
        })
    }

    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// Returns the atomic numbers as a Python list of ints (chemtrayzer
    /// compatibility: chemtrayzer's `atom_types` is a list of element
    /// labels; users can build symbols from the integers when needed).
    #[getter]
    fn atom_types<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new_bound(py, self.inner.atom_types().iter().map(|&n| n as i64))
            .extract()
    }

    /// Returns the atomic numbers as element symbols.
    fn atom_symbols<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new_bound(
            py,
            self.inner.atom_types().iter().map(|&n| atom_symbol(n)),
        )
        .extract()
    }

    /// `(n_atoms, 3)` `float64` numpy array (a copy, so mutation by the
    /// caller does not corrupt the stored geometry).
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        coords_to_pyarray(py, self.inner.coords())
    }

    #[pyo3(signature = (comment = None))]
    fn xyz_str(&self, comment: Option<&str>) -> String {
        self.inner.xyz_str(comment)
    }

    fn mirror_geometry(&self) -> Self {
        Self { inner: self.inner.mirror_geometry() }
    }

    /// Optimal rigid-body RMSD (Theobald / QCP) to another geometry.
    fn qcp_rmsd(&self, other: &PyGeometry) -> PyResult<f64> {
        self.inner.qcp_rmsd(&other.inner).map_err(rs_err_to_py)
    }

    /// Rotational constants in Hz. Returns a 3-tuple `(A, B, C)` for
    /// non-linear molecules and a 1-tuple `(B,)` for linear ones.
    fn rotational_constants<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let masses =
            atomic_weights(self.inner.atom_types()).map_err(rs_err_to_py)?;
        let rc = rotational_constants_hz(self.inner.coords(), &masses)
            .map_err(rs_err_to_py)?;
        match rc {
            RotationalConstants::Linear(b) => {
                Ok(PyTuple::new_bound(py, [b]))
            }
            RotationalConstants::Nonlinear([a, b, c]) => {
                Ok(PyTuple::new_bound(py, [a, b, c]))
            }
        }
    }

    #[classmethod]
    fn from_xyz_str(_cls: &Bound<'_, PyType>, input: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RsGeometry::from_xyz_str(input).map_err(rs_err_to_py)?,
        })
    }

    #[classmethod]
    fn from_xyz_file(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RsGeometry::from_xyz_file(path).map_err(rs_err_to_py)?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.n_atoms()
    }

    fn __repr__(&self) -> String {
        format!("Geometry(n_atoms={})", self.inner.n_atoms())
    }
}

// ---------------------------------------------------------------------------
// ConformerEnsemble

#[pyclass(name = "ConformerEnsemble", module = "conformerensemblers")]
#[derive(Clone, Debug)]
pub struct PyEnsemble {
    pub(crate) inner: RsEnsemble,
}

#[pymethods]
impl PyEnsemble {
    /// Build an ensemble from a list of `Geometry` objects + matching
    /// energies. Mirrors chemtrayzer's `ConformerEnsemble(geos=, energies=)`.
    #[new]
    #[pyo3(signature = (geos = None, energies = None))]
    fn new(
        geos: Option<&Bound<'_, PySequence>>,
        energies: Option<&Bound<'_, PySequence>>,
    ) -> PyResult<Self> {
        let geos_vec: Vec<PyGeometry> = match geos {
            Some(seq) => {
                let n = seq.len()? as usize;
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    v.push(seq.get_item(i)?.extract::<PyGeometry>()?);
                }
                v
            }
            None => Vec::new(),
        };
        let energies_vec: Vec<f64> = match energies {
            Some(seq) => {
                let n = seq.len()? as usize;
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    v.push(seq.get_item(i)?.extract::<f64>()?);
                }
                v
            }
            None => vec![0.0; geos_vec.len()],
        };
        if geos_vec.is_empty() {
            // Mirror chemtrayzer: an empty ensemble has no atom types.
            return Ok(Self {
                inner: RsEnsemble::new(vec![], vec![], energies_vec)
                    .map_err(rs_err_to_py)?,
            });
        }
        let atom_types = geos_vec[0].inner.atom_types().to_vec();
        for (i, g) in geos_vec.iter().enumerate().skip(1) {
            if g.inner.atom_types() != atom_types.as_slice() {
                return Err(PyValueError::new_err(format!(
                    "geos[{i}] has inconsistent atom types"
                )));
            }
        }
        let coords: Vec<Array2<f64>> =
            geos_vec.into_iter().map(|g| g.inner.into_coords()).collect();
        Ok(Self {
            inner: RsEnsemble::new(atom_types, coords, energies_vec)
                .map_err(rs_err_to_py)?,
        })
    }

    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    #[getter]
    fn n_conformers(&self) -> usize {
        self.inner.n_conformers()
    }

    #[getter]
    fn atom_types<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new_bound(py, self.inner.atom_types().iter().map(|&n| n as i64))
            .extract()
    }

    /// Returns energies as a plain Python list of floats so equality
    /// comparisons (`ensemble.energies == [0.3, 0.2, 0.1]`) work naturally.
    #[getter]
    fn energies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new_bound(py, self.inner.energies().iter().copied()).extract()
    }

    fn add_geometry(&mut self, geo: &PyGeometry, energy: f64) -> PyResult<()> {
        self.inner
            .push_geometry(geo.inner.clone(), energy)
            .map_err(rs_err_to_py)
    }

    #[pyo3(signature = (descending = false))]
    fn sort(&mut self, descending: bool) {
        self.inner.sort_by_energy(descending);
    }

    /// Boltzmann weights for the conformers selected by `key`.
    ///
    /// `key` accepts either an integer (returns a 0-d-like 1-element array
    /// for that index) or a Python slice.
    #[pyo3(signature = (key, *, temperature))]
    fn get_boltzmann_weight<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'_, PyAny>,
        temperature: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let weights = self
            .inner
            .boltzmann_weights(temperature)
            .map_err(rs_err_to_py)?;
        let n = weights.len();
        let selection: Vec<f64> = if let Ok(slice) = key.downcast::<PySlice>() {
            let indices = slice.indices(n as isize)?;
            let start = indices.start as isize;
            let stop = indices.stop as isize;
            let step = indices.step as isize;
            let mut out = Vec::new();
            if step > 0 {
                let mut i = start;
                while i < stop {
                    out.push(weights[i as usize]);
                    i += step;
                }
            } else if step < 0 {
                let mut i = start;
                while i > stop {
                    out.push(weights[i as usize]);
                    i += step;
                }
            } else {
                return Err(PyValueError::new_err("slice step cannot be zero"));
            }
            out
        } else if let Ok(idx) = key.extract::<isize>() {
            let n_i = n as isize;
            let real = if idx < 0 { idx + n_i } else { idx };
            if real < 0 || real >= n_i {
                return Err(PyIndexError::new_err("index out of range"));
            }
            vec![weights[real as usize]]
        } else {
            return Err(PyTypeError::new_err(
                "key must be int or slice",
            ));
        };
        Ok(ndarray::Array1::from(selection).into_pyarray_bound(py))
    }

    /// Returns a single conformer as a `Geometry`.
    fn geometry(&self, index: usize) -> PyResult<PyGeometry> {
        self.inner
            .geometry(index)
            .map(|g| PyGeometry { inner: g })
            .ok_or_else(|| PyIndexError::new_err("conformer index out of range"))
    }

    fn to_xyz_str(&self) -> String {
        self.inner.to_xyz_str()
    }

    fn xyz_str(&self) -> String {
        self.inner.to_xyz_str()
    }

    fn to_xyz_file(&self, path: &str) -> PyResult<()> {
        self.inner.to_xyz_file(path).map_err(rs_err_to_py)
    }

    #[classmethod]
    fn from_xyz_str(_cls: &Bound<'_, PyType>, input: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RsEnsemble::from_xyz_str(input).map_err(rs_err_to_py)?,
        })
    }

    #[classmethod]
    fn from_xyz_file(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RsEnsemble::from_xyz_file(path).map_err(rs_err_to_py)?,
        })
    }

    /// chemtrayzer-compatible `.filter`. Returns a new ensemble unless
    /// `inplace=True`, in which case the ensemble is mutated and `None` is
    /// returned.
    #[pyo3(signature = (options = None, *, inplace = false))]
    fn filter<'py>(
        &mut self,
        py: Python<'py>,
        options: Option<&Bound<'_, PyFilter>>,
        inplace: bool,
    ) -> PyResult<Option<Self>> {
        // Default options if none supplied: use chemtrayzer defaults.
        let rs_opts = match options {
            Some(o) => pyfilter_to_rs(o),
            None => RsFilter::default(),
        };
        let filtered = rs_filter(
            &self.inner,
            &rs_opts,
            &ValidatedMappings::empty(),
        )
        .map_err(rs_err_to_py)?;
        if inplace {
            self.inner = filtered;
            Ok(None)
        } else {
            let _ = py;
            Ok(Some(Self { inner: filtered }))
        }
    }

    fn __len__(&self) -> usize {
        self.inner.n_conformers()
    }

    fn __getitem__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let n = slf.inner.n_conformers();
        if let Ok(idx) = key.extract::<isize>() {
            let n_i = n as isize;
            let real = if idx < 0 { idx + n_i } else { idx };
            if real < 0 || real >= n_i {
                return Err(PyIndexError::new_err("conformer index out of range"));
            }
            let geo = slf.inner.geometry(real as usize).unwrap();
            let py_geo = PyGeometry { inner: geo };
            let energy = slf.inner.energies()[real as usize];
            return Ok(PyTuple::new_bound(
                py,
                [py_geo.into_py(py), energy.into_py(py)],
            )
            .into_py(py));
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let indices = slice.indices(n as isize)?;
            let start = indices.start as isize;
            let stop = indices.stop as isize;
            let step = indices.step as isize;
            let mut geos: Vec<PyGeometry> = Vec::new();
            let mut energies: Vec<f64> = Vec::new();
            let push_at = |i: isize,
                          geos: &mut Vec<PyGeometry>,
                          energies: &mut Vec<f64>| {
                geos.push(PyGeometry {
                    inner: slf.inner.geometry(i as usize).unwrap(),
                });
                energies.push(slf.inner.energies()[i as usize]);
            };
            if step > 0 {
                let mut i = start;
                while i < stop {
                    push_at(i, &mut geos, &mut energies);
                    i += step;
                }
            } else if step < 0 {
                let mut i = start;
                while i > stop {
                    push_at(i, &mut geos, &mut energies);
                    i += step;
                }
            } else {
                return Err(PyValueError::new_err("slice step cannot be zero"));
            }
            // Build a new ensemble from the slice.
            let atom_types = slf.inner.atom_types().to_vec();
            let coords: Vec<Array2<f64>> =
                geos.into_iter().map(|g| g.inner.into_coords()).collect();
            let ens = RsEnsemble::new(atom_types, coords, energies)
                .map_err(rs_err_to_py)?;
            return Ok(PyEnsemble { inner: ens }.into_py(py));
        }
        Err(PyTypeError::new_err("index must be int or slice"))
    }

    fn __repr__(&self) -> String {
        format!(
            "ConformerEnsemble(n_conformers={}, n_atoms={})",
            self.inner.n_conformers(),
            self.inner.n_atoms(),
        )
    }
}

// ---------------------------------------------------------------------------
// Module-level functions

/// Identifies the active backend; useful for downstream wrappers that
/// support multiple implementations.
#[pyfunction]
pub fn backend_name() -> &'static str {
    "rust"
}

/// Direct pairwise check matching chemtrayzer's `is_different_conformer`.
#[pyfunction]
#[pyo3(signature = (a, b, options, energy_a = None, energy_b = None))]
pub fn is_different_conformer(
    a: &PyGeometry,
    b: &PyGeometry,
    options: &Bound<'_, PyDiff>,
    energy_a: Option<f64>,
    energy_b: Option<f64>,
) -> PyResult<bool> {
    let py_diff = options.borrow();
    let energy_threshold = py_diff.energy_threshold;
    let mirror_check = py_diff.mirror_check;
    let py_geo: PyRef<'_, PyGeoDiff> = PyRef::into_super(py_diff);
    let rs = RsDiff {
        rmsd_threshold: py_geo.rmsd_threshold,
        rot_threshold: py_geo.rot_threshold,
        energy_threshold,
        mass_weighted: py_geo.mass_weighted,
        rigid_transform: py_geo.rigid_transform,
        mirror_check,
    };
    rs_is_diff(
        a.inner.coords(),
        b.inner.coords(),
        a.inner.atom_types(),
        energy_a,
        energy_b,
        &ValidatedMappings::empty(),
        &rs,
    )
    .map_err(rs_err_to_py)
}

// ---------------------------------------------------------------------------
// QcpRmsdWorkspace bindings (f32 / f64) — reusable workspaces for the
// optimal rotational RMSD via Theobald/Liu (QCP). Coordinates are
// expected to be `(n_atoms, 3)` arrays of matching `dtype`. The fast
// path is taken when the array is **Fortran-ordered** (i.e.
// `np.asfortranarray(coords)`); other layouts are accepted but require
// an internal SoA gather (still zero allocation in the workspace).
// ---------------------------------------------------------------------------

macro_rules! define_pyqcp_ws {
    ($pycls:ident, $py_name:literal, $rust_ty:ty, $scalar:ty) => {
        #[pyclass(name = $py_name, module = "conformerensemblers")]
        pub struct $pycls {
            inner: $rust_ty,
        }

        #[pymethods]
        impl $pycls {
            #[new]
            #[pyo3(signature = (n_atoms))]
            fn new(n_atoms: usize) -> Self {
                Self {
                    inner: <$rust_ty>::new(n_atoms),
                }
            }

            #[getter]
            fn n_atoms(&self) -> usize {
                self.inner.n_atoms()
            }

            /// Prepares the reference geometry for repeated comparisons.
            /// Pass a Fortran-ordered `(n_atoms, 3)` array of matching
            /// dtype for the zero-copy fast path.
            fn prepare_reference(
                &mut self,
                coords: PyReadonlyArray2<'_, $scalar>,
            ) -> PyResult<()> {
                self.inner
                    .prepare_reference(coords.as_array())
                    .map_err(rs_err_to_py)
            }

            /// RMSD of `coords` against the previously prepared reference.
            fn rmsd_prepared(
                &mut self,
                coords: PyReadonlyArray2<'_, $scalar>,
            ) -> PyResult<f64> {
                self.inner
                    .rmsd_prepared(coords.as_array())
                    .map_err(rs_err_to_py)
            }

            /// One-shot RMSD of `left` against `right`.
            fn rmsd(
                &mut self,
                left: PyReadonlyArray2<'_, $scalar>,
                right: PyReadonlyArray2<'_, $scalar>,
            ) -> PyResult<f64> {
                self.inner
                    .rmsd(left.as_array(), right.as_array())
                    .map_err(rs_err_to_py)
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(n_atoms={})",
                    $py_name,
                    self.inner.n_atoms()
                )
            }
        }
    };
}

define_pyqcp_ws!(
    PyQcpWsF64,
    "QcpRmsdWorkspaceF64",
    QcpRmsdWorkspaceT<f64>,
    f64
);
define_pyqcp_ws!(
    PyQcpWsF32,
    "QcpRmsdWorkspaceF32",
    QcpRmsdWorkspaceF32,
    f32
);

/// Optimal rotational RMSD (QCP/Theobald) in `f64`.
#[pyfunction]
#[pyo3(name = "qcp_rmsd_f64")]
pub fn py_qcp_rmsd_f64(
    left: PyReadonlyArray2<'_, f64>,
    right: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    rs_qcp_rmsd(left.as_array(), right.as_array()).map_err(rs_err_to_py)
}

/// Optimal rotational RMSD (QCP/Theobald) in `f32`. The cross-covariance
/// is accumulated in `f32`; the eigenvalue root-find is still done in
/// `f64`, so the returned scalar is `f64`.
#[pyfunction]
#[pyo3(name = "qcp_rmsd_f32")]
pub fn py_qcp_rmsd_f32(
    left: PyReadonlyArray2<'_, f32>,
    right: PyReadonlyArray2<'_, f32>,
) -> PyResult<f64> {
    let mut ws = QcpRmsdWorkspaceF32::new(left.as_array().nrows());
    ws.rmsd(left.as_array(), right.as_array()).map_err(rs_err_to_py)
}

// ---------------------------------------------------------------------------
// Module init

#[pymodule]
fn conformerensemblers(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyEnsemble>()?;
    m.add_class::<PyGeoDiff>()?;
    m.add_class::<PyDiff>()?;
    m.add_class::<PyFilter>()?;
    m.add_class::<PyQcpWsF64>()?;
    m.add_class::<PyQcpWsF32>()?;
    m.add_function(wrap_pyfunction!(backend_name, m)?)?;
    m.add_function(wrap_pyfunction!(is_different_conformer, m)?)?;
    m.add_function(wrap_pyfunction!(py_qcp_rmsd_f64, m)?)?;
    m.add_function(wrap_pyfunction!(py_qcp_rmsd_f32, m)?)?;
    m.add("__doc__", "Rust-backed conformer ensemble primitives.")?;
    Ok(())
}
