# ConformerEnsemble

ConformerEnsemble is a single package folder that combines the Rust core,
PyO3 bindings, Python API, tests, and CLI for conformer ensemble work.

The merged layout keeps the Rust crate as the implementation root and ships a
canonical Python package, `conformerensemble`, from the same maturin build.
For compatibility, the distribution still exposes `conformerensemblers` as the
extension package and `conformerensemblepy` as a thin Python shim.

## What is in this folder

- `src/`: Rust library and PyO3 bindings.
- `python/conformerensemble/`: canonical Python API and `crest-filter` CLI.
- `python/conformerensemblepy/`: compatibility re-exports for the old wrapper
  import path.
- `tests/`: Rust integration tests plus Python API and CLI tests.

## Python usage

```python
from conformerensemble import ConformerEnsemble, Geometry

geo = Geometry(
    atom_types=["C", "H"],
    coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
)
ensemble = ConformerEnsemble(geos=[geo], energies=[0.0])
print(ensemble.xyz_str())
```

Install an editable development build from this folder:

```powershell
python -m pip install -e .
```

The installed CLI is:

```powershell
crest-filter input.xyz -o filtered.xyz
```

## Development

Run both Python and Rust tests from the same folder:

```powershell
pixi run test
```

Run only one side when needed:

```powershell
pixi run test-python
pixi run test-rust
```

Rust-only workflows are still available:

```powershell
cargo test
cargo bench --bench rmsd
```

## License

MIT
