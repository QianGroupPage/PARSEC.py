# Modular PARSEC-Style Isolated Single Points

This package is a self-contained, spin-unpolarized real-space DFT path for
isolated systems. It follows PARSEC's Rydberg-unit algorithms and reads
Martins-new `*_POTRE.DAT` files directly; it does not use `elements_new.csv` or
the older Python pseudopotential helpers.

This is a native Python implementation. Runtime calculations do not launch
PARSEC, WSL, MPI, or another executable, and do not bind Fortran through
`ctypes`, `cffi`, `f2py`, or a shared library. The first SCF solve executes
the explicitly selected native CHEBFF or CHEBDAV algorithm; every later solve
executes one translated SUBSPACE filter. ARPACK is never used as a fallback.

The implementation is intentionally decomposed so a caller can build a grid,
evaluate a local potential, construct KB projectors, solve Poisson, evaluate
CA-LDA, or run the complete SCF calculation independently.

## Run a PARSEC input file

Install the two runtime dependencies into the active environment:

```powershell
cd src\new_architecture
python -m pip install -r requirements.txt
```

Put `parsec.in` and each Martins-new pseudopotential beside one another. The
pseudopotential filename must be `<Atom_Type>_POTRE.DAT`:

```text
H2/
├── parsec.in
└── H_POTRE.DAT
```

From `src\new_architecture`, validate the input without starting the large
grid calculation:

```powershell
cd src\new_architecture
python main.py H2\parsec.in --dry-run
```

Then run the single point:

```powershell
python main.py H2\parsec.in
```

No `PYTHONPATH` setting is needed for this folder-local command. By default the
launcher writes these files beside `parsec.in`:

- `parsec.out`: PARSEC-style input/grid sections, per-iteration state
  tables, energy decomposition, SRE lines, convergence, and timing
- `parsec_python_results.npz`: atoms, grid, densities, potentials,
  eigenvalues, occupations, energy terms, and SCF history

`Output_All_States: true` also stores wavefunctions in the archive. Use
`--no-archive` to produce only the log, `--output NAME.npz` or `--log NAME` to
change paths, and `--pp-dir DIRECTORY` to explicitly override the adjacent
pseudopotential search. Relative output paths are resolved beside `parsec.in`,
not relative to the shell's current directory.

A successful converged run returns exit code 0. A completed run that reaches
`Max_Iter` without convergence still writes its results and returns exit code
3. Input errors return 2.

The reader accepts PARSEC/ESDF `:`, `=`, or whitespace separators, `#`, `;`,
or `!` comments, Fortran `D` exponents, dotted booleans, and matching
`begin`/`end` blocks. Its current physical scope is the one listed below;
unsupported options are rejected rather than silently approximated.

The text layout follows `parsec.out`, but its banner identifies
`PARSEC-PYTHON`. It omits unavailable quantities rather than inventing them:
forces, dipoles, point-group decomposition, per-representation statistics,
and MPI/build metadata are not reported.

## Physical PARSEC comparison

The requested calculation with PARSEC's canonical 863-point H potential and
the currently supported no-double-grid path is under
`benchmarks/h2_canonical_nodg`:

```powershell
python main.py benchmarks\h2_canonical_nodg\parsec.in --no-archive
```

It converges to `-2.13713415 Ry` and writes the PARSEC-style report to
`benchmarks/h2_canonical_nodg/parsec.out`. The stock PARSEC H2 result,
`-2.13713410 Ry`, used `Double_Grid_Order: 3`; this Python input uses order
one. See that benchmark's `README.md` for the component-by-component comparison
and why the close energies are not yet a strict like-for-like reproduction.

The completed full-pseudopotential H2 comparison is under
`benchmarks/h2_full_nonlocal`. It uses an 861-point POTRE file and six
nonlocal projector columns, not the synthetic test fixture:

```powershell
python main.py benchmarks\h2_full_nonlocal\parsec.in --dry-run
python main.py benchmarks\h2_full_nonlocal\parsec.in --no-archive
```

For this matched case, PARSEC gives `-2.29319728 Ry` and Python gives
`-2.293197288860 Ry`, an absolute difference of `8.86e-9 Ry`. See
`benchmarks/h2_full_nonlocal/COMPARISON.md` for the energy decomposition,
convergence values, timing, and limitations.

The larger 12-atom benzene comparison is under `benchmarks/0d_benzene`:

```powershell
python main.py benchmarks\0d_benzene\parsec.in --no-archive
```

Using the unchanged PARSEC example input and pseudopotentials, the native
CHEBFF/SUBSPACE Python path gives `-75.38863311 Ry` in 15 SCF iterations and
a newly generated PARSEC reference gives `-75.38863314 Ry` in 17 iterations,
a difference of `+3e-8 Ry`. See that benchmark's
`COMPARISON.md` for energy components, eigenvalues, convergence, and the
full-grid-versus-symmetry-reduced timing distinction.

Start with:

- [PARSEC_ALGORITHM.md](PARSEC_ALGORITHM.md): the reviewed Fortran call path,
  formulas, defaults, and source locations.
- [PYTHON_IMPLEMENTATION.md](PYTHON_IMPLEMENTATION.md): Python module map,
  examples, validation, supported scope, and known differences.
- [ARCHITECTURE.md](ARCHITECTURE.md): concept-package boundaries and the
  native-port rule.
- [provenance/source_map.json](provenance/source_map.json): machine-readable
  implementation and parity status for every calculation stage.

Minimal entry points:

```python
from new_architecture import prepare_single_point, run_single_point

system = prepare_single_point(problem)  # static terms, no SCF
result = run_single_point(problem)      # complete CA-LDA single point
```

Run the focused test suite from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s src\new_architecture\tests -v
```

Current scope: isolated sphere/box domains, full active grid without point-group
reduction, scalar norm-conserving Martins-new pseudopotentials through `l=3`,
optional NLCC, CA/PZ LDA, and potential-mixed SCF. Periodic systems, spin,
spin-orbit, forces, relaxation, DFT+U, and PARSEC restart files are outside this
first implementation. Ono-Hirose `Double_Grid_Order > 1` is also not yet
implemented. Sphere Hartree boundaries use multipoles; box boundaries use the
exact but slower direct Coulomb sum.
