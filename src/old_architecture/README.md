# Legacy Python implementation

This directory contains the Python implementation that previously lived
directly under `src/`. It is intentionally separate from
`src/new_architecture`, which is the native translation of PARSEC's
isolated single-point calculation.

From the repository root, run the refactored legacy driver with:

```bash
python src/old_architecture/main_new.py
python src/old_architecture/main_new.py --cpu path/to/input.in
```

It can also be imported as a package when `src` is on `PYTHONPATH`:

```python
from old_architecture.main_new import main
```

The principal folders retain the organization of the previous implementation:

- `Eigensolvers`
- `Laplacian`
- `Mixer`
- `Splines`
- `V_ion`
- `V_xc`
- `GUI`
- `Tools`
- `native`: C++/OpenMP acceleration kernels for `V_ion`

`elements_new.csv` and `V_ion/splineData.mat` remain beside the code that
loads them, so their resource paths do not depend on the shell's working
directory.
