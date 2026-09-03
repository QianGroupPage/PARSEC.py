# H2 full-POTRE physical comparison

This is the first runnable end-to-end PARSEC-versus-Python comparison. The
input, full 861-point Martins-new pseudopotential, and PARSEC reference output
come from:

```text
<PARSEC_FORTRAN_ROOT>/tests/H2/python_pp/
```

The potential is a full PARSEC-format reconstruction of the original Python
hydrogen radial table. It contains `s` and `p` channels; `s` is local and the
unoccupied `p` channel produces six Kleinman-Bylander projector columns for
the two H atoms. It is not the six-point synthetic unit-test fixture.

The paired PARSEC calculation uses no Ono-Hirose double grid, so every enabled
physical setting is within the current Python implementation.

Reference PARSEC result:

```text
converged SCF iteration       19
weighted SRE                  0.0000658059 Ry
total energy                 -2.29319728 Ry
occupied eigenvalue          -0.7558302836 Ry
Hartree energy                2.58928199 Ry
exchange-correlation energy  -1.29868597 Ry
electron-ion energy          -7.04547304 Ry
ion-ion energy                1.41113867 Ry
```

Run Python from the repository root:

```powershell
python src\parsec_python\main.py examples\h2_full_nonlocal\parsec.in --dry-run
python src\parsec_python\main.py examples\h2_full_nonlocal\parsec.in --no-archive
```

The Python log is written here as `parsec.out`. The source PARSEC output
is retained as `parsec_reference.out`.

The completed numerical comparison is in `COMPARISON.md`.
