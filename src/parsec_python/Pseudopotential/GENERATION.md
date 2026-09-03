# Pseudopotential generation

`pp_generation` is a Python API and CLI for reproducible pseudopotential
generation, format conversion, and Kleinman--Bylander ghost-state screening.
It currently orchestrates two audited numerical backends:

- **FHI98PP**: default; Hamann and Troullier--Martins NCPP, including explicit
  core-shell occupations and the validated core-hole cases in this checkout.
- **ATOM 6.x**: Troullier--Martins NCPP; independent generation/checking and
  broader native serialization.

This is not yet a pure-Python radial atomic solver. The public model separates
family, scheme, backend, and format so native Python kernels and future
ONCV/PAW/ultrasoft backends can be added without changing user input.

## Core-hole generation

Generate the validated PBE/Troullier--Martins Si 2p full-core-hole potential:

```bash
git clone https://github.com/QianGroupPage/PARSEC.py.git
cd PARSEC.py
export PARSEC_FORTRAN_ROOT=/path/to/PARSEC
PYTHONPATH=src python3 -m pp_generation Si \
  --backend fhi98pp \
  --fhi-root "$PARSEC_FORTRAN_ROOT/fhi98pp/adka_v1_0/Dfhipp" \
  --core-hole 2p --hole-charge 1 \
  --cutoff-radius 2.5 \
  --format fhi \
  --output-dir pp-output/si-2p-hole
```

The command tests every possible local angular channel by default, rejects
ghost/undetermined/ill-defined candidates, chooses the passing channel with
the largest worst-channel spectral margin, and records all candidates in a
JSON report. Force a reviewed choice with `--local-channel L`. Use
`--allow-ghosts` only to retain diagnostic output.

For a fractional transition-potential hole, use, for example,
`--core-hole 2p --hole-charge 0.5`.

In a molecular input, give the excited species a short configuration label
such as `C-1s`, and retain the chemical identity explicitly:

```text
Atom_Type: C-1s
Element_Symbol: C
```

The label selects `C-1s_POTRE.DAT`; `Element_Symbol` is used for element
validation and output.  A full 1s-hole carbon PP has ionic charge five rather
than four.  For a fully screened neutral-electron final state, pair it with
`Net_Charges: 1 e`.  Initial/final delta-SCF energies made with different PPs
must use the corresponding FHI98PP all-electron-minus-pseudo atomic reference
corrections (`Atomic_Energy_Correction` in the input); raw pseudo total
energies from the two calculations do not have a common zero.

For a core-hole request, the FHI98PP adapter emits this readable species alias
automatically (`C-1s_POTRE.DAT` and `C-1s_FHIPP.DAT` for a full C 1s hole),
while retaining the historical element-only artifact names for compatibility.
Fractional holes include the removed charge, for example
`C-1s-0.5_POTRE.DAT`.

## Output formats

Extensions alone are ambiguous, especially `.DAT`. The CLI uses semantic
format names:

| CLI name | Typical file | Meaning | Backend |
|---|---|---|---|
| `fhi` (`fhipp`) | `Si_FHIPP.DAT` | Lossless FHI CPI archive; usable by legacy Fortran PARSEC when configured, but not by the Python solver | FHI98PP |
| `parsec` (`potre`) | `Si_POTRE.DAT` | PARSEC Martins-new semilocal table required by the Python solver | both |
| `upf` | `*.UPF` | Quantum ESPRESSO UPF v2 | both |
| `psp8` | `*.psp8` | ABINIT PSP8 separable NCPP | ATOM |
| `siesta` (`psf`) | `*.psf` | SIESTA PSF | ATOM |
| `cpw2000` | `*_POTKB_F.DAT` | Fourier-space KB table | ATOM |

FHI-to-UPF/POTRE conversion requires the already built QE reader and the
repository converter:

```bash
export PARSEC_FORTRAN_ROOT=/path/to/PARSEC
PYTHONPATH=src python3 -m pp_generation P \
  --fhi-root "$PARSEC_FORTRAN_ROOT/fhi98pp/adka_v1_0/Dfhipp" \
  --qe-converter "$PARSEC_FORTRAN_ROOT/fhi98pp/adka_v1_0/Dfhipp/tools/fhi2upf_qe.x" \
  --potre-converter src/tools/upf_to_parsec.py \
  --core-hole 1s --cutoff-radius 1.95 \
  --format fhi --format upf --format parsec \
  --output-dir pp-output/p-1s-hole
```

Retain `*_FHIPP.DAT` as the lossless generator-grid reference. The Python
PARSEC implementation reads only `*_POTRE.DAT`; POTRE conversion resamples
radial fields and must be independently converged. The converter preserves
UPF `PP_NLCC` as PARSEC radial core charge and supports genuinely all-local
H/He potentials. UPF's `total_aeenergy` field written here is useful
provenance but is a workflow extension, not a standard UPF v2 attribute.

## Independent ATOM run

ATOM's local built-in writer produces CA-LDA defaults. PBE with ATOM requires
a reviewed `atom.dat` (the online portal maintains a separate PBE library):

```bash
export PARSEC_FORTRAN_ROOT=/path/to/PARSEC
PYTHONPATH=src python3 -m pp_generation Si \
  --backend atom --xc ca \
  --atom-executable "$PARSEC_FORTRAN_ROOT/pseudopotential/Src/atom_all_gfortran.exe" \
  --atom-kb-executable "$PARSEC_FORTRAN_ROOT/pseudopotential/Src/kb_conv_gfortran.exe" \
  --format parsec --format upf --format psp8 --format siesta --format cpw2000 \
  --output-dir pp-output/si-atom-ca
```

The adapter explicitly inspects `atom.out`; ATOM can return exit code zero
even when the atomic potential did not converge, which is treated as failure.
Automatic ATOM core-hole editing is intentionally disabled until an ATOM
core-hole reference is available.

## Validation

Run the standard-library suite (also discoverable by pytest):

```bash
PARSEC_FORTRAN_ROOT=/path/to/PARSEC \
PYTHONPATH=src python3 -m unittest -v pp_generation.tests.test_pp_generation
```

The suite regenerates Si 2p-hole and P 1s-hole FHI references, requires
byte-identical CPI/FHIPP/UPF/POTRE artifacts, checks energies and ionic charge,
scans every local channel, exercises a synthetic ghost failure, and
independently generates all five ATOM output formats.

Generator defaults remain starting points. Production approval also needs
logarithmic-derivative, excited/ionized-configuration, cutoff, and condensed-
phase transferability tests appropriate to the target chemistry.
