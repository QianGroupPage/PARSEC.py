#!/usr/bin/env python3
"""C 1s binding energies of CH4 and CF4 with the library pseudopotentials.

    BE    = E_AE(FS_1s) - E_AE(IS)        reference-corrected all-electron totals
    shift = BE(CF4) - BE(CH4)

Compared with all-electron calculations and with experiment.
Energies of the solver logs are converted from Rydberg with the CODATA factor.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RY_EV = 13.605693122994
TOTAL = re.compile(r"Reference-corrected all-electron total\s*:\s*([-+0-9.Ee]+)\s*\[Ry\]")
EXPERIMENT = {"CH4": 290.80, "CF4": 301.85}

# All-electron references of this study: PySCF 2.14, spin-free X2C, aug-cc-pwCVQZ on C
# and aug-cc-pVQZ on H and F, grid level 5, geometries of the IS inputs (see README.md).
# "unpolarized": occupation 1 in the C 1s orbital of a spin-restricted calculation, the
# all-electron analogue of the pseudopotential calculation.  "polarized": one C 1s
# electron of one spin removed, spin-unrestricted, maximum-overlap method.
AE = {
    "pbe": {"unpolarized": {"CH4": 297.0188, "CF4": 307.2903},
            "polarized": {"CH4": 289.9993, "CF4": 300.2075}},
    # LDA: Slater exchange with Perdew-Zunger 1981 correlation (LDA_X,LDA_C_PZ)
    "ca": {"unpolarized": {"CH4": 293.8324, "CF4": 304.3822},
           "polarized": {"CH4": 287.0743, "CF4": 297.5473}},
}


def total(path: Path) -> float | None:
    for name in ("parsec.out", "parsec_python.out"):
        try:
            found = TOTAL.findall((path / name).read_text(errors="replace"))
        except OSError:
            continue
        if found:
            return float(found[-1]) * RY_EV
    return None


def binding(root: Path) -> dict[str, float | None]:
    out = {}
    for molecule in ("CH4", "CF4"):
        initial, final = total(root / molecule / "IS"), total(root / molecule / "FS_1s")
        out[molecule] = None if initial is None or final is None else final - initial
    return out


def row(label: str, values: dict[str, float | None]) -> str:
    a, b = values["CH4"], values["CF4"]
    if a is None or b is None:
        return f"{label:<50s} not available"
    return f"{label:<50s} {a:9.3f} {b:9.3f} {b - a:8.3f}"


def main() -> None:
    print(f"{'C 1s binding energy (eV)':<50s} {'CH4':>9s} {'CF4':>9s} {'shift':>8s}")
    print("-" * 80)
    print("PBE")
    print(row("  library potentials", binding(HERE / "pbe")))
    print(row("  all-electron, unpolarized hole", AE["pbe"]["unpolarized"]))
    print(row("  all-electron, polarized hole", AE["pbe"]["polarized"]))
    print("LDA (CA)")
    print(row("  library potentials", binding(HERE / "ca")))
    print(row("  all-electron, unpolarized hole", AE["ca"]["unpolarized"]))
    print(row("  all-electron, polarized hole", AE["ca"]["polarized"]))
    print(row("experiment", EXPERIMENT))


if __name__ == "__main__":
    main()
