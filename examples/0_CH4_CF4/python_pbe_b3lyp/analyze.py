#!/usr/bin/env python3
"""Compare the Python PP-PBE(B3LYP) core-hole calculations with the ARES runs.

Every case is one PBE SCF followed by one non-self-consistent B3LYP
evaluation on its orbitals (``Energy_Refinement: b3lyp``; ARES ``Lrefine=T``).
The comparable quantities are the reference-corrected all-electron totals:
the ordinary and the C-1s-hole pseudopotentials have different ionic charges,
so their pseudopotential total energies do not share an energy zero.

    BE(C 1s) = E_AE(FS_1s) - E_AE(IS)

The Python energies are read in Rydberg and converted with the CODATA value
of the Rydberg energy.  The ``[eV]`` lines of ``parsec.out`` use PARSEC's
display constant 13.6058, which is 7.9e-6 too large: 2.3 meV on a binding
energy of 297 eV.  ARES converts with 27.21138 eV per Hartree.

Usage:
    python analyze.py                 # the parsec.out of the four cases
    python analyze.py --log NAME      # another log name inside the case folders
    python analyze.py --root DIR      # another tree of {CH4,CF4}/{IS,FS_1s}

A run writes ``NAME.out``; the copy kept with the example is
``NAME_python.out`` and is read where the run's own log is absent.  The ARES
values are read from the ``ares.log`` files of the ``b3lyp`` tree and, where
those are absent, from the copy of their numbers below.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARES = HERE.parent / "0_ARES_CH4_CF4" / "test_CH4_CF4" / "b3lyp"
MOLECULES = ("CH4", "CF4")
STATES = ("IS", "FS_1s")
RYDBERG_EV = 13.605693122994

# Gas-phase experiment and the values of the paper (Xu, Prendergast, Qian,
# J. Chem. Theory Comput. 18, 5471 (2022), Table S1), in eV.
EXPERIMENT = {"CH4": 290.80, "CF4": 301.85}
PAPER = {
    "PP-PBE": {"CH4": 297.146, "CF4": 307.715},
    "PP-PBE(B3LYP)": {"CH4": 297.232, "CF4": 308.425},
}

NUMBER = r"([-+]?[0-9]+\.[0-9]+(?:[Ee][-+]?[0-9]+)?)"
# Label in parsec.out -> name; all in Rydberg.
PYTHON_FIELDS = {
    "hf": r"Exact \(Hartree-Fock\) exchange\s*=\s*" + NUMBER,
    "slater": r"Slater exchange\s*=\s*" + NUMBER,
    "b88": r"Becke 88 exchange\s*=\s*" + NUMBER,
    "vwn": r"VWN \(RPA\) correlation\s*=\s*" + NUMBER,
    "lyp": r"Lee-Yang-Parr correlation\s*=\s*" + NUMBER,
    "xc_pbe": r"Exc \(PBE, self-consistent\)\s*=\s*" + NUMBER,
    "xc_b3lyp": r"Exc \(B3LYP\)\s*=\s*" + NUMBER,
    "total_pbe": r"Total Energy \(PBE\)\s*=\s*" + NUMBER,
    "total_b3lyp": r"Refined Total Energy \(B3LYP\)\s*=\s*" + NUMBER + r" \[Ry\]",
    "ae_pbe": r"PBE all-electron total \(as above\)\s*:\s*" + NUMBER,
    "ae_b3lyp": r"B3LYP-refined all-electron total\s*:\s*" + NUMBER + r" \[Ry\]",
}
PYTHON_TIMINGS = {
    "scf_seconds": r"Time for self-consistent field \[sec\] :\s*" + NUMBER,
    "refinement_seconds": r"Time for energy refinement \[sec\] :\s*" + NUMBER,
    "pair_seconds": r"Time per pair-density Poisson solve \[sec\] :\s*" + NUMBER,
    "total_seconds": r"Total accelerated Python wall time \[sec\] :\s*" + NUMBER,
    "memory_after": r"Process memory high water \[GiB\] :\s*" + NUMBER,
    "memory_before": r"after the refinement,\s*" + NUMBER + r" before it",
}
PYTHON_COUNTS = {
    "orbitals": r"Exact exchange:\s*(\d+) occupied orbitals",
    "pairs": r"occupied orbitals,\s*(\d+) pair-density Poisson solves",
    "iterations": r"SCF iter #\s*(\d+)",
}
# Label in ares.log -> name; all in eV.
ARES_FIELDS = {
    "hf": r"\*HF-X\s*->\s*" + NUMBER,
    "slater": r"\*Slater-X\s*->\s*" + NUMBER,
    "b88": r"\*B88-X\s*->\s*" + NUMBER,
    "vwn": r"\*VWN1RPA-C\s*->\s*" + NUMBER,
    "lyp": r"\*LYP-C\s*->\s*" + NUMBER,
    "xc_pbe": r"\*XC energy\s*->\s*" + NUMBER,
    "xc_b3lyp": r"\*B3LYP-XC\s*->\s*" + NUMBER,
    "total_pbe": r"\*Total-energy\s*->\s*" + NUMBER,
    "total_b3lyp": r"\*ref\.Total-energy->\s*" + NUMBER,
    "ae_pbe": r"\*AE energy \(test\)->\s*" + NUMBER,
    "ae_b3lyp": r"\*ref\.AE energy\s*->\s*" + NUMBER,
}
# The numbers of the four ARES logs (eV), for a checkout without them.
ARES_REFERENCE = {
    "CH4": {
        "IS": {
            "hf": -79.4511076,
            "slater": -72.6679043,
            "b88": -79.3529742,
            "vwn": -15.1734967,
            "lyp": -5.702502,
            "xc_pbe": -85.7796042,
            "xc_b3lyp": -86.3397862,
            "total_pbe": -219.7624519,
            "total_b3lyp": -220.322634,
            "ae_pbe": -1101.487739,
            "ae_b3lyp": -1102.047921,
        },
        "FS_1s": {
            "hf": -90.2656663,
            "slater": -83.6053481,
            "b88": -90.9038639,
            "vwn": -15.9123225,
            "lyp": -6.0932367,
            "xc_pbe": -97.6763144,
            "xc_b3lyp": -98.1512061,
            "total_pbe": -321.7157972,
            "total_b3lyp": -322.1906888,
            "ae_pbe": -804.340159,
            "ae_b3lyp": -804.8150507,
        },
    },
    "CF4": {
        "IS": {
            "hf": -512.3035104,
            "slater": -483.8659784,
            "b88": -518.449392,
            "vwn": -71.4697692,
            "lyp": -30.1719528,
            "xc_pbe": -549.92709,
            "xc_b3lyp": -552.4720805,
            "total_pbe": -2781.3293316,
            "total_b3lyp": -2783.874322,
            "ae_pbe": -11909.4306244,
            "ae_b3lyp": -11911.9756148,
        },
        "FS_1s": {
            "hf": -514.6730918,
            "slater": -488.9124284,
            "b88": -524.0287982,
            "vwn": -71.8605367,
            "lyp": -30.2794695,
            "xc_pbe": -555.6934131,
            "xc_b3lyp": -557.5282196,
            "total_pbe": -2872.7151316,
            "total_b3lyp": -2874.5499381,
            "ae_pbe": -11601.7154992,
            "ae_b3lyp": -11603.5503056,
        },
    },
}
ROWS = (
    ("xc_pbe", "PBE XC"),
    ("xc_b3lyp", "B3LYP XC"),
    ("total_pbe", "PBE total"),
    ("total_b3lyp", "refined total"),
    ("ae_pbe", "PBE all-electron total"),
    ("ae_b3lyp", "refined all-electron total"),
)
PARTS = (
    ("hf", "HF exchange"),
    ("slater", "Slater exchange"),
    ("b88", "B88 exchange"),
    ("vwn", "VWN-RPA correlation"),
    ("lyp", "LYP correlation"),
)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _last(text: str, pattern: str, convert=float):
    matches = re.findall(pattern, text)
    return convert(matches[-1]) if matches else None


def _log_text(folder: Path, log: str) -> str | None:
    """The log of a run, or the copy kept with the example (``*_python.out``)."""

    text = _read(folder / log)
    if text is None:
        name = Path(log)
        text = _read(folder / f"{name.stem}_python{name.suffix}")
    return text


def load_python(root: Path, log: str) -> dict:
    """``{molecule: {state: {name: value in eV, timings...}}}``; ``{}`` if unfinished."""

    data: dict = {}
    for molecule in MOLECULES:
        data[molecule] = {}
        for state in STATES:
            text = _log_text(root / molecule / state, log)
            values: dict = {}
            if text is not None and "Energy refinement:" in text:
                for name, pattern in PYTHON_FIELDS.items():
                    value = _last(text, pattern)
                    values[name] = None if value is None else value * RYDBERG_EV
                for name, pattern in PYTHON_TIMINGS.items():
                    values[name] = _last(text, pattern)
                for name, pattern in PYTHON_COUNTS.items():
                    values[name] = _last(text, pattern, int)
                values["converged"] = "Self-consistency convergence achieved." in text
                if any(values.get(name) is None for name in PYTHON_FIELDS):
                    values = {}
            data[molecule][state] = values
    return data


def load_ares() -> dict:
    """The ARES energies in eV: from the logs, else from ``ARES_REFERENCE``."""

    data: dict = {}
    for molecule in MOLECULES:
        data[molecule] = {}
        for state in STATES:
            text = _read(ARES / molecule / state / "ares.log")
            if text is not None and "Well Done" in text:
                values = {
                    name: _last(text, pattern) for name, pattern in ARES_FIELDS.items()
                }
            else:
                values = dict(ARES_REFERENCE[molecule][state])
            data[molecule][state] = values
    return data


def _difference(first, second):
    return None if first is None or second is None else first - second


def _number(value, width: int = 16, digits: int = 6) -> str:
    return f"{'unfinished':>{width}s}" if value is None else f"{value:{width}.{digits}f}"


def binding_energy(cases: dict, key: str):
    initial, final = cases["IS"].get(key), cases["FS_1s"].get(key)
    return _difference(final, initial)


def print_cases(python: dict, ares: dict, rows, title: str) -> None:
    print(title)
    print(f"{'case':<10s} {'quantity':<27s} {'Python':>16s} {'ARES':>16s} {'Python-ARES':>13s}")
    print("-" * 86)
    for molecule in MOLECULES:
        for state in STATES:
            for key, label in rows:
                py_value = python[molecule][state].get(key)
                ref_value = ares[molecule][state].get(key)
                print(
                    f"{molecule + ' ' + state:<10s} {label:<27s} {_number(py_value)} "
                    f"{_number(ref_value)} {_number(_difference(py_value, ref_value), 13)}"
                )
        print()


def print_binding(python: dict, ares: dict) -> None:
    print("C 1s Delta-SCF binding energies (eV)")
    print(
        f"{'quantity':<26s} {'Python':>12s} {'ARES':>12s} {'Python-ARES':>12s} "
        f"{'paper':>10s} {'experiment':>11s}"
    )
    print("-" * 88)
    for key, method in (("ae_pbe", "PP-PBE"), ("ae_b3lyp", "PP-PBE(B3LYP)")):
        py_be = {m: binding_energy(python[m], key) for m in MOLECULES}
        ref_be = {m: binding_energy(ares[m], key) for m in MOLECULES}
        for molecule in MOLECULES:
            print(
                f"{method + ' BE(' + molecule + ')':<26s} {_number(py_be[molecule], 12, 4)} "
                f"{_number(ref_be[molecule], 12, 4)} "
                f"{_number(_difference(py_be[molecule], ref_be[molecule]), 12, 4)} "
                f"{PAPER[method][molecule]:10.3f} {EXPERIMENT[molecule]:11.2f}"
            )
        py_shift = _difference(py_be["CF4"], py_be["CH4"])
        ref_shift = _difference(ref_be["CF4"], ref_be["CH4"])
        print(
            f"{method + ' CF4-CH4':<26s} {_number(py_shift, 12, 4)} "
            f"{_number(ref_shift, 12, 4)} {_number(_difference(py_shift, ref_shift), 12, 4)} "
            f"{PAPER[method]['CF4'] - PAPER[method]['CH4']:10.3f} "
            f"{EXPERIMENT['CF4'] - EXPERIMENT['CH4']:11.2f}"
        )
        print()
    print("B3LYP correction of the binding energy, [E_xc(B3LYP)-E_xc(PBE)](FS) - (IS) (eV)")
    for molecule in MOLECULES:
        corrections = []
        for source in (python, ares):
            pbe = binding_energy(source[molecule], "ae_pbe")
            refined = binding_energy(source[molecule], "ae_b3lyp")
            corrections.append(_difference(refined, pbe))
        print(
            f"{molecule:<26s} {_number(corrections[0], 12, 4)} {_number(corrections[1], 12, 4)} "
            f"{_number(_difference(*corrections), 12, 4)}"
        )


def print_timings(python: dict) -> None:
    print("\nTime and memory of the Python runs")
    print(
        f"{'case':<10s} {'SCF steps':>9s} {'SCF [s]':>9s} {'orbitals':>8s} {'pairs':>6s} "
        f"{'refinement [s]':>14s} {'s/pair':>7s} {'total [s]':>10s} "
        f"{'host GiB before/after':>22s}"
    )
    print("-" * 104)
    for molecule in MOLECULES:
        for state in STATES:
            values = python[molecule][state]
            if not values:
                print(f"{molecule + ' ' + state:<10s} unfinished")
                continue
            print(
                f"{molecule + ' ' + state:<10s} {values['iterations']:9d} "
                f"{values['scf_seconds']:9.1f} {values['orbitals']:8d} {values['pairs']:6d} "
                f"{values['refinement_seconds']:14.1f} {values['pair_seconds']:7.3f} "
                f"{values['total_seconds']:10.1f} "
                f"{values['memory_before']:10.2f} /{values['memory_after']:6.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log", default="parsec.out", help="log name in each case folder")
    parser.add_argument("--root", type=Path, default=HERE, help="tree of the Python cases")
    parser.add_argument(
        "--parts", action="store_true", help="also compare the five parts of E_xc(B3LYP)"
    )
    arguments = parser.parse_args()

    python = load_python(arguments.root.resolve(), arguments.log)
    ares = load_ares()
    for molecule in MOLECULES:
        for state in STATES:
            values = python[molecule][state]
            if values and not values["converged"]:
                print(f"WARNING: {molecule} {state}: the SCF did not converge")

    print_cases(python, ares, ROWS, "Energies of the four cases (eV)")
    if arguments.parts:
        print_cases(python, ares, PARTS, "Parts of E_xc(B3LYP), unweighted (eV)")
    print_binding(python, ares)
    print_timings(python)


if __name__ == "__main__":
    main()
