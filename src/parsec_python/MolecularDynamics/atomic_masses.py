"""Atomic masses of PARSEC's molecular dynamics.

Translation of subroutine ``ptable`` (``ptable.f90``) and of the three lines
of ``pseudo.f90`` that turn its output into ``clust%amass``::

    call ptable(name(ity),am,ierr)
    clust%amass(kk) = am*1836.d0

``ptable`` returns the standard atomic weight (NIST SP 966, September 2003).
PARSEC converts it to electron masses with the factor 1836, the rounded
proton-to-electron mass ratio, and not with the 1822.888 electron masses of
one atomic mass unit; every PARSEC mass is therefore about 0.7 per cent
heavier than the physical one.  The factor is reproduced here because the
trajectory depends on it.

The table is reproduced as it is, including what looks like slips:

* ``Pr`` occurs twice in the Fortran ``if`` chain, first with 140.90765 and
  then with 145.0.  The second entry was meant for promethium and can never
  be reached, so praseodymium has 140.90765 and ``Pm`` is an unknown element.
* The type names ``H0`` to ``H9``, ``F0`` to ``F8``, ``HI`` to ``HR`` and
  ``FI`` to ``FQ`` all carry the mass of hydrogen, also the ones that begin
  with ``F``.
* ``uF`` has the mass of iron.

Names are compared as PARSEC does: the atom type name is a two-character
string, case matters, and a longer name is cut after two characters.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

#: ``clust%amass = am*1836.d0`` in ``pseudo.f90``.
ELECTRON_MASSES_PER_ATOMIC_WEIGHT = 1836.0

_HYDROGEN = 1.00794
_HYDROGEN_LIKE = (
    ["H"]
    + [f"H{digit}" for digit in range(10)]
    + [f"F{digit}" for digit in range(9)]
    + [f"H{letter}" for letter in "IJKLMNOPQR"]
    + [f"F{letter}" for letter in "IJKLMNOPQ"]
)

#: Atomic weights of ``ptable.f90`` in the order of the Fortran ``if`` chain.
PARSEC_ATOMIC_WEIGHTS: dict[str, float] = {
    **{name: _HYDROGEN for name in _HYDROGEN_LIKE},
    "Li": 6.941,
    "Na": 22.98977,
    "K": 39.0983,
    "Rb": 85.4678,
    "Cs": 132.90545,
    "Fr": 223.0,
    "Be": 9.012182,
    "Mg": 24.3050,
    "Ca": 40.078,
    "Sr": 87.62,
    "Ba": 137.327,
    "Ra": 226.0,
    "B": 10.811,
    "Al": 26.981538,
    "Ga": 69.723,
    "In": 114.818,
    "Tl": 204.3833,
    "C": 12.0107,
    "Si": 28.0855,
    "Ge": 72.64,
    "Sn": 118.710,
    "Pb": 207.2,
    "N": 14.0067,
    "P": 30.973761,
    "As": 74.92160,
    "Sb": 121.760,
    "Bi": 208.98038,
    "O": 15.9994,
    "S": 32.065,
    "Se": 78.96,
    "Te": 127.60,
    "Po": 209.00,
    "F": 18.9984032,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.90447,
    "At": 210.0,
    "He": 4.002602,
    "Ne": 20.1797,
    "Ar": 39.948,
    "Kr": 83.798,
    "Xe": 131.293,
    "Rn": 222.0,
    "Sc": 44.955910,
    "Y": 88.90585,
    "La": 138.9055,
    "Ac": 227.0,
    "Ti": 47.867,
    "Zr": 91.224,
    "Hf": 178.49,
    "V": 50.9415,
    "Nb": 92.90638,
    "Ta": 180.9479,
    "Cr": 51.9961,
    "Mo": 95.94,
    "W": 183.84,
    "Mn": 54.938049,
    "Tc": 98.0,
    "Re": 186.207,
    "Fe": 55.845,
    "uF": 55.845,
    "Ru": 101.07,
    "Os": 190.23,
    "Co": 58.933200,
    "Rh": 102.90550,
    "Ir": 192.217,
    "Ni": 58.6934,
    "Pd": 106.42,
    "Pt": 195.078,
    "Cu": 63.546,
    "Ag": 107.8682,
    "Au": 196.96655,
    "Zn": 65.409,
    "Cd": 112.411,
    "Hg": 200.59,
    "Ce": 140.116,
    "Pr": 140.90765,
    "Nd": 144.24,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.92534,
    "Dy": 162.500,
    "Ho": 164.93032,
    "Er": 167.259,
    "Tm": 168.93421,
    "Yb": 173.04,
    "Lu": 174.967,
    "Th": 232.0381,
    "Pa": 231.03588,
    "U": 238.02891,
}


def parsec_atom_name(name: str) -> str:
    """Return ``name`` as PARSEC stores it in ``clust%name`` (two characters)."""
    return name.strip()[:2].rstrip()


def atomic_weight(name: str) -> float:
    """Atomic weight of ``ptable`` for one atom type name.

    Raises ``KeyError`` for a name the Fortran table does not contain
    (PARSEC stops with error 201, ``unknown element``).
    """
    key = parsec_atom_name(name)
    try:
        return PARSEC_ATOMIC_WEIGHTS[key]
    except KeyError:
        raise KeyError(
            f"unknown element {name!r}: ptable.f90 has no atomic mass for it"
        ) from None


def atomic_masses(names: Iterable[str]) -> np.ndarray:
    """Masses of the atoms in electron masses, the unit of ``clust%amass``.

    ``names`` holds one atom type name per atom.  The result has shape
    ``(atoms,)`` and equals ``ptable(name)*1836``.
    """
    return np.array(
        [atomic_weight(name) * ELECTRON_MASSES_PER_ATOMIC_WEIGHT for name in names],
        dtype=float,
    )
