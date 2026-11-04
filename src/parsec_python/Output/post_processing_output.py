"""``parsec.out`` lines of the non-self-consistent run, the DOS input and ``Calc_Hartree_G``.

The input echo follows ``usrinputfile.F90`` (l.907-912 for
``Calc_Hartree_G``, l.1068-1103 for ``nscf_kpoints``, l.1126-1176 for
``Create_DOS``); the NSCF block follows ``parsec.F90`` l.2087-2093,
``calc_nscf.F90`` (maximum matrix-vector count, diagonalization time) and
the NSCF branch of ``flevel.f90`` (Fermi level, electrons per spin, the
eigenvalues of every k-point).  List-directed character records start with a
blank, as gfortran writes them.
"""

from __future__ import annotations

import numpy as np

#: ``rydberg`` of ``const.f90``.
RYDBERG_TO_EV = 13.6058


def post_processing_header_lines(translation) -> list[str]:
    """The input echo of ``Calc_Hartree_G``, ``nscf_kpoints`` and ``Create_DOS``.

    Empty for an input that gives none of them.
    """

    lines: list[str] = []
    problem = translation.problem
    if getattr(problem.hartree, "periodic_g_space", False):
        lines += [
            "",
            " CAUTION: The Hartree potential will be calculated in reciprocal space",
            "",
        ]
    nscf = getattr(translation, "nscf", None)
    if nscf is not None:
        lines += [
            "  Calculating non self consistent electronic structure",
            "  Kpoints (units 1/Bohr) and sampling weights",
        ]
        for k_point, weight in zip(nscf.k_points, nscf.weights):
            lines.append(
                f"   {k_point[0]:8.5f} {k_point[1]:8.5f} {k_point[2]:8.5f} {weight:8.5f}"
            )
    gw = getattr(translation, "gw", None)
    if gw is not None:
        # usrinputfile.F90 l.2481-2530 (the grid lines with nscf_kpoints only).
        lines.append(" Writing files for GW calculation using BerkeleyGW")
        if nscf is not None:
            n1, n2, n3 = gw.kgrid
            s1, s2, s3 = gw.kshift
            lines += [
                "  Non-self consistent kgrid dimensions",
                f" {n1:3d} X{n2:3d} X{n3:3d}",
                "  Non-selfconsistent kgrid shift",
                f" {s1:7.3f} X{s2:7.3f} X{s3:7.3f}",
            ]
    dos = getattr(translation, "dos", None)
    if dos is not None and problem.periodic_cell is not None:
        lines.append("  Full DOS creation is on")
        if dos.monkhorst_pack_grid is not None:
            n1, n2, n3 = dos.monkhorst_pack_grid
            lines += ["  DOS MP grid dimensions specified", f" {n1:3d} X{n2:3d} X{n3:3d}"]
        if dos.monkhorst_pack_shift is not None:
            s1, s2, s3 = dos.monkhorst_pack_shift
            lines += ["  DOS MP shift specified", f" {s1:7.3f} X{s2:7.3f} X{s3:7.3f}"]
    return lines


def _state_lines(eigenvalues: np.ndarray, occupations: np.ndarray) -> list[str]:
    """``flevel.f90`` format 63, representation 1."""

    return [
        f"{index:5d}   {value:18.10f}   {value * RYDBERG_TO_EV:18.10f} {occupation:9.4f}   {1:6d}"
        for index, (value, occupation) in enumerate(zip(eigenvalues, occupations), start=1)
    ]


def _spinor_state_lines(
    eigenvalues: np.ndarray, occupations: np.ndarray, spin_z: np.ndarray
) -> list[str]:
    """``flevel.f90`` format 65 (spinors), representation 1."""

    return [
        f"{index:5d}   {value:14.6f}   {value * RYDBERG_TO_EV:14.6f} {occupation:9.4f}"
        f"   {1:6d}   {moment:9.4f}"
        for index, (value, occupation, moment) in enumerate(
            zip(eigenvalues, occupations, spin_z), start=1
        )
    ]


def nscf_report_lines(
    result,
    lattice_vectors: np.ndarray,
    *,
    matvec_budget: int,
    diagonalization_seconds: float,
) -> list[str]:
    """The NSCF block of ``parsec.out`` for a :class:`parsec_python.SCF.nscf.NSCFResult`.

    ``matvec_budget`` is ``solver%maxmv`` (``max(Maximum_matvec, n_grid *
    nscf_states_num)``), ``lattice_vectors`` (rows) give the k-points in
    reciprocal-lattice coordinates (``latt_vec^T k / 2 pi``).
    """

    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    lines = [
        "",
        " calling calc_nscf ",
        "",
        f"  Maximum number of matrix-vector operations = {int(matvec_budget):12d}",
        "",
        "",
        (
            f" Diagonalization time [sec] : {diagonalization_seconds:10.2f},"
            f"      tdiag_sum ={diagonalization_seconds:12.2f}"
        ),
        "",
    ]
    spin = len(result.channels) == 2
    spin_z = getattr(result, "spin_z", None)
    if spin_z is not None:
        # flevel.f90 l.334-351 for is_so (mxwd = 2, not non-collinear).
        lines += [
            "",
            " Spin-Orbit: each state has a magnetic moment!",
            "    Spin is not a good quantum number here!",
            " ---------------------------------------------",
            f" Net magnetic moment (in Bohr magnetons): {result.spin_orbit_moment:7.2f}",
            "",
        ]
    if spin:
        up, down = result.electrons
        lines += [
            "",
            f" Electrons in up, down spins: {up:7.2f} {down:7.2f}",
            f" Net magnetic moment (in Bohr magnetons): {up - down:7.2f}",
            "",
        ]
    lines.append(f" Fermi level at {result.fermi_level:10.4f} [Ry]")
    # A spinor channel is labelled "up" (flevel.f90: jj = isp - 1 + nspin
    # with nspin = 2).
    labels = ("up", "dn") if spin else (("up",) if spin_z is not None else ("  ",))
    for channel, label in enumerate(labels):
        lines.append("")
        for k_index, k_point in enumerate(result.k_points):
            fractional = lattice_vectors @ k_point / (2.0 * np.pi)
            lines += [
                f" k-point = {k_point[0]:9.4f}{k_point[1]:9.4f}{k_point[2]:9.4f} (a.u.)^-1",
                (
                    f" k-point = {fractional[0]:9.4f}{fractional[1]:9.4f}"
                    f"{fractional[2]:9.4f} (recip. latt. vec.) "
                ),
                (
                    f"{label} State   Eigenv.[Ry]      Eigenv.[eV]    Occup.     Repr.   "
                    "Spin_z [hbar]"
                    if spin_z is not None
                    else f"{label} State   Eigenvalue [Ry]      Eigenvalue [eV]    Occup.     Repr."
                ),
                "",
                *(
                    _spinor_state_lines(
                        result.eigenvalues[channel, k_index],
                        result.occupations[channel, k_index],
                        spin_z[k_index],
                    )
                    if spin_z is not None
                    else _state_lines(
                        result.eigenvalues[channel, k_index],
                        result.occupations[channel, k_index],
                    )
                ),
                "",
            ]
    lines.append(" ...NSCF calculation done")
    return lines


__all__ = ["nscf_report_lines", "post_processing_header_lines"]
