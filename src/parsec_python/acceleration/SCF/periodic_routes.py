"""Which periodic inputs of the round's features the accelerated driver runs, and which it routes.

The accelerated driver runs a Gamma-point SCF of a periodic cell on the
reference preparation (:func:`parsec_python.SCF.pbc.prepare_periodic_single_point`,
which hands a wire or slab to :mod:`parsec_python.SCF.wire_slab`) with its own
device eigensolver, and the Hartree solve of the prepared system
(:mod:`parsec_python.acceleration.Hartree.periodic_device` on the device).
That covers, at the Gamma point and without spin or forces (routed by their
own entries of :func:`parsec_python.acceleration.cli._reference_only_features`):

* ``Boundary_Conditions: wire`` / ``slab``, with charged sheets;
* ``Calc_Hartree_G`` (the spectral Hartree solve on the device);
* ``Periodic_Local_Potential: g_space`` and ``Periodic_Alpha_Term`` (in the
  reference preparation);
* ``Zero_Average_Vxc`` (the readable XC shifts its potential; the device
  evaluators are wrapped by :class:`parsec_python.acceleration.V_xc.device_xc.ZeroAverageXC`).

This module names what stays with the reference solver.
"""

from __future__ import annotations


def _mpi_run() -> bool:
    from ..Parallel import mpi

    return mpi.requested()


def wire_slab_reference_only(translation) -> list[str]:
    """The features of a wire or slab input that the accelerated driver does not run.

    A restart from ``parsec.dat`` (``Restart_Run``) of a wire or slab reads the
    stored fields through the reference CLI only, and an MPI run (``--mpi``)
    keeps the reference route it had before the accelerated one existed.
    """

    features = []
    if getattr(translation, "restart_run", False):
        features.append(
            "Restart_Run of a wire or slab (Boundary_Conditions: wire or slab)"
        )
    if _mpi_run():
        features.append(
            "a wire or slab in an MPI run (Boundary_Conditions: wire or slab with --mpi)"
        )
    return features


def cell_options_reference_only(translation) -> list[str]:
    """``Calc_Hartree_G``, ``g_space`` and ``Zero_Average_Vxc`` of a 3-D cell that stay routed.

    Only an MPI run (``--mpi``), which keeps the reference route it had before.
    """

    problem = translation.problem
    if problem.periodic_cell is None or not _mpi_run():
        return []
    names = []
    if getattr(problem.hartree, "periodic_g_space", False):
        names.append("the reciprocal-space Hartree potential (Calc_Hartree_G)")
    if getattr(problem.grid, "local_potential", "real_space") != "real_space":
        names.append("Fortran's G-space local potential (Periodic_Local_Potential: g_space)")
    if getattr(problem.scf, "zero_average_xc_potential", False):
        names.append("the zero-average exchange-correlation potential (Zero_Average_Vxc)")
    return [f"{name} in an MPI run (--mpi)" for name in names]


def moving_cell_options_reference_only(translation) -> list[str]:
    """Forces, relaxation and dynamics of a cell with the options above: on the reference solver.

    The accelerated forces of a periodic cell (``Forces/periodic_device.py``)
    are those of a bulk cell at the Gamma point on the readable or device
    fields of the real-space local potential, validated without these
    options (and with LDA+U and TS); the reference solver keeps the wire and
    slab forces and the forces of a cell with ``Calc_Hartree_G``, ``g_space``
    or ``Zero_Average_Vxc``.
    """

    problem = translation.problem
    if problem.periodic_cell is None or not (
        getattr(translation, "compute_forces", False)
        or getattr(translation, "relaxation", None) is not None
        or getattr(translation, "molecular_dynamics", None) is not None
    ):
        return []
    names = []
    if getattr(problem.grid, "periodicity", 3) in (1, 2):
        names.append("a wire or slab (Boundary_Conditions: wire or slab)")
    if getattr(problem.hartree, "periodic_g_space", False):
        names.append("Calc_Hartree_G")
    if getattr(problem.grid, "local_potential", "real_space") != "real_space":
        names.append("Periodic_Local_Potential: g_space")
    if getattr(problem.scf, "zero_average_xc_potential", False):
        names.append("Zero_Average_Vxc")
    # LDA+U (J = 0) and TS of a bulk cell at Gamma move on the accelerated
    # path: the LDA+U term from the orbitals on the device with the gradients
    # of the cell images (Hamiltonian/hubbard_device.py), the TS forces of the
    # device sums (Energy/dispersion_device.py), validated against the
    # reference forces, relaxations and dynamics.
    return [
        f"forces, relaxation or molecular dynamics of a periodic cell with {name}"
        for name in names
    ]


__all__ = [
    "cell_options_reference_only",
    "moving_cell_options_reference_only",
    "wire_slab_reference_only",
]
