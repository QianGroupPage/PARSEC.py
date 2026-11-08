"""Post-SCF outputs of the accelerated CLI: the band structure of a Gamma-point cell and the cluster DOS.

The accelerated driver runs the Gamma-point SCF of a periodic cell and the SCF of a cluster.  Two
outputs that PARSEC computes after such an SCF need nothing the accelerated result lacks, so they run
here, after the accelerated SCF, with the readable post-processing of the reference CLI:

``bandstruc`` (``calc_bands``) after a Gamma-point SCF of a scalar periodic cell
    :func:`parsec_python.cli._run_band_structure` with the converged potential of the accelerated
    result and the reference system it was prepared from (the accelerated periodic preparation *is*
    the reference preparation).  Its k-points go through the band machinery of
    :mod:`parsec_python.SCF.bands`, i.e. through the device k-point workers of
    :mod:`parsec_python.acceleration.SCF.kpoint_parallel` when a GPU is visible: the same eigensolves
    as the reference CLI makes after its own Gamma-point SCF, which runs on the host.

``Create_DOS`` of a cluster (``dos.F90`` l.149-168)
    :func:`parsec_python.cli._run_dos`: the histogram of the occupations of the accelerated levels.
    Nothing of it is worth a device (a histogram of ``States_Num`` levels).

Everything else these features combine with stays on the reference solver by name
(:func:`acceleration.cli._reference_only_features`): k-point SCFs (whose eigensolves already run on
the device workers), spin and spin-orbit bands, LDA+U, wires and slabs, Fortran's general grid, the
periodic DOS (it needs k-points), and both features under ``--mpi`` (the accelerated periodic path
has no MPI; the MPI k-point solver is a reference-driver path).
"""

from __future__ import annotations

from pathlib import Path
import time


def _mpi_requested() -> bool:
    from ..Parallel import mpi

    return mpi.requested()


def accelerated_band_structure(translation) -> bool:
    """Whether the ``bandstruc`` of ``translation`` runs after the accelerated SCF.

    A scalar periodic cell sampled at Gamma only (the accelerated periodic SCF), not a wire or slab,
    without spin, spin-orbit species, LDA+U or ``--mpi``.  Inputs that fail any of these keep the
    feature on the reference solver; most of them are routed there by another feature as well.
    """

    problem = translation.problem
    if problem.bands is None or problem.periodic_cell is None:
        return False
    scf = problem.scf
    if problem.has_kpoint_sampling or scf.spin_polarized:
        return False
    if scf.self_consistent_spin_orbit or scf.noncollinear:
        return False
    if getattr(problem.grid, "periodicity", 3) in (1, 2):
        return False
    for specification in problem.pseudopotentials.values():
        if specification.spin_orbit or getattr(specification, "has_hubbard", False):
            return False
    return not _mpi_requested()


def accelerated_dos(translation) -> bool:
    """Whether ``Create_DOS`` of ``translation`` runs after the accelerated SCF: clusters, no ``--mpi``.

    The DOS of a periodic cell needs a k-point SCF (Monkhorst-Pack tetrahedra), which runs on the
    reference k-point drivers.
    """

    if getattr(translation, "dos", None) is None:
        return False
    return translation.problem.periodic_cell is None and not _mpi_requested()


def run_post_scf_outputs(system, translation, result, reporter, write, directory: Path) -> None:
    """Bands and DOS of the accelerated run, after the archive and the data files are written.

    ``system`` and ``result`` are the accelerated system and result of the SCF (of the last
    structure of an ionic loop), ``reporter`` the run's reporter (its ``reference`` is the
    :class:`parsec_python.Output.ParsecTextReporter` of the run), ``write`` the log.  Does nothing
    when the input asks for neither output, so the other runs are unchanged.
    """

    bands = accelerated_band_structure(translation)
    dos = accelerated_dos(translation)
    if not (bands or dos):
        return
    from parsec_python.cli import _run_band_structure, _run_dos

    started = time.perf_counter()
    text_reporter = getattr(reporter, "reference", reporter)
    reference_system = getattr(system, "reference", system)
    if bands:
        _run_band_structure(reference_system, translation, result, text_reporter, directory)
    if dos:
        _run_dos(reference_system, translation, result, text_reporter, directory)
    write(f" Post-SCF band structure / DOS wall time [sec] : {time.perf_counter() - started:11.2f}")


__all__ = ["accelerated_band_structure", "accelerated_dos", "run_post_scf_outputs"]
