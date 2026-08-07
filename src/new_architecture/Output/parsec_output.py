"""PARSEC-like text reporting for the Python single-point CLI.

The formatter intentionally omits data Python does not calculate, including
point-group representation numbers, forces, dipoles, and MPI statistics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, TYPE_CHECKING

import numpy as np

from ..models import SCFIteration, SinglePointResult

if TYPE_CHECKING:
    from ..Input.parsec_input import ParsecInputTranslation
    from ..SCF import PreparedSinglePointSystem


# Match the display conversion used by this PARSEC build. Internal Python
# physics remains in Rydberg and does not depend on this reporting constant.
RYDBERG_TO_EV = 13.6058


class ParsecTextReporter:
    """Stateful writer for the subset of ``parsec.out`` Python can support."""

    def __init__(
        self,
        write: Callable[[str], None],
        translation: "ParsecInputTranslation",
    ) -> None:
        self.write = write
        self.translation = translation
        self.problem = translation.problem
        self._previous_total: float | None = None
        self._diagonalization_total = 0.0
        self._hartree_total = 0.0

    def header(self) -> None:
        grid = self.problem.grid
        scf = self.problem.scf
        eigensolver = self.problem.eigensolver
        mixing = self.problem.mixing
        now = datetime.now().astimezone().strftime("%d-%b-%Y %H:%M:%S %z")
        shifted = any(abs(value) > 1.0e-14 for value in grid.shift)
        shape_name = "Spherical" if grid.domain_shape == "sphere" else "Box"
        if grid.domain_shape == "sphere":
            domain_line = f" --- Radius is {grid.radius:10.6f} bohrs"
        else:
            domain_line = (
                " --- Full side lengths are "
                + " ".join(f"{value:10.6f}" for value in grid.box_lengths)
                + " bohrs"
            )

        lines = [
            "",
            " =================================================================",
            "",
            "  PARSEC-PYTHON - Modular real-space DFT program",
            "  PARSEC-like report; Python full-grid implementation",
            "",
            f" starting run on {now}",
            f" input file: {self.translation.source}",
            "",
            " =================================================================",
            "",
            " Initial Run - starting from atomic potentials",
            " ignoresym= T  (Python always uses the full grid)",
            "",
            " Grid data:",
            " ~~------~~",
            "",
            " Confined system (cluster) with zero boundary condition!",
            f"  {shape_name} domain shape:",
            domain_line,
            f" Grid spacing is {grid.spacing:9.6f} bohrs",
            " Order of double grid is    1",
            (
                " Grid points are shifted from origin!"
                if shifted
                else " Grid points include the origin!"
            ),
            (
                " shift vector = "
                + " ".join(f"{value:8.4f}" for value in grid.shift)
                + "   [units of grid spacing]"
            ),
            " WAVEFUNCTIONS ARE REAL!",
            (
                " The Finite-difference expansion is of order "
                f"{grid.expansion_order:2d}"
            ),
            " Python retains the full active grid without symmetry reduction.",
            "",
            " Eigenvalue data:",
            " ----------------",
            f" Number of states: {scf.number_of_states:12d}",
            f" Net cluster charge = {scf.net_charge:8.3f}      [e]",
            f" Fermi temperature = {scf.fermi_temperature_kelvin:9.2f} [K]",
            "",
            " Self-consistency data:",
            " ----------------------",
            f" Maximum number of iterations is {scf.max_iterations:12d}",
            (
                " Performing Chebyshev subspace filtering"
                if eigensolver.method == "chebff"
                else (
                    " Performing Chebyshev-Davidson diagonalization"
                    if eigensolver.method == "chebdav"
                    else " Performing Lanczos/ARPACK diagonalization"
                )
            ),
            (
                " Polynomial degree for First-filter is "
                f"{eigensolver.first_filter_degree:6d}"
                if eigensolver.method == "chebff"
                else (
                    " Polynomial degree for Chebyshev-Davidson is "
                    f"{eigensolver.first_filter_degree:6d}"
                    if eigensolver.method == "chebdav"
                    else ""
                )
            ),
            (
                " The matvec operations use block size "
                f"{eigensolver.matvec_block_size:6d}"
            ),
            (
                " Polynomial degree for Chebyshev filtering is "
                f"{eigensolver.filter_degree:6d}"
            ),
            (
                " Change in polynomial degree (dpm) for Chebyshev filtering is "
                f"{eigensolver.filter_degree_delta:6d}"
            ),
            (
                " Self-consistency convergence criterion is "
                f"{scf.convergence_criterion:25.16E}  Ry"
            ),
            f" Diagonalization tolerance is {eigensolver.tolerance:25.16E}",
            f" Buffer size in subspace is {eigensolver.subspace_buffer:6d}",
            "",
            " Mixer data:",
            " -----------",
            f" solver lpole is : {self.problem.hartree.multipole_order:12d}",
            " Anderson mixer",
            (
                f" Initial Jacobian: {mixing.parameter:6.3f}"
                f"  Mixing memory is {mixing.memory:2d}"
            ),
            f" Mixing restarted after {mixing.restart:12d} iterations",
            "",
            " Correlation data:",
            " -----------------",
            " Exchange-Correlation functional is ca",
            " LDA, Ceperley-Alder, Perdew-Zunger parametrization",
            "",
            " Other input data:",
            " -----------------",
            f" output level [1 - 6] = {self.translation.output_level:2d}",
            " No spin effects!",
            " No minimization!",
            "",
        ]
        self.write("\n".join(lines))

    def setup(self, system: "PreparedSinglePointSystem") -> None:
        lines = [
            " Atom data:",
            " --~--~---",
            "",
            f" Tot. # of atom types is {len(system.pseudopotentials):5d}",
            "",
        ]
        for symbol, potential in system.pseudopotentials.items():
            atoms = [atom for atom in system.atoms if atom.symbol == symbol]
            specification = self.problem.pseudopotentials[symbol]
            lines.extend(
                [
                    f" Chemical element : {symbol}",
                    " martins_new",
                    "  pseudopotential format : new Martins",
                    f"  pseudopotential file   : {potential.source}",
                    (
                        f"  radial points/channels : {potential.radii.size}"
                        f" / {potential.number_of_channels}"
                    ),
                    (
                        f"  selected local channel : "
                        f"l={specification.local_angular_momentum}"
                    ),
                    f" There are {len(atoms):6d} {symbol}  atoms",
                    " and their initial coordinates are:",
                    "",
                    "    x [bohr]          y [bohr]          z [bohr]",
                ]
            )
            for atom in atoms:
                lines.append(
                    f" {atom.position[0]:16.9f}"
                    f" {atom.position[1]:16.9f}"
                    f" {atom.position[2]:16.9f}"
                )
            lines.append("")
        lines.extend(
            [
                f" Tot. number of atoms = {len(system.atoms):7d}",
                "",
                " Real-space setup:",
                " -----------------",
                f" Full active grid points = {system.grid.size:12d}",
                f" Hamiltonian dimension   = {system.grid.size:12d}",
                (
                    " Sparse Laplacian nonzeros = "
                    f"{system.negative_laplacian.nnz:11d}"
                ),
                (
                    " Nonlocal projector columns = "
                    f"{system.nonlocal_operator.projectors.shape[1]:8d}"
                ),
                f" Number of electrons = {system.electron_count:14.8f}",
                (
                    " Initial density integral = "
                    f"{system.grid.integrate(system.initial_density):14.8f}"
                ),
                "",
            ]
        )
        self.write("\n".join(lines))

    def iteration(self, step: SCFIteration) -> None:
        self._diagonalization_total += step.diagonalization_seconds
        self._hartree_total += step.hartree_seconds
        energies = step.energies
        atom_count = len(self.problem.atoms)
        energy_per_atom_ev = (
            energies.total * RYDBERG_TO_EV / max(atom_count, 1)
        )
        lines = [
            f" SCF iter # {step.iteration:3d}",
            "",
            (
                " Diagonalization time [sec] : "
                f"{step.diagonalization_seconds:10.2f},"
                f"      tdiag_sum = {self._diagonalization_total:11.2f}"
            ),
            "",
            f" Fermi level at {step.fermi_level:10.4f} [Ry]",
            "",
            "   State   Eigenvalue [Ry]      Eigenvalue [eV]    Occup.     Repr.",
            "",
        ]
        for index, (eigenvalue, occupation) in enumerate(
            zip(step.eigenvalues, step.occupations),
            start=1,
        ):
            lines.append(
                f"{index:5d}   {eigenvalue:18.10f}"
                f"   {eigenvalue * RYDBERG_TO_EV:18.10f}"
                f" {occupation:9.4f}   {1:6d}"
            )
        lines.extend(
            [
                "",
                (
                    " Max and min values of charge density [e/bohr^3]:"
                    f" {step.density_maximum:12.4E}"
                    f" {step.density_minimum:12.4E}"
                ),
                "",
                f" Hartree potential time [sec]: {step.hartree_seconds:10.2f}",
                "",
                f"   Eigenvalue Energy             = {energies.eigenvalue:20.8f} [Ry]",
                f"   Hartree Energy                = {energies.hartree:20.8f} [Ry]",
                (
                    "   Integral_{Vxc*rho}            = "
                    f"{energies.integral_vxc_rho:20.8f} [Ry]"
                ),
                (
                    "   Exc = Integral{eps_xc*rho}    = "
                    f"{energies.exchange_correlation:20.8f} [Ry]"
                ),
                (
                    "   Electron-Ion energy           = "
                    f"{energies.electron_ion:20.8f} [Ry]"
                ),
                (
                    "   Ion-Ion Energy                = "
                    f"{energies.ion_ion:20.8f} [Ry]"
                ),
            ]
        )
        if self._previous_total is not None:
            delta_ev_per_atom = (
                (energies.total - self._previous_total)
                * RYDBERG_TO_EV
                / max(atom_count, 1)
            )
            lines.append(
                "   (E(new)-E(old))/atom  = "
                f"{delta_ev_per_atom:22.8f} [eV]"
            )
        lines.extend(
            [
                "",
                f"   Total Energy = {energies.total:22.8f} [Ry]",
                f"   Energy/atom  = {energy_per_atom_ev:22.8f} [eV]",
                "",
                (
                    f"  0-{step.iteration:3d}    "
                    "SRE of pot. & charge weighted pot = "
                    f"{step.plain_residual:14.10f}"
                    f" {step.weighted_residual:14.10f}"
                ),
                "",
            ]
        )
        self._previous_total = energies.total
        self.write("\n".join(lines))

    def finish(self, result: SinglePointResult, elapsed_seconds: float) -> None:
        status = (
            "Self-consistency convergence achieved."
            if result.converged
            else "Maximum SCF iterations reached without convergence."
        )
        lines = [
            status,
            "",
            (
                "Time for self-consistent field [sec] : "
                f"{elapsed_seconds:10.2f},"
                f"    tdiag_sum = {self._diagonalization_total:11.2f}"
            ),
            f"Time spent on Hartree potential [sec] : {self._hartree_total:11.2f}",
            "",
            "Forces, dipoles, symmetry representations, and MPI statistics",
            "are not calculated by this Python single-point implementation.",
            "",
            " =================================================================",
        ]
        self.write("\n".join(lines))


__all__ = ["ParsecTextReporter", "RYDBERG_TO_EV"]
