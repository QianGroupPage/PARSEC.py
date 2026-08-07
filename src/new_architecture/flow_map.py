"""Call-map notes for the PARSEC single-point translation.

The entries here are intentionally small and factual. They are useful when
checking whether a Python stage still corresponds to the Fortran stage it is
trying to translate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowStep:
    """One stage in the constrained single-point PARSEC flow."""

    order: int
    fortran_routine: str
    fortran_file: str
    python_stage: str
    status: str
    notes: str


PARSEC_SINGLE_POINT_FLOW: tuple[FlowStep, ...] = (
    FlowStep(1, "usrinput", "usrinputfile.F90", "SinglePointInput", "translated", "Typed Python configuration with PARSEC defaults for the supported scope."),
    FlowStep(2, "pseudo", "pseudo.f90", "read_parsec_pseudopotential", "translated", "Reads Martins-new POTRE files directly; no elements_new.csv dependency."),
    FlowStep(3, "initial", "initial.f90", "prepare_single_point/center_cluster_geometry", "translated", "Valence-electron count and optional bounding-box midpoint recentering."),
    FlowStep(4, "symmetries", "symmetries.f90", "identity symmetry", "scoped", "Python retains the complete active grid instead of an irreducible wedge."),
    FlowStep(5, "init_var", "init_var.F90", "GridSettings/second_derivative_coefficients", "translated", "Base dimensions and centered Fornberg coefficients."),
    FlowStep(6, "grid_partition/setup", "grid_partition.f90/setup.F90", "build_cluster_grid/build_negative_laplacian", "translated", "Active mask, index map, zero exterior, and sparse neighbor stencil."),
    FlowStep(7, "nonloc", "nonloc.F90", "build_nonlocal_projectors", "translated", "Sparse normalized Kleinman-Bylander projectors through f channels."),
    FlowStep(8, "upot", "upot.F90", "none", "omitted", "DFT+U is outside the requested CA-LDA scope."),
    FlowStep(9, "corecd", "corecd.f90", "superpose_atomic_density(core=True)", "translated", "NLCC contributes to XC but not the Hartree source."),
    FlowStep(10, "ionpot", "ionpot.f90", "build_local_ionic_potential", "translated", "Linear rV interpolation and exact -2Z/r tail."),
    FlowStep(11, "forceion", "forceion.f90", "ion_ion_energy", "translated", "Direct isolated 2*Zi*Zj/R pair energy."),
    FlowStep(12, "initchrg", "initchrg.f90", "superpose_atomic_density", "translated", "File or reconstructed atomic valence density, then normalization."),
    FlowStep(13, "hartset/hpotcg", "hartset.F90/hpotcg.F90", "solve_hartree", "translated", "Multipole/direct Dirichlet boundary and unpreconditioned CG."),
    FlowStep(14, "exc_nspn", "exc_nspn.f90", "ca_lda", "translated", "Unpolarized CA/PZ LDA in Rydberg."),
    FlowStep(15, "eigval/chebff/chebdav/subspace", "eigval.F90/chebff.f90z/chebdav.f90z/subspace.f90z", "Eigensolvers.solve_eigval", "native-port", "First call uses the selected fixed-cycle CHEBFF or locking/restart CHEBDAV algorithm; later calls perform exactly one saved-subspace filter. No ARPACK fallback."),
    FlowStep(16, "flevel/newrho", "flevel.f90/newrho.F90", "fermi_occupations/density_from_orbitals", "translated", "Spin-degenerate finite-temperature occupations and 2/h^3 density."),
    FlowStep(17, "totnrg", "totnrg.f90", "total_energy", "translated", "Input-Hxc/new-density double-counting expression."),
    FlowStep(18, "mixer/getsre", "anderson.f90/getsre.f90", "AndersonMixer/potential_residual_metrics", "translated", "Potential mixing and weighted SRE convergence."),
)


def format_flow_map() -> str:
    """Return a readable single-point call map."""
    lines = []
    for step in PARSEC_SINGLE_POINT_FLOW:
        lines.append(
            f"{step.order:02d}. {step.fortran_routine} "
            f"({step.fortran_file}) -> {step.python_stage} [{step.status}]"
        )
        lines.append(f"    {step.notes}")
    return "\n".join(lines)
