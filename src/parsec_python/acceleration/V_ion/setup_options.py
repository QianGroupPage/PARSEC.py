"""Pseudopotential set-up options the accelerated path takes from the input.

``Double_Grid_Order`` (:mod:`.double_grid`), ``Old_Interpolation_Format``
(:mod:`.native_ionic`, :mod:`.cupy_ionic`, the device local forces) and the
Fourier filters of a periodic system (``Alpha_Filter``, ``Beta1_Filter``,
``Core_Filter``, applied by the readable periodic preparation) change the
static ionic fields.  The double-grid order is part of ``repr`` of the grid
settings, which the prepared-system cache hashes; the per-species options
are hashed here, and nothing is hashed for a species that leaves them at
their defaults, whose key stays what it was.
"""

from __future__ import annotations

from parsec_python.models import SpeciesPotential


def reference_only_setup_features(problem) -> list[str]:
    """The set-up options of ``problem`` the accelerated path leaves to the reference.

    Clusters: the double grid and ``Old_Interpolation_Format`` run on the
    accelerated builders.  A periodic cell at Gamma: the accelerated path
    takes the readable periodic preparation, which builds the double-grid
    projectors and applies the Fourier filters (``SCF/pbc.py``); the parser
    ignores ``Old_Interpolation_Format`` and a cluster's filters.  What the
    accelerated periodic path does not run (forces, k-points, wire and slab,
    the general grid) is routed by its own entries.  Left to the reference:
    the double grid together with spin-orbit species (``SO_PSP``), whose
    spin-orbit projectors and spinor loops were not validated on the
    accelerated path with it.
    """

    specifications = tuple(problem.pseudopotentials.values())
    double_grid = int(getattr(problem.grid, "double_grid_order", 1)) > 1
    if double_grid and any(getattr(item, "spin_orbit", False) for item in specifications):
        return [
            "the double grid with spin-orbit projectors (Double_Grid_Order > 1 with SO_PSP)"
        ]
    return []


def setup_option_details(problem, selection) -> tuple[tuple[str, str], ...]:
    """Backend-report entries of the options a calculation sets; none at the defaults."""

    details: list[tuple[str, str]] = []
    # Stand-in problems of the driver tests carry neither species nor a grid.
    specifications = tuple(getattr(problem, "pseudopotentials", {}).values())
    periodic = getattr(problem, "periodic_cell", None) is not None
    order = int(getattr(getattr(problem, "grid", None), "double_grid_order", 1))
    native = getattr(selection, "finite_difference_builder", None) == "native"
    if order > 1:
        details.append(
            (
                "projector_double_grid",
                f"order {order}: "
                + (
                    "NumPy fine-point sums on the support boxes of the atoms "
                    "(acceleration/V_ion/double_grid.py)"
                    if native and not periodic
                    else "readable builder of the preparation (V_ion/double_grid.py)"
                ),
            )
        )
    if any(getattr(item, "old_interpolation", False) for item in specifications):
        details.append(
            (
                "local_potential_interpolation",
                "V directly (Old_Interpolation_Format) where no cubic spline is set",
            )
        )
    if any(getattr(item, "has_fourier_filter", False) for item in specifications):
        details.append(
            (
                "pseudopotential_fourier_filter",
                "applied by the readable periodic preparation (fourier_f.f90)",
            )
        )
    return tuple(details)


def hash_pseudopotential_setup(digest, specification: SpeciesPotential) -> None:
    """Add the non-default set-up options of one species to ``digest``."""

    if getattr(specification, "old_interpolation", False):
        digest.update(b"old-interpolation")
    if getattr(specification, "has_fourier_filter", False):
        digest.update(
            repr(
                (
                    "fourier-filter",
                    float(specification.fourier_filter_alpha),
                    float(specification.fourier_filter_beta1),
                    specification.fourier_filter_core,
                )
            ).encode("utf-8")
        )


__all__ = [
    "hash_pseudopotential_setup",
    "reference_only_setup_features",
    "setup_option_details",
]
