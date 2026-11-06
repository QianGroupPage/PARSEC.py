"""The frozen sample of an FDET tip run (PARSEC 1.5).

PARSEC 1.5 embeds the tip ``B`` (solved by Kohn-Sham) in a frozen sample
``A`` that a separate run wrote to ``pot.dat``
(:mod:`parsec_python.Output.pot_dat`).  On the grid of the tip run
(``parsec.F90:1017-1090``, ``potfield.F90``):

* the sample ionic and Hartree potentials ``v_A,ion + v_H^A`` join the local
  ionic potential of the tip, ``V_ion = v_tip,ion + v_A,ion + v_H^A``; the
  sample ions also enter as point charges of the tip run, in the ion-ion
  energy and forces only (their potential is already ``v_A,ion``:
  ``parsec.F90:1022-1028``);
* the Hartree potential is that of ``rho_B`` alone, the exchange-correlation
  term reads ``rho_B + rho_core + rho_A`` (``hartree.F90:1882-1883``);
* the non-additive kinetic potential ``v_nad`` of
  :mod:`parsec_python.Embedding.kinetic` joins the input and output
  potentials of every iteration (``parsec.F90:1292-1298, 1623-1636``);
* the total energy is the FDET energy of ``A + B`` (``totnrg.f90:159-197``):
  with ``E_KS`` the energy of the tip in ``V_ion`` with the exchange-
  correlation energy of ``rho_B + rho_core + rho_A``,

  ``E = E_KS + T_nad + Integral rho_A v_tip,ion + E_H[rho_A]
  + Integral rho_A v_A,ion + T_s^A``,

  ``T_s^A`` the kinetic energy record ``newk`` of the sample file.

This module maps a sample file onto the tip grid (the classical mode: the tip
run uses the grid of the sample run) and evaluates the FDET terms of one SCF
iteration.  The tip sphere (``Boundary_Sphere_Around_Tip``) is
:mod:`parsec_python.Embedding.tip_sphere`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import EmbeddingEnergyBreakdown, EmbeddingSettings, EnergyBreakdown
from ..Output.pot_dat import PotentialFieldError, PotentialFieldFile, read_potential_field
from .kinetic import KineticFunctional, NonadditiveKinetic, kinetic_functional, nonadditive_kinetic

#: Grid spacings of the sample file and the run that agree (``potfield.F90:240-247``).
SPACING_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class FrozenSampleField:
    """The frozen sample ``A`` on the grid of the tip run.

    ``density`` is ``rho_A`` (e/bohr^3), ``ionic_potential`` ``v_A,ion`` and
    ``hartree_potential`` ``v_H^A`` (Ry), one value per point of the tip grid;
    ``potential`` their sum, the part of the ionic potential of the tip run
    that the sample gives.  ``kinetic_energy`` is ``T_s^A`` (Ry, ``newk`` of
    the file).  ``functional`` is the kinetic functional of ``v_nad``.
    ``source`` is the file the field was taken from; ``tip_sphere`` the state
    of :mod:`parsec_python.Embedding.tip_sphere` (``None`` in the classical
    mode).
    """

    density: np.ndarray
    ionic_potential: np.ndarray
    hartree_potential: np.ndarray
    potential: np.ndarray
    kinetic_energy: float
    functional: KineticFunctional
    source: PotentialFieldFile | None = None
    tip_sphere: object | None = None
    # T[rho_A] of the functional by grid (Ry): it does not change in an SCF loop.
    _sample_energy: dict = field(default_factory=dict, repr=False, compare=False)

    def nonadditive_kinetic(
        self, density: np.ndarray, core_density: np.ndarray | None, grid, operators=None
    ) -> NonadditiveKinetic:
        """``T_nad`` and ``v_nad`` for the tip density ``density`` (``exc.f90:816-822``).

        The total density is ``rho_B + rho_core + rho_A``, the XC input of
        ``hartree.F90:1882-1883``.  ``operators`` evaluate the functional on a
        device (:mod:`parsec_python.acceleration.Embedding`); the potential
        returned is a host array either way.
        """

        total = np.asarray(density, dtype=np.float64)
        if core_density is not None:
            total = total + np.asarray(core_density, dtype=np.float64)
        total = total + self.density
        if operators is not None:
            return operators.run(self._device_nonadditive, total, grid, operators)
        frozen_energy = self._sample_energy.get(id(grid))
        if frozen_energy is None and self.functional.name != "none":
            frozen_energy = kinetic_functional(self.density, grid, self.functional).energy
            self._sample_energy[id(grid)] = frozen_energy
        return nonadditive_kinetic(total, self.density, grid, self.functional, frozen_energy=frozen_energy)

    def _device_nonadditive(self, total, grid, operators) -> NonadditiveKinetic:
        key = ("device", id(grid))
        frozen = self._sample_energy.get(key)
        if frozen is None:
            frozen = (operators.asarray(self.density), None)
            self._sample_energy[key] = frozen
        sample, energy = frozen
        if energy is None and self.functional.name != "none":
            energy = kinetic_functional(sample, grid, self.functional, operators=operators).energy
            self._sample_energy[key] = (sample, energy)
        terms = nonadditive_kinetic(
            operators.asarray(total), sample, grid, self.functional, frozen_energy=energy, operators=operators
        )
        return NonadditiveKinetic(energy=terms.energy, potential=operators.to_host(terms.potential))


def kinetic_functional_of(settings: EmbeddingSettings) -> KineticFunctional:
    """The :class:`KineticFunctional` of an :class:`EmbeddingSettings`."""

    return KineticFunctional(
        name=settings.kinetic_functional,
        form=settings.kinetic_form,
        mu=settings.pbe_mu,
        kappa=settings.pbe_kappa,
    )


def check_cluster_file(data: PotentialFieldFile, path) -> None:
    """Refuse a sample file this port cannot embed a cluster tip in."""

    if not data.is_cluster:
        raise PotentialFieldError(
            0,
            f"{path} is the field of a periodic sample (stype(3) = {data.boundary_code}); "
            "the FDET of this port embeds a cluster tip in a cluster sample only",
        )
    if int(data.system_type[0]) != 1 or int(data.system_type[1]) != 0:
        raise PotentialFieldError(
            0,
            f"{path} is the field of a spin-polarized or complex sample run (stype {list(data.system_type[:2])}); "
            "the FDET of PARSEC 1.5 reads the total density of a real, unpolarized sample",
        )


def map_classical_field(data: PotentialFieldFile, grid, path="pot.dat") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``potfield.F90``: ``(rho_A, v_A,ion, v_H^A)`` of ``data`` on the tip grid.

    The tip run uses the grid of the sample run: the same spacing (601), the
    same number of points (602), indices inside the box of the tip grid
    (603) and every tip point in the file (604).  Python also requires the
    same grid shift, which the 1.5 reader does not check, and a file of the
    full grid (``nwedge = ndim``; a tip run has no symmetry).
    """

    check_cluster_file(data, path)
    steps = np.asarray(data.steps, dtype=np.float64)
    if np.any(np.abs(steps - float(grid.spacing)) > SPACING_TOLERANCE):
        raise PotentialFieldError(
            601,
            f"FDET field grid spacing mismatch: the run has {grid.spacing}, {path} has {steps.tolist()}",
        )
    if data.nwedge != grid.size or data.ndim != grid.size:
        raise PotentialFieldError(
            602,
            f"FDET field grid size mismatch: the run has ndim = nwedge = {grid.size}, "
            f"{path} has ndim {data.ndim}, nwedge {data.nwedge} (the sample run needs the radius, spacing "
            "and grid shift of the tip run and Ignore_Symmetry)",
        )
    points = np.asarray(data.points, dtype=np.int64)
    if np.any(points.min(axis=0) < grid.index_min) or np.any(points.max(axis=0) > grid.index_max):
        raise PotentialFieldError(603, f"FDET field of {path} extends outside the current grid")
    if np.any(np.abs(np.asarray(data.shift, dtype=np.float64) - np.asarray(grid.settings.shift)) > 1.0e-12):
        raise PotentialFieldError(
            0,
            f"FDET field grid shift mismatch: the run has {list(grid.settings.shift)}, {path} has "
            f"{np.asarray(data.shift).tolist()} (the grids must be the same; set Ignore_Symmetry alike in both runs)",
        )
    rows = grid.rows_for_integer_coordinates(points)
    found = np.zeros(grid.size, dtype=bool)
    valid = rows >= 0
    found[rows[valid]] = True
    if not np.all(found) or int(np.count_nonzero(valid)) != grid.size:
        missing = np.flatnonzero(~found)
        where = grid.integer_coordinates[missing[0]].tolist() if missing.size else "duplicated"
        raise PotentialFieldError(604, f"FDET field grid point not found in {path}: {where}")
    density = np.empty(grid.size, dtype=np.float64)
    ionic = np.empty(grid.size, dtype=np.float64)
    hartree = np.empty(grid.size, dtype=np.float64)
    density[rows] = data.density
    ionic[rows] = data.ionic_potential
    hartree[rows] = data.hartree_potential
    return density, ionic, hartree


def frozen_sample_field(settings: EmbeddingSettings, grid) -> FrozenSampleField:
    """The classical frozen field of ``settings`` on ``grid`` (the sample file is read here)."""

    data = read_potential_field(settings.field_path)
    density, ionic, hartree = map_classical_field(data, grid, settings.field_path)
    return FrozenSampleField(
        density=density,
        ionic_potential=ionic,
        hartree_potential=hartree,
        # vion + vsion + vshart of parsec.F90:1087-1089, added to the ionic potential in this order
        potential=ionic + hartree,
        kinetic_energy=float(data.kinetic_energy),
        functional=kinetic_functional_of(settings),
        source=data,
    )


def with_sample_potential(ionic_potential: np.ndarray, field: FrozenSampleField) -> np.ndarray:
    """``vion = vion + vsion + vshart`` (``parsec.F90:1087-1089``)."""

    return ionic_potential + field.ionic_potential + field.hartree_potential


def embedding_energies(
    energies: EnergyBreakdown,
    field: FrozenSampleField,
    density: np.ndarray,
    ionic_potential: np.ndarray,
    nonadditive: NonadditiveKinetic,
    volume_element: float,
) -> EmbeddingEnergyBreakdown:
    """The FDET energy of ``totnrg.f90:159-197`` from the plain breakdown.

    ``energies`` is :func:`parsec_python.Energy.total_energy` of the
    iteration with the ionic potential of the tip run (sample field
    included) and the XC terms of ``rho_B + rho_core + rho_A``; its
    electronic energy is ``eele - enadd`` of the Fortran, because the input
    potential holds ``v_nad``.  ``density`` is ``rho_B``,
    ``ionic_potential`` the ionic potential of the tip run.
    """

    sample = field.density
    tip_ionic = np.asarray(ionic_potential, dtype=np.float64) - field.ionic_potential - field.hartree_potential
    tip_ion_sample_electron = float(volume_element * np.dot(tip_ionic, sample))
    sample_electron_ion = float(volume_element * np.dot(field.ionic_potential, sample))
    sample_hartree = 0.5 * float(volume_element * np.dot(field.hartree_potential, sample))
    integral_vnad = float(volume_element * np.dot(nonadditive.potential, np.asarray(density, dtype=np.float64)))
    additions = (
        tip_ion_sample_electron
        + sample_hartree
        + sample_electron_ion
        + float(nonadditive.energy)
        + float(field.kinetic_energy)
    )
    tip_sphere = field.tip_sphere
    sample_xc = 0.0
    outside = 0.0
    if tip_sphere is not None:
        # totnrg.f90:195-196: the sample-only constants leave the total.
        sample_xc = float(tip_sphere.sample_xc_energy)
        outside = float(tip_sphere.outside_energy)
        additions += -sample_hartree - sample_electron_ion - float(field.kinetic_energy) - sample_xc + outside
    electronic = float(energies.electronic + additions)
    return EmbeddingEnergyBreakdown(
        eigenvalue=energies.eigenvalue,
        hartree=energies.hartree,
        integral_vxc_rho=energies.integral_vxc_rho,
        exchange_correlation=energies.exchange_correlation,
        electron_ion=energies.electron_ion,
        ion_ion=energies.ion_ion,
        electronic=electronic,
        total=electronic + float(energies.ion_ion),
        nonadditive_kinetic=float(nonadditive.energy),
        integral_vnad_rho=integral_vnad,
        tip_ion_sample_electron=tip_ion_sample_electron,
        sample_hartree=sample_hartree,
        sample_electron_ion=sample_electron_ion,
        sample_kinetic=float(field.kinetic_energy),
        tip_sphere=tip_sphere is not None,
        sample_xc_on_tip_grid=sample_xc,
        tip_ion_sample_electron_outside=outside,
    )


def embedded_exchange_correlation(system, density: np.ndarray):
    """XC of ``rho_B + rho_core + rho_A`` and the non-additive kinetic terms.

    Returns ``(xc, nonadditive)``: ``system.evaluate_xc`` of
    ``rho_B + rho_A`` (the system adds its core density) and
    :meth:`FrozenSampleField.nonadditive_kinetic`; ``nonadditive`` is
    ``None`` for a system without FDET, whose XC is that of ``density``.
    """

    field = getattr(system, "embedding", None)
    if field is None:
        return system.evaluate_xc(density), None
    xc = system.evaluate_xc(np.asarray(density, dtype=np.float64) + field.density)
    # An accelerated system evaluates the kinetic terms on its device (kinetic_operators).
    return xc, field.nonadditive_kinetic(
        density,
        getattr(system, "core_density", None),
        system.grid,
        operators=getattr(system, "kinetic_operators", None),
    )


__all__ = [
    "FrozenSampleField",
    "check_cluster_file",
    "embedded_exchange_correlation",
    "embedding_energies",
    "frozen_sample_field",
    "kinetic_functional_of",
    "map_classical_field",
    "with_sample_potential",
]
