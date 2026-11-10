"""The periodic local potential of tables whose Coulomb tail is exact only to the digits of their file.

The fast preparation (:mod:`parsec_python.acceleration.V_ion.periodic_setup_fortran`, installed by ``import
parsec_python``) sums ``V_a + 2 Z erf(eta r)/r`` over the images of every atom out to the radius where the table of
``r V`` is ``-2 Z`` (:func:`parsec_python.acceleration.V_ion.periodic_device.coulomb_radius`).  A FHIPP table is
``-2 Z`` only to its digits (1e-13 to 1e-11 relative) out to its end; up to 2026-11-07 that radius was then the end
of the table, and with ``Cubic_Spline`` the images added the error of the spline of ``V`` near the end of the table:
the local potential of an orthorhombic silicon cell 0.05 Ry low, its total energy 1.6 Ry low, while fcc cells (the
readable sum) were right.  These tests hold the noisy table against its noise-free twin and the FHIPP file against
the POTRE file of the same library potential, on orthorhombic and fcc cells, at Gamma and with k-points.
"""

from __future__ import annotations

import dataclasses
import functools
import unittest
import warnings
from pathlib import Path

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.acceleration.V_ion import periodic_device
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.Pseudopotential.fhipp import read_fhipp_pseudopotential
from parsec_python.SCF import run_scf
from parsec_python.SCF.kpoints import run_scf_kpoints
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.V_ion.ewald import ewald_local_ionic_potential
from parsec_python.V_ion.ionic_potential import coulomb_onset_radius, coulomb_tail_radius

_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_ROOT = Path(__file__).resolve().parents[4]
# The PBE hydrogen of the library in its two formats (src/pp_generation/NCPPs/PBE/H/FHI and PARSEC).
_H_FHIPP = _DATA / "H_PBE_FHIPP.DAT"
_H_POTRE = _ROOT / "src" / "pp_generation" / "NCPPs" / "PBE" / "H" / "PARSEC" / "H_POTRE.DAT"
_SI_FHIPP = _DATA / "Si_FHIPP.DAT"
_POTRE_TABLES = (
    _ROOT / "examples" / "0d_AuH" / "Au_POTRE.DAT",
    _ROOT / "examples" / "0d_AuH" / "H_POTRE.DAT",
    _DATA / "Si_POTRE.DAT",
    _H_POTRE,
)
_CUBE = 6.0 * np.eye(3)
_FCC = 4.0 * (np.ones((3, 3)) - np.eye(3))  # oblique: the general grid of the Fortran-derived routines
_FRACTIONAL = np.array([[0.31, 0.27, 0.40], [0.52, 0.47, 0.44]])  # an H2 molecule, 1.4 bohr in the cube


def _silicon_fhipp():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_fhipp_pseudopotential(
            _SI_FHIPP, symbol="Si", number_of_channels=3, electrons_per_orbital=(2.0, 2.0, 0.0), core_cutoff_radius=2.1
        )


def _strict_radius(potential, local_l: int) -> float:
    """The rule of ``coulomb_radius`` up to 2026-11-07 (``|r V + 2 Z| <= 1e-15 x 2 Z``, else the table end)."""
    radii = np.asarray(potential.radii)
    deviation = np.abs(radii * np.asarray(potential.channel_potentials[local_l]) + 2.0 * potential.ionic_charge)
    deviating = np.flatnonzero(deviation > 1.0e-15 * max(2.0 * potential.ionic_charge, 1.0))
    if deviating.size == 0:
        return float(radii[0])
    last = int(deviating[-1]) + 1
    cutoff = float(potential.interpolation_cutoff)
    return cutoff if last >= radii.size else min(cutoff, float(radii[last]))


def _noise_free(potential, local_l: int):
    """The table with ``r V = -2 Z`` exactly from its Coulomb tail on (a POTRE file of the same potential)."""
    radii = np.asarray(potential.radii)
    start = int(np.searchsorted(radii, coulomb_tail_radius(potential, local_l)))
    values = np.array(potential.channel_potentials[local_l], dtype=float)
    values[start:] = -2.0 * potential.ionic_charge / radii[start:]
    channels = dict(potential.channel_potentials)
    channels[local_l] = values
    return dataclasses.replace(potential, channel_potentials=channels)


class CoulombTailRadiusTests(unittest.TestCase):
    def test_a_fhipp_table_settles_where_its_digits_end(self) -> None:
        potential = _silicon_fhipp()
        for local_l in sorted(potential.channel_potentials):
            with self.subTest(l=local_l):
                # the rule to round-off never settles: the end of the table
                self.assertEqual(_strict_radius(potential, local_l), potential.interpolation_cutoff)
                tail = coulomb_tail_radius(potential, local_l)
                self.assertLess(tail, 10.0)
                self.assertEqual(periodic_device.coulomb_radius(potential, local_l), tail)
                self.assertEqual(coulomb_onset_radius(potential, local_l), tail)
                radii = np.asarray(potential.radii)
                beyond = radii >= tail
                deviation = np.abs(radii * potential.channel_potentials[local_l] + 8.0)[beyond]
                self.assertLess(float(np.max(deviation)), 8.0e-10)

    def test_potre_tables_keep_the_rule_to_round_off(self) -> None:
        for path in _POTRE_TABLES:
            potential = read_parsec_pseudopotential(path)
            for local_l in sorted(potential.channel_potentials):
                with self.subTest(table=path.name, l=local_l):
                    strict = _strict_radius(potential, local_l)
                    self.assertLess(strict, potential.interpolation_cutoff)
                    self.assertEqual(periodic_device.coulomb_radius(potential, local_l), strict)

    def test_a_table_without_coulomb_tail_keeps_the_end_of_the_table(self) -> None:
        potential = _silicon_fhipp()
        radii = np.asarray(potential.radii)
        values = np.array(potential.channel_potentials[2]) + 1.0e-5 / radii  # r V + 2 Z = 1e-5 everywhere outside
        broken = dataclasses.replace(potential, channel_potentials={**potential.channel_potentials, 2: values})
        self.assertIsNone(coulomb_tail_radius(broken, 2))
        self.assertEqual(periodic_device.coulomb_radius(broken, 2), broken.interpolation_cutoff)
        self.assertEqual(coulomb_onset_radius(broken, 2), broken.interpolation_cutoff)
        # with Cubic_Spline the periodic sum to the end of the table is announced (the spline error enters it)
        from parsec_python.acceleration.V_ion.periodic_setup_fortran import _local_reach

        for spline in (True, False):
            specification = SpeciesPotential(
                _SI_FHIPP, 2, use_spline=spline, pseudopotential_format="fhipp", number_of_channels=3,
                electrons_per_orbital=(2.0, 2.0, 0.0), core_cutoff_radius=2.1,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reach = _local_reach(broken, specification, 0.5, 10.0, "physical")
            self.assertEqual(reach, broken.interpolation_cutoff)
            self.assertEqual(any("Cubic_Spline" in str(item.message) for item in caught), spline)


class NoisyTableLocalPotentialTests(unittest.TestCase):
    """The local potential of the FHIPP table equals that of its noise-free twin to the noise of the file (5e-10 Ry
    in the balls of the fast preparation, 1e-9 Ry over the reach of the readable sum; the spline tail of the old
    rule lowered the potential of a silicon crystal by 0.05 Ry), with the cubic spline and with the linear
    interpolation, on the orthorhombic grid (the fast preparation) and on the general grid (by default the split
    with the cubic spline, the readable sum without) of the Fortran-derived routines."""

    def _potentials(self, lattice, spline: bool):
        noisy = _silicon_fhipp()
        clean = _noise_free(noisy, 2)
        grid = build_periodic_grid(
            PeriodicCell(lattice_vectors=lattice),
            PeriodicGridSettings(spacing=0.5, expansion_order=6, fortran_routines=True),
        )
        atoms = [Atom("Si", (fraction @ lattice).tolist()) for fraction in _FRACTIONAL]
        specification = {
            "Si": SpeciesPotential(
                _SI_FHIPP, 2, use_spline=spline, pseudopotential_format="fhipp", number_of_channels=3,
                electrons_per_orbital=(2.0, 2.0, 0.0), core_cutoff_radius=2.1,
            )
        }
        return [
            ewald_local_ionic_potential(grid, atoms, {"Si": potential}, specification, lattice)
            for potential in (noisy, clean)
        ]

    def test_orthorhombic_and_fcc_cells(self) -> None:
        for name, lattice in (("cube", _CUBE), ("fcc", _FCC)):
            for spline in (True, False):
                with self.subTest(cell=name, spline=spline):
                    noisy, clean = self._potentials(lattice, spline)
                    self.assertLess(float(np.max(np.abs(noisy - clean))), 2.0e-9)


def _force_case():
    """Two Si atoms (Cubic_Spline, a Gaussian core density as nonlinear core correction) in the cube, the fields of the
    local forces: the superposed density, a residual and an exchange-correlation potential made from it."""

    potential = _silicon_fhipp()
    radii = np.asarray(potential.radii)
    potential = dataclasses.replace(potential, core_marker="pcec", core_density=0.3 * np.exp(-((radii / 0.9) ** 2)))
    specification = SpeciesPotential(
        _SI_FHIPP, 2, use_spline=True, pseudopotential_format="fhipp", number_of_channels=3,
        electrons_per_orbital=(2.0, 2.0, 0.0), core_cutoff_radius=2.1,
    )
    grid = build_periodic_grid(
        PeriodicCell(lattice_vectors=_CUBE), PeriodicGridSettings(spacing=0.5, expansion_order=6, fortran_routines=True)
    )
    atoms = tuple(Atom("Si", (fraction @ _CUBE).tolist()) for fraction in _FRACTIONAL)
    potentials, specifications = {"Si": potential}, {"Si": specification}
    from parsec_python.V_ion import superpose_atomic_density

    density = superpose_atomic_density(grid, atoms, potentials, specifications, _CUBE)
    residual = 1.0e-3 * np.sin(grid.coordinates[:, 0]) * density
    xc = -0.9 * np.cbrt(density + superpose_atomic_density(grid, atoms, potentials, specifications, _CUBE, core=True))
    return grid, atoms, potentials, specifications, density, residual, xc


class SplineForceTests(unittest.TestCase):
    """The local forces of Cubic_Spline species on the orthorhombic grid of the Fortran-derived routines
    (:mod:`parsec_python.acceleration.Forces.periodic_spline`): the derivative of the potential of the fast
    preparation, and the density terms of the readable sum."""

    def setUp(self) -> None:
        from parsec_python.Forces import periodic as force_sums

        self.force_sums = force_sums
        self.case = _force_case()

    def _forces(self, provider):
        grid, atoms, potentials, specifications, density, residual, xc = self.case
        self.addCleanup(self.force_sums.set_local_force_provider, self.force_sums._local_force_provider)
        self.force_sums.set_local_force_provider(provider)
        return self.force_sums.periodic_local_forces(
            grid, atoms, potentials, specifications, _CUBE, density, residual, xc
        )

    def test_the_provider_is_installed(self) -> None:
        from parsec_python.acceleration.Forces.periodic_spline import spline_periodic_local_forces

        self.assertIs(self.force_sums._local_force_provider, spline_periodic_local_forces)

    def test_the_ionic_term_is_the_derivative_of_the_fast_potential(self) -> None:
        from parsec_python.acceleration.Forces.periodic_spline import spline_periodic_local_forces

        grid, atoms, potentials, specifications, density, _, _ = self.case
        forces = self._forces(spline_periodic_local_forces).ionic
        step = 2.0e-4
        for axis in range(3):
            energies = []
            for shift in (step, -step, 2.0 * step, -2.0 * step):
                moved = list(atoms)
                position = np.array(atoms[1].position, dtype=float)
                position[axis] += shift
                moved[1] = Atom("Si", position.tolist())
                potential = ewald_local_ionic_potential(grid, tuple(moved), potentials, specifications, _CUBE)
                energies.append(grid.volume_element * float(density @ potential))
            derivative = (8.0 * (energies[0] - energies[1]) - (energies[2] - energies[3])) / (12.0 * step)
            with self.subTest(axis=axis):
                # the fast potential jumps by the spline error of -2Z/r (1e-7 Ry) where a point leaves a ball
                self.assertAlmostEqual(forces[1, axis], -derivative, delta=2.0e-6)

    def test_the_density_terms_are_those_of_the_readable_sum(self) -> None:
        from parsec_python.acceleration.Forces.periodic_spline import spline_periodic_local_forces

        fast = self._forces(spline_periodic_local_forces)
        dense = self._forces(None)
        for term in ("self_consistency", "core_correction"):
            with self.subTest(term=term):
                expected = getattr(dense, term)
                self.assertGreater(float(np.max(np.abs(expected))), 1.0e-6)
                # the same terms (the device kernel adds them in another order: 1e-13 relative)
                scale = float(np.max(np.abs(expected)))
                np.testing.assert_allclose(getattr(fast, term), expected, rtol=0.0, atol=1.0e-11 * scale)
        # the readable sum adds the spline error of -2Z/r out to erfcinv(1e-26)/eta (26 bohr here): 2.6e-7 Ry/bohr
        np.testing.assert_allclose(fast.ionic, dense.ionic, rtol=0.0, atol=2.0e-6)

    def test_host_and_device_density_terms_agree(self) -> None:
        import os
        from unittest import mock

        from parsec_python.acceleration.Forces.periodic_spline import spline_periodic_local_forces
        from parsec_python.acceleration.V_ion import periodic_device as fields

        if fields._cupy() is None:
            self.skipTest("no usable CUDA device")
        device = self._forces(spline_periodic_local_forces)
        with mock.patch.dict(os.environ, {"PARSEC_DEVICE": "cpu"}):
            host = self._forces(spline_periodic_local_forces)
        for term in ("ionic", "self_consistency", "core_correction"):
            with self.subTest(term=term):
                reference = getattr(host, term)
                scale = max(float(np.max(np.abs(reference))), 1.0e-30)
                np.testing.assert_allclose(getattr(device, term), reference, rtol=0.0, atol=1.0e-12 * scale)


def _problem(path: Path, lattice: np.ndarray, *, fhipp: bool) -> SinglePointInput:
    if fhipp:
        species = SpeciesPotential(
            path, 0, use_spline=True, pseudopotential_format="fhipp", number_of_channels=1,
            electrons_per_orbital=(1.0,), core_cutoff_radius=1.0,
        )
    else:
        species = SpeciesPotential(path, 0, use_spline=True)
    return SinglePointInput(
        atoms=[Atom("H", (fraction @ lattice).tolist()) for fraction in _FRACTIONAL],
        pseudopotentials={"H": species},
        grid=PeriodicGridSettings(spacing=0.4, expansion_order=6, fortran_routines=True),
        periodic_cell=PeriodicCell(lattice_vectors=lattice),
        scf=SCFSettings(max_iterations=60, number_of_states=2, convergence_criterion=1.0e-8, xc_functional="pbe"),
        hartree=HartreeSettings(relative_tolerance=1e-10, absolute_tolerance=1e-13, max_iterations=4000),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-9),
    )


@functools.lru_cache(maxsize=None)
def _energy(fhipp: bool, cell: str, kpoints: bool) -> float:
    lattice = _CUBE if cell == "cube" else _FCC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = prepare_periodic_single_point(_problem(_H_FHIPP if fhipp else _H_POTRE, lattice, fhipp=fhipp))
        if kpoints:
            result = run_scf_kpoints(system, np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]), np.array([0.5, 0.5]))
        else:
            result = run_scf(system)
    return float(result.energies.total)


class FhippPotreEnergyTests(unittest.TestCase):
    """H2 with the FHIPP and the POTRE file of the library's PBE hydrogen (Cubic_Spline, Fortran-derived routines):
    the same total energy on an orthorhombic and an fcc cell, at Gamma and with k-points.  The FHIPP file's
    orthorhombic energies were 0.032 Ry low before 2026-11-07; the two files now differ by 2e-7 Ry (cube) and
    3e-7 Ry (fcc; 1.5e-6 Ry on the readable sum of the fcc cell, which adds the spline error of each radial grid
    out to its erfc reach)."""

    def test_the_two_files_give_the_same_energy(self) -> None:
        for cell in ("cube", "fcc"):
            for kpoints in (False, True):
                with self.subTest(cell=cell, kpoints=kpoints):
                    difference = _energy(True, cell, kpoints) - _energy(False, cell, kpoints)
                    self.assertLess(abs(difference), 1.0e-5)


class ObliqueSplitTests(unittest.TestCase):
    """``PARSEC_PERIODIC_OBLIQUE_SPLIT``: the split of the orthorhombic grid on the general grid of the Fortran-derived
    routines (``fast``; by default, ``auto``, where a species uses ``Cubic_Spline``) or the readable sum.  A linearly
    interpolated species gets the readable potential to round-off; a ``Cubic_Spline`` one loses the spline error of
    ``-2 Z/r`` that the readable sum adds out to its erfc reach; the local forces are the derivative of that
    potential."""

    def _potential(self, split: str, spline: bool):
        import os
        from unittest import mock

        clean = _noise_free(_silicon_fhipp(), 2)
        grid = build_periodic_grid(
            PeriodicCell(lattice_vectors=_FCC),
            PeriodicGridSettings(spacing=0.5, expansion_order=6, fortran_routines=True),
        )
        atoms = [Atom("Si", (fraction @ _FCC).tolist()) for fraction in _FRACTIONAL]
        specification = {
            "Si": SpeciesPotential(
                _SI_FHIPP, 2, use_spline=spline, pseudopotential_format="fhipp", number_of_channels=3,
                electrons_per_orbital=(2.0, 2.0, 0.0), core_cutoff_radius=2.1,
            )
        }
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_OBLIQUE_SPLIT": split}):
            return grid, atoms, specification, ewald_local_ionic_potential(
                grid, atoms, {"Si": clean}, specification, _FCC
            )

    def test_linear_species_get_the_readable_potential(self) -> None:
        readable = self._potential("readable", False)[3]
        fast = self._potential("fast", False)[3]
        self.assertLess(float(np.max(np.abs(fast - readable))), 1.0e-11)

    def test_the_default_takes_the_split_for_spline_species_only(self) -> None:
        for spline, route in ((True, "fast"), (False, "readable")):
            with self.subTest(spline=spline):
                np.testing.assert_array_equal(self._potential("auto", spline)[3], self._potential(route, spline)[3])

    def test_spline_species_lose_the_spline_tail(self) -> None:
        difference = self._potential("fast", True)[3] - self._potential("readable", True)[3]
        self.assertLess(float(np.max(np.abs(difference))), 1.0e-4)
        self.assertGreater(abs(float(np.mean(difference))), 1.0e-7)  # nearly a constant: the tail of every image

    def test_forces_are_the_derivative_of_the_fast_potential(self) -> None:
        import os
        from unittest import mock

        from parsec_python.Forces import periodic as force_sums
        from parsec_python.V_ion import superpose_atomic_density

        grid, atoms, specification, _ = self._potential("readable", True)
        potentials = {"Si": _noise_free(_silicon_fhipp(), 2)}
        # a distorted cell: the second atom off its site
        atoms = [atoms[0], Atom("Si", (np.asarray(atoms[1].position) + np.array([0.13, -0.07, 0.05])).tolist())]
        density = superpose_atomic_density(grid, atoms, potentials, specification, _FCC)
        residual = 1.0e-3 * np.sin(grid.coordinates[:, 0]) * density
        arguments = (grid, atoms, potentials, specification, _FCC, density, residual, np.zeros(grid.size))
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_OBLIQUE_SPLIT": "fast"}):
            fast = force_sums.periodic_local_forces(*arguments)
            step = 2.0e-4
            for axis in range(3):
                energies = []
                for shift in (step, -step, 2.0 * step, -2.0 * step):
                    position = np.array(atoms[1].position, dtype=float)
                    position[axis] += shift
                    moved = [atoms[0], Atom("Si", position.tolist())]
                    potential = ewald_local_ionic_potential(grid, moved, potentials, specification, _FCC)
                    energies.append(grid.volume_element * float(density @ potential))
                derivative = (8.0 * (energies[0] - energies[1]) - (energies[2] - energies[3])) / (12.0 * step)
                with self.subTest(axis=axis):
                    self.assertAlmostEqual(fast.ionic[1, axis], -derivative, delta=2.0e-6)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_OBLIQUE_SPLIT": "readable"}):
            self.addCleanup(force_sums.set_local_force_provider, force_sums._local_force_provider)
            force_sums.set_local_force_provider(None)
            dense = force_sums.periodic_local_forces(*arguments)
        # the same density term; the ionic terms differ by the spline error the readable sum adds (small cell)
        scale = float(np.max(np.abs(dense.self_consistency)))
        np.testing.assert_allclose(fast.self_consistency, dense.self_consistency, rtol=0.0, atol=1.0e-11 * scale)
        np.testing.assert_allclose(fast.ionic, dense.ionic, rtol=0.0, atol=2.0e-5)

    def test_the_switch_is_checked(self) -> None:
        import os
        from unittest import mock

        from parsec_python.acceleration.V_ion import periodic_setup_fortran

        self.assertEqual(periodic_setup_fortran.oblique_split(), "auto")
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_OBLIQUE_SPLIT": "fastest"}):
            with self.assertRaises(ValueError):
                periodic_setup_fortran.oblique_split()


class ObliqueDeviceTests(unittest.TestCase):
    """The device sums of an oblique cell (``acceleration/V_ion/periodic_device.py``, lattice kernels): the
    superposed valence and linear core densities and the density terms of the local forces equal the host sums to
    round-off."""

    def setUp(self) -> None:
        from parsec_python.acceleration.V_ion import periodic_device as fields

        if fields._cupy() is None:
            self.skipTest("no usable CUDA device")
        potential = _silicon_fhipp()
        radii = np.asarray(potential.radii)
        potential = dataclasses.replace(potential, core_marker="pcec", core_density=0.3 * np.exp(-((radii / 0.9) ** 2)))
        self.potentials = {"Si": potential}
        self.specification = {
            "Si": SpeciesPotential(
                _SI_FHIPP, 2, use_spline=True, pseudopotential_format="fhipp", number_of_channels=3,
                electrons_per_orbital=(2.0, 2.0, 0.0), core_cutoff_radius=2.1,
            )
        }
        self.grid = build_periodic_grid(
            PeriodicCell(lattice_vectors=_FCC),
            PeriodicGridSettings(spacing=0.5, expansion_order=6, fortran_routines=True),
        )
        self.atoms = [
            Atom("Si", (fraction @ _FCC).tolist()) for fraction in _FRACTIONAL
        ]

    def _both(self, compute):
        import os
        from unittest import mock

        device = compute()
        with mock.patch.dict(os.environ, {"PARSEC_DEVICE": "cpu"}):
            host = compute()
        return device, host

    def test_densities(self) -> None:
        from parsec_python.V_ion import superpose_atomic_density

        linear_core = {"Si": dataclasses.replace(self.specification["Si"], use_spline=False)}
        for core, specification in ((False, self.specification), (True, linear_core)):
            with self.subTest(core=core):
                device, host = self._both(
                    lambda: superpose_atomic_density(
                        self.grid, self.atoms, self.potentials, specification, _FCC, core=core
                    )
                )
                scale = float(np.max(np.abs(host)))
                self.assertGreater(scale, 0.0)
                np.testing.assert_allclose(device, host, rtol=0.0, atol=1.0e-14 * scale)

    def test_density_terms_of_the_forces(self) -> None:
        from parsec_python.Forces import periodic as force_sums
        from parsec_python.V_ion import superpose_atomic_density

        linear_core = {"Si": dataclasses.replace(self.specification["Si"], use_spline=False)}
        density = superpose_atomic_density(self.grid, self.atoms, self.potentials, self.specification, _FCC)
        residual = 1.0e-3 * np.sin(self.grid.coordinates[:, 0]) * density
        xc = -0.9 * np.cbrt(density)
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_OBLIQUE_SPLIT": "fast"}):
            for specification in (self.specification, linear_core):
                arguments = (self.grid, self.atoms, self.potentials, specification, _FCC, density, residual, xc)
                device, host = self._both(lambda: force_sums.periodic_local_forces(*arguments))
                for term in ("ionic", "self_consistency", "core_correction"):
                    with self.subTest(spline=specification is self.specification, term=term):
                        reference = getattr(host, term)
                        scale = float(np.max(np.abs(reference)))
                        self.assertGreater(scale, 0.0)
                        np.testing.assert_allclose(getattr(device, term), reference, rtol=0.0, atol=1.0e-12 * scale)


if __name__ == "__main__":
    unittest.main()
