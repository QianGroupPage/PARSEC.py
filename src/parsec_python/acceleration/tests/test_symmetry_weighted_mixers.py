"""The Broyden and multisecant mixers on a symmetry wedge.

``SymmetrySCFReducer.mixer`` keeps :class:`SymmetryAndersonMixer` for
Anderson and runs the other mixers of :mod:`parsec_python.Mixer` on one value
per orbit with the orbit multiplicities as inner-product weights.  The mixed
fields must be those of the full-grid mixers, step by step and along a whole
SCF loop.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.Mixer import make_mixer
from parsec_python.SCF.single_point import prepare_single_point, run_scf
from parsec_python.acceleration.SCF import (
    SymmetryAndersonMixer,
    SymmetrySCFReducer,
    SymmetryScalarField,
    SymmetryWeightedMixer,
)
from parsec_python.acceleration.Symmetry import AxisReflectionReduction
from parsec_python.models import MixingSettings

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

_METHODS = (
    {"method": "broyden"},
    {"method": "broyden", "memory": 2, "restart": 5},
    {"method": "msecant1"},
    {"method": "msecant2", "group_size": 0},
    {"method": "msecant3", "group_size": 2},
    {"method": "msecant1", "en_like": 1},
)


def _reducer(size: int, seed: int = 5) -> SymmetrySCFReducer:
    """A reducer on ``size`` orbits of the sizes of a group of order eight."""

    rng = np.random.default_rng(seed)
    multiplicities = rng.choice(np.array([1, 2, 4, 8]), size=size)
    return SymmetrySCFReducer(
        SimpleNamespace(
            multiplicities=multiplicities,
            full_to_wedge=np.repeat(np.arange(size), multiplicities),
            wedge_size=size,
            full_size=int(multiplicities.sum()),
        )
    )


class WeightedWedgeMixerTests(unittest.TestCase):
    def test_anderson_keeps_its_wedge_mixer(self) -> None:
        reducer = _reducer(9)
        self.assertIs(type(reducer.mixer(MixingSettings())), SymmetryAndersonMixer)
        for options in _METHODS:
            mixer = reducer.mixer(MixingSettings(**options))
            self.assertIs(type(mixer), SymmetryWeightedMixer)
            np.testing.assert_array_equal(mixer.mixer.weights, reducer.weights)

    def test_each_step_is_the_full_grid_step(self) -> None:
        size = 15
        reducer = _reducer(size, seed=7)
        expand = reducer.reduction.full_to_wedge
        rng = np.random.default_rng(8)
        matrix = 0.4 * rng.standard_normal((size, size)) / np.sqrt(size)
        offset = 0.3 * rng.standard_normal(size)
        for options in _METHODS:
            settings = MixingSettings(**options)
            wedge = reducer.mixer(settings)
            full = make_mixer(settings)
            x = reducer.field(np.zeros(size))
            for iteration in range(1, 12):
                y = reducer.field(np.tanh(matrix @ x.values) + offset)
                full_mixed = full.mix(
                    reducer.to_full(x), reducer.to_full(y), iteration=iteration
                )
                mixed = wedge.mix(x, y, iteration=iteration)
                self.assertIsInstance(mixed, SymmetryScalarField)
                np.testing.assert_allclose(
                    reducer.to_full(mixed),
                    full_mixed,
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"{options} iteration {iteration}",
                )
                x = mixed
            # A full-grid input gives a full-grid result.
            again = reducer.mixer(settings)
            result = again.mix(reducer.to_full(x), reducer.to_full(y), iteration=1)
            self.assertEqual(result.shape, (reducer.reduction.full_size,))


class WeightedWedgeSCFTests(unittest.TestCase):
    """A whole SCF loop of H2 on the half-shifted grid of its D2h wedge."""

    @classmethod
    def setUpClass(cls) -> None:
        text = (
            "Boundary_Sphere_Radius: 5.0 bohr\nGrid_Spacing: 0.6 bohr\n"
            "Coordinate_Unit: Cartesian_Bohr\nExpansion_Order: 4\nStates_Num: 2\n"
            "Max_Iter: 30\nConvergence_Criterion: 1e-7\nEigensolver: chebff\n"
            "Chebdav_Degree: 30\nFF_MaxIter: 6\nChebyshev_Degree: 40\n"
            "Hartree_Boundary_Tolerance: off\nAtom_Types_Num: 1\nAtom_Type: H\n"
            "Local_Component: s\nbegin Atom_Coord\n 0.0 0.0 -0.7\n 0.0 0.0 0.7\n"
            "end Atom_Coord\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "parsec.in"
            path.write_text(text, encoding="utf-8")
            cls.problem = parse_parsec_input(
                path, pseudopotential_directory=DATA
            ).problem
        cls.system = prepare_single_point(cls.problem)
        cls.reduction = AxisReflectionReduction.detect(
            cls.system.grid, cls.problem.atoms
        )

    def test_wedge_mixers_reproduce_the_full_grid_scf_history(self) -> None:
        """The wedge loop follows the full-grid loop as the Anderson one does.

        The full-grid orbitals of the reference eigensolver, started from
        random vectors, leave the fields symmetric only to about 1e-9 in the
        residual; the wedge loop averages that part out over every orbit.
        Anderson, whose wedge mixer is the one of before, sets the scale.
        """

        self.assertEqual(self.reduction.group_order, 8)
        reducer = SymmetrySCFReducer(self.reduction)
        for options in ({}, {"method": "broyden"}, {"method": "msecant1"},
                        {"method": "msecant2"}, {"method": "msecant3", "group_size": 2}):
            system = replace(
                self.system,
                input=replace(self.problem, mixing=MixingSettings(**options)),
            )
            full = run_scf(system)
            wedge = run_scf(
                system,
                mixer_factory=reducer.mixer,
                residual_metrics_evaluator=reducer.potential_residual_metrics,
                total_energy_evaluator=reducer.total_energy,
            )
            self.assertTrue(full.converged, msg=str(options))
            self.assertEqual(full.iterations, wedge.iterations, msg=str(options))
            for a, b in zip(full.history, wedge.history):
                self.assertLess(
                    abs(a.weighted_residual - b.weighted_residual), 5e-9, msg=str(options)
                )
                self.assertAlmostEqual(a.energies.total, b.energies.total, places=10)


if __name__ == "__main__":
    unittest.main()
