"""Post-SCF B3LYP refinement on the accelerated backends.

The physics is tested with the reference package
(``parsec_python/tests/unit/test_energy_refinement.py``).  Here the pieces an
accelerated calculation supplies are compared with it: the pair-density
Poisson solver built from the Hartree components of a prepared system, the
orbitals of symmetry-reduced and device runs, and the command-line hook.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import parsec_python.acceleration.cli as cli_module
from parsec_python.Energy.refinement import refine_energy as refine_reference_energy
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.V_xc.exact_exchange import reference_pair_poisson
from parsec_python.acceleration.Hartree.pair_poisson import build_pair_poisson
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.driver import prepare_single_point, run_scf
from parsec_python.acceleration.refinement import refine_energy
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA = PACKAGE_ROOT / "tests" / "data"
BACKENDS = ("scipy",) + (("native",) if native_available() else ())


def hydrogen_molecule() -> SinglePointInput:
    """H2 on a coarse grid: one doubly occupied orbital, reflection symmetry."""

    return SinglePointInput(
        atoms=[Atom("H", [0.0, 0.0, 0.7]), Atom("H", [0.0, 0.0, -0.7])],
        pseudopotentials={"H": SpeciesPotential(DATA / "H_POTRE.DAT", 0)},
        grid=GridSettings(spacing=0.6, radius=6.0, expansion_order=8),
        scf=SCFSettings(
            max_iterations=60,
            number_of_states=2,
            convergence_criterion=1.0e-6,
            # The bundled hydrogen pseudopotential is a CA-LDA one.
            xc_functional="ca",
        ),
        # PARSEC's boundary: no atomic tail, so that the Hartree energy of
        # the SCF and the pair solves share their boundary model.
        hartree=HartreeSettings(
            relative_tolerance=1.0e-10, max_iterations=4000, boundary_tolerance=None
        ),
        eigensolver=EigensolverSettings(method="chebff"),
    )


class PairPoissonTests(unittest.TestCase):
    def test_pair_solver_of_every_host_backend_is_the_reference_solve(self) -> None:
        problem = hydrogen_molecule()
        for backend in BACKENDS:
            for symmetry in ("off", "auto"):
                with self.subTest(backend=backend, symmetry=symmetry):
                    system = prepare_single_point(
                        problem, backend=backend, symmetry=symmetry
                    )
                    grid = system.grid
                    points = grid.coordinates
                    # A pair density without the symmetry of the molecule.
                    shifted = points - np.array([0.4, -0.3, 0.2])
                    density = (points[:, 0] + 0.3) * np.exp(
                        -0.6 * np.einsum("ij,ij->i", shifted, shifted)
                    )
                    reference = reference_pair_poisson(
                        grid, build_negative_laplacian(grid), system.input.hartree
                    )(density)
                    solver = build_pair_poisson(system)
                    solved = solver(density)
                    np.testing.assert_allclose(
                        np.asarray(solved.potential),
                        reference.potential,
                        rtol=0.0,
                        atol=1.0e-8 * np.max(np.abs(reference.potential)),
                    )
                    # An explicit SciPy calculation stays on SciPy.
                    self.assertIn(
                        "SciPy conjugate gradients"
                        if backend == "scipy"
                        else "C++/OpenMP conjugate gradients",
                        solver.description,
                    )
                    self.assertIn("multipole boundary of order 9", solver.description)
                    # Linear in the density: no static boundary term.
                    doubled = solver(2.0 * density)
                    np.testing.assert_allclose(
                        np.asarray(doubled.potential),
                        2.0 * np.asarray(solved.potential),
                        rtol=0.0,
                        atol=1.0e-8 * np.max(np.abs(reference.potential)),
                    )


class BoxDomainPairPoissonTests(unittest.TestCase):
    def test_box_pair_solver_uses_the_direct_boundary_of_the_pair_density(self) -> None:
        """A box has no multipole boundary: the direct Coulomb sum of the pair."""

        base = hydrogen_molecule()
        problem = replace(
            base,
            grid=GridSettings(
                spacing=0.8,
                radius=6.0,
                expansion_order=4,
                domain_shape="box",
                box_lengths=(7.0, 7.0, 8.0),
            ),
        )
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                system = prepare_single_point(problem, backend=backend, symmetry="off")
                grid = system.grid
                points = grid.coordinates
                density = (0.2 + points[:, 2]) * np.exp(
                    -0.5 * np.einsum("ij,ij->i", points, points)
                )
                reference = reference_pair_poisson(
                    grid, build_negative_laplacian(grid), system.input.hartree
                )(density)
                solver = build_pair_poisson(system)
                self.assertIn("direct", solver.description)
                np.testing.assert_allclose(
                    np.asarray(solver(density).potential),
                    reference.potential,
                    rtol=0.0,
                    atol=1.0e-8 * np.max(np.abs(reference.potential)),
                )


class AcceleratedRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = hydrogen_molecule()
        system = prepare_single_point(cls.problem, backend="scipy", symmetry="off")
        result = run_scf(system)
        assert result.converged
        cls.reference = refine_reference_energy(system.reference, result)
        cls.hartree = float(result.energies.hartree)

    def check(self, backend: str, symmetry: str) -> None:
        system = prepare_single_point(self.problem, backend=backend, symmetry=symmetry)
        # As the command line leaves them without an archive of the orbitals.
        system.materialize_final_wavefunctions = False
        result = run_scf(system)
        self.assertTrue(result.converged)
        refinement = refine_energy(system, result)
        self.assertIsInstance(result.wavefunctions, np.ndarray)
        self.assertEqual(refinement.exchange.pair_solves, 1)
        # Two electrons in one orbital: E_x = -E_Hartree / 2.
        self.assertAlmostEqual(
            refinement.exact_exchange / (-0.5 * float(result.energies.hartree)),
            1.0,
            delta=5.0e-7,
        )
        for name in (
            "exact_exchange",
            "slater_exchange",
            "b88_exchange",
            "vwn_correlation",
            "lyp_correlation",
            "exchange_correlation",
            "scf_exchange_correlation",
            "total",
        ):
            with self.subTest(backend=backend, symmetry=symmetry, energy=name):
                self.assertAlmostEqual(
                    getattr(refinement, name), getattr(self.reference, name), delta=2.0e-6
                )

    def test_host_backends_with_and_without_symmetry(self) -> None:
        for backend in BACKENDS:
            for symmetry in ("off", "auto"):
                self.check(backend, symmetry)

    @unittest.skipUnless(cupy_available(), "CuPy/CUDA is not available")
    def test_device_orbitals_and_symmetry_sectors(self) -> None:
        for backend in ("cupy", "auto"):
            for symmetry in ("off", "auto"):
                self.check(backend, symmetry)

    def test_reference_identity(self) -> None:
        self.assertAlmostEqual(
            self.reference.exact_exchange / (-0.5 * self.hartree), 1.0, delta=5.0e-7
        )
        self.assertAlmostEqual(
            self.reference.total,
            self.reference.scf_total
            - self.reference.scf_exchange_correlation
            + self.reference.exchange_correlation,
            places=12,
        )


class AcceleratedCommandLineTests(unittest.TestCase):
    INPUT = DATA / "H_cli_smoke.in"

    def run_cli(self, *options: str):
        messages: list[str] = []
        tables: list[dict] = []

        class MemoryLog:
            def __init__(self, path: Path, quiet: bool = False) -> None:
                del path, quiet

            def __enter__(self) -> "MemoryLog":
                return self

            def __exit__(self, _type, _value, _traceback) -> None:
                return None

            def write(self, message: str = "") -> None:
                messages.append(message)

        def fake_archive(path, _result, *, include_wavefunctions=False, **given):
            del include_wavefunctions
            tables.append(given)
            return Path(path)

        with (
            patch.object(cli_module, "_RunLog", MemoryLog),
            patch.object(cli_module, "save_result_archive", side_effect=fake_archive),
        ):
            code = cli_module.main(
                [str(self.INPUT), "--backend", "scipy", "--quiet", *options]
            )
        return code, "\n".join(messages), tables

    def test_option_adds_the_refinement_after_the_scf_report(self) -> None:
        # Max_Iter=1 stops before self-consistency: exit status 3 either way.
        code, plain, plain_tables = self.run_cli()
        self.assertEqual(code, 3)
        self.assertNotIn("refinement", plain.lower())
        self.assertEqual(plain_tables, [{}])

        code, refined, tables = self.run_cli("--energy-refinement", "b3lyp")
        self.assertEqual(code, 3)
        self.assertIn("Post-SCF energy refinement: B3LYP", refined)
        self.assertIn("Energy refinement: B3LYP exchange-correlation energy", refined)
        self.assertLess(
            refined.index("Maximum SCF iterations reached"),
            refined.index("Energy refinement: B3LYP"),
        )
        # One electron in a spin-unpolarized orbital and an unconverged SCF.
        self.assertIn("fractional occupations", refined)
        self.assertIn("the SCF did not converge", refined)
        self.assertIn(" Post-SCF energy refinement [sec] :", refined)
        self.assertEqual(list(tables[0]), ["refinement"])
        refinement = tables[0]["refinement"]
        self.assertEqual(refinement.functional, "b3lyp")
        self.assertTrue(refinement.exchange.fractional)
        self.assertAlmostEqual(refinement.exchange.largest_fractional_deviation, 0.5)

        code, off, off_tables = self.run_cli("--energy-refinement", "none")
        self.assertEqual(code, 3)
        self.assertNotIn("refinement", off.lower())
        self.assertEqual(off_tables, [{}])


if __name__ == "__main__":
    unittest.main()
