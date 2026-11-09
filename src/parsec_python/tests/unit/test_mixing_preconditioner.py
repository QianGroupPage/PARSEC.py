"""The Kerker preconditioner of the mixing of 3-D periodic cells (``Mixing_Preconditioner: kerker``).

The filter ``P(G) = (G^2 + q0^2/eps)/(G^2 + q0^2)``, ``P(0) = 1``, on plane waves of orthorhombic and oblique
cells; the charge part of two spin channels; the wrapped mixers (first step, unchanged fixed point, faster
convergence of a model with charge sloshing); the settings and keywords, their refusals, and an SCF of a small
bulk cell that ends at the energy of the run without the preconditioner.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from parsec_python.Grid.cell import reciprocal_lattice
from parsec_python.Input.parsec_input import (
    ParsecInputError,
    UnsupportedParsecOptionError,
    parse_parsec_input,
)
from parsec_python.Mixer import (
    AndersonMixer,
    GeneralizedBroydenMixer,
    KerkerPreconditioner,
    PreconditionedMixer,
    make_mixer,
    preconditioned_mixer,
)
from parsec_python.Mixer.preconditioner import kerker_filter
from parsec_python.models import MixingSettings

DATA = Path(__file__).resolve().parents[1] / "data"


def _points(lattice: np.ndarray, shape) -> np.ndarray:
    """Cartesian points ``i a1/n1 + j a2/n2 + k a3/n3``, row-major."""

    indices = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1).reshape(-1, 3)
    return (indices / np.asarray(shape, dtype=float)) @ lattice


def _plane_wave(lattice, shape, m, phase=0.3):
    g = np.asarray(m, dtype=float) @ reciprocal_lattice(lattice)
    return np.cos(_points(lattice, shape) @ g + phase), float(g @ g)


class _Grid:
    """The attributes of a bulk periodic grid that the preconditioner reads."""

    boundary_modes = ("wrap", "wrap", "wrap")

    def __init__(self, lattice, shape):
        self.cell = type("Cell", (), {"lattice_vectors": np.asarray(lattice, dtype=float)})()
        self.points_per_axis = np.asarray(shape)
        self.step_vectors = self.cell.lattice_vectors / self.points_per_axis[:, None]
        self.size = int(np.prod(shape))


class _System:
    def __init__(self, grid, mixing):
        self.grid = grid
        self.input = type("Input", (), {"mixing": mixing})()


CUBIC = np.diag([7.0, 7.0, 7.0])
OBLIQUE = np.array([[6.0, 0.0, 0.0], [1.5, 5.5, 0.0], [0.8, -1.1, 6.5]])


class FilterTests(unittest.TestCase):
    def test_plane_waves_are_scaled_by_the_filter(self) -> None:
        for lattice, shape in ((CUBIC, (8, 8, 8)), (np.diag([5.0, 6.0, 8.0]), (6, 7, 10)), (OBLIQUE, (9, 8, 10))):
            for q0, eps in ((1.0, float("inf")), (0.7, 5.7)):
                preconditioner = KerkerPreconditioner(lattice, shape, q0, eps)
                for m in ((1, 0, 0), (0, -1, 2), (2, 1, -1), (1, 1, 1)):
                    with self.subTest(lattice=lattice.tolist(), q0=q0, eps=eps, m=m):
                        wave, g2 = _plane_wave(lattice, shape, m)
                        expected = (g2 + (0.0 if np.isinf(eps) else q0 * q0 / eps)) / (g2 + q0 * q0)
                        np.testing.assert_allclose(preconditioner.apply(wave), expected * wave, atol=1.0e-12)

    def test_the_mean_passes_unchanged(self) -> None:
        preconditioner = KerkerPreconditioner(CUBIC, (6, 6, 6), 1.0)
        constant = np.full(216, 0.37)
        np.testing.assert_allclose(preconditioner.apply(constant), constant, atol=1.0e-15)
        self.assertEqual(kerker_filter(CUBIC, (6, 6, 6), 1.0)[0, 0, 0], 1.0)

    def test_the_operator_is_real_symmetric_and_positive_on_an_oblique_cell(self) -> None:
        # even counts: the Nyquist planes of an oblique cell take the mean of their aliases
        shape = (6, 8, 4)
        preconditioner = KerkerPreconditioner(OBLIQUE, shape, 0.9, 4.0)
        size = int(np.prod(shape))
        matrix = np.column_stack([preconditioner.apply(np.eye(size)[:, column]) for column in range(size)])
        np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-13)
        values = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        self.assertGreater(values.min(), 0.9 * 0.9 / 4.0 / (1.0 + 0.9 * 0.9) * 0.99)
        self.assertLessEqual(values.max(), 1.0 + 1.0e-12)

    def test_spin_channels_filter_the_charge_part_only(self) -> None:
        shape = (6, 6, 6)
        preconditioner = KerkerPreconditioner(CUBIC, shape, 1.0)
        wave, _ = _plane_wave(CUBIC, shape, (1, 0, 0))
        other, _ = _plane_wave(CUBIC, shape, (0, 2, 1), phase=1.1)
        up, down = wave + other, wave - other
        filtered = preconditioner.apply(np.concatenate((up, down)), channels=2)
        charge = preconditioner.apply(wave)
        np.testing.assert_allclose(filtered[:216], charge + other, atol=1.0e-12)
        np.testing.assert_allclose(filtered[216:], charge - other, atol=1.0e-12)

    def test_a_grid_that_is_not_periodic_along_every_axis_is_refused(self) -> None:
        grid = _Grid(CUBIC, (4, 4, 4))
        grid.boundary_modes = ("wrap", "wrap", "constant")
        with self.assertRaises(NotImplementedError):
            KerkerPreconditioner.for_grid(grid, 1.0)
        grid = _Grid(CUBIC, (4, 4, 4))
        grid.is_confined_periodic = True
        with self.assertRaises(NotImplementedError):
            KerkerPreconditioner.for_grid(grid, 1.0)
        cluster = type("Cluster", (), {"size": 10})()
        with self.assertRaises(NotImplementedError):
            KerkerPreconditioner.for_grid(cluster, 1.0)


class _SloshingModel:
    """A linear fixed point ``V_out = V* + (1 - eps) (V_in - V*)``, ``eps(G) = 1 + kTF^2/G^2`` on a cubic cell.

    The response of a metal in Thomas-Fermi form; the mean of ``V_out`` is that of ``V*`` (the gauge).
    """

    def __init__(self, edge: float, count: int, screening: float = 1.2):
        self.shape = (count,) * 3
        lattice = np.diag([edge] * 3)
        g2 = np.zeros(self.shape)
        for axis in range(3):
            m = np.rint(np.fft.fftfreq(count, d=1.0 / count)) * 2.0 * np.pi / edge
            g2 = g2 + (m**2).reshape([-1 if index == axis else 1 for index in range(3)])
        self.response = np.where(g2 > 0.0, -screening**2 / np.where(g2 > 0.0, g2, 1.0), 0.0)
        rng = np.random.default_rng(7)
        self.fixed_point = rng.standard_normal(count**3)
        self.lattice = lattice

    def output(self, potential):
        error = (potential - self.fixed_point).reshape(self.shape)
        return self.fixed_point + np.real(np.fft.ifftn(self.response * np.fft.fftn(error))).reshape(-1)

    def iterations(self, mixer, limit=300, tolerance=1.0e-9):
        potential = np.zeros_like(self.fixed_point)
        for iteration in range(1, limit + 1):
            output = self.output(potential)
            if np.linalg.norm(output - potential) < tolerance * np.sqrt(potential.size):
                return iteration, potential
            potential = mixer.mix(potential, output, iteration=iteration)
        return limit, potential


class WrappedMixerTests(unittest.TestCase):
    def test_none_keeps_the_mixer_object(self) -> None:
        mixer = AndersonMixer(MixingSettings())
        system = _System(_Grid(CUBIC, (4, 4, 4)), MixingSettings())
        self.assertIs(preconditioned_mixer(mixer, system), mixer)

    def test_first_step_is_linear_mixing_of_the_filtered_residual(self) -> None:
        settings = MixingSettings(preconditioner="kerker", kerker_wavevector=0.8)
        grid = _Grid(CUBIC, (6, 6, 6))
        mixer = preconditioned_mixer(make_mixer(settings), _System(grid, settings))
        self.assertIsInstance(mixer, PreconditionedMixer)
        rng = np.random.default_rng(3)
        given, output = rng.standard_normal(216), rng.standard_normal(216)
        expected = given + 0.3 * KerkerPreconditioner(CUBIC, (6, 6, 6), 0.8).apply(output - given)
        np.testing.assert_allclose(mixer.mix(given, output, iteration=1), expected, atol=1.0e-13)
        self.assertEqual(mixer.safeguard_resets, 0)
        self.assertIs(mixer.settings, settings)

    def test_sloshing_converges_faster_to_the_same_fixed_point(self) -> None:
        for edge in (20.0, 40.0):
            model = _SloshingModel(edge, 12)
            grid = _Grid(model.lattice, model.shape)
            plain = MixingSettings(parameter=0.3, memory=4, restart=200)
            kerker = replace(plain, preconditioner="kerker", kerker_wavevector=1.2)
            counts = {}
            for name, settings in (("none", plain), ("kerker", kerker)):
                mixer = preconditioned_mixer(make_mixer(settings), _System(grid, settings))
                counts[name], potential = model.iterations(mixer)
                if name == "kerker" or counts[name] < 300:
                    with self.subTest(edge=edge, mixer=name):
                        np.testing.assert_allclose(potential, model.fixed_point, atol=1.0e-7)
            with self.subTest(edge=edge):
                self.assertLess(counts["kerker"], 15)
                self.assertGreater(counts["none"], 2 * counts["kerker"])

    def test_broyden_takes_the_filtered_residual_too(self) -> None:
        model = _SloshingModel(30.0, 10)
        settings = MixingSettings(method="broyden", parameter=0.3, memory=4, restart=200, preconditioner="kerker",
                                  kerker_wavevector=1.2)
        mixer = preconditioned_mixer(make_mixer(settings), _System(_Grid(model.lattice, model.shape), settings))
        self.assertIsInstance(mixer.inner, GeneralizedBroydenMixer)
        count, potential = model.iterations(mixer)
        self.assertLess(count, 30)
        np.testing.assert_allclose(potential, model.fixed_point, atol=1.0e-7)

    def test_an_adapter_filters_on_the_full_grid(self) -> None:
        class Halves:
            """A scalar-field adapter of a permuted full grid (the order of the wedge values)."""

            def __init__(self, size):
                self.order = np.random.default_rng(1).permutation(size)

            def to_full(self, values):
                full = np.empty_like(values)
                full[self.order] = values
                return full

            def from_full(self, values):
                return np.asarray(values)[self.order]

        settings = MixingSettings(preconditioner="kerker")
        grid = _Grid(CUBIC, (6, 6, 6))
        adapter = Halves(216)
        mixer = preconditioned_mixer(AndersonMixer(settings), _System(grid, settings), adapter=adapter)
        rng = np.random.default_rng(5)
        given, output = rng.standard_normal(216), rng.standard_normal(216)
        mixed = mixer.mix(adapter.from_full(given), adapter.from_full(output), iteration=1)
        expected = given + 0.3 * KerkerPreconditioner(CUBIC, (6, 6, 6), 1.0).apply(output - given)
        np.testing.assert_allclose(adapter.to_full(mixed), expected, atol=1.0e-13)
        with self.assertRaises(NotImplementedError):
            preconditioned_mixer(AndersonMixer(settings), _System(grid, settings), adapter=adapter, channels=2)


class SettingsTests(unittest.TestCase):
    def test_defaults_print_nothing_new(self) -> None:
        self.assertNotIn("preconditioner", repr(MixingSettings()))
        self.assertNotIn("kerker", repr(MixingSettings(method="broyden")))
        self.assertIn("preconditioner='kerker'", repr(MixingSettings(preconditioner="kerker")))

    def test_values_are_checked(self) -> None:
        for keywords in (
            {"preconditioner": "tf"},
            {"preconditioner": "kerker", "kerker_wavevector": 0.0},
            {"preconditioner": "kerker", "kerker_wavevector": float("inf")},
            {"preconditioner": "kerker", "kerker_dielectric_constant": 0.5},
        ):
            with self.subTest(**keywords), self.assertRaises(ValueError):
                MixingSettings(**keywords)
        self.assertEqual(MixingSettings(preconditioner=" Kerker ").preconditioner, "kerker")


_BULK = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
 10.26 10.26 10.26
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: lattice_vectors
Grid_Spacing: 0.75
Expansion_Order: 6
States_Num: 26
Fermi_Temp: 500 K
Max_Iter: 80
Convergence_Criterion: 1.0e-6 Ry
Correlation_Type: ca
Eigensolver: chebff
Mixing_Method: Anderson
Mixing_Param: 0.3
{extra}
Atom_Types_Num: 1
Atom_Type: Si
Local_Component: p
begin Atom_Coord
 0.00 0.00 0.00
 0.00 0.50 0.50
 0.50 0.00 0.50
 0.50 0.50 0.00
 0.25 0.25 0.25
 0.25 0.75 0.75
 0.75 0.25 0.75
 0.75 0.75 0.25
end Atom_Coord
"""

_CLUSTER = (
    "Boundary_Sphere_Radius: 4.0 bohr\nGrid_Spacing: 0.8 bohr\n"
    "Coordinate_Unit: Cartesian_Bohr\nExpansion_Order: 4\nStates_Num: 3\n"
    "Eigensolver: chebff\n{extra}\nAtom_Types_Num: 1\nAtom_Type: H\n"
    "Local_Component: s\nbegin Atom_Coord\n 0.0 0.0 0.0\nend Atom_Coord\n"
)


class _Inputs(unittest.TestCase):
    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.directory = Path(self._folder.name)
        self.count = 0

    def write(self, text: str) -> Path:
        self.count += 1
        folder = self.directory / f"case{self.count}"
        folder.mkdir()
        for name in ("Si_POTRE.DAT", "H_POTRE.DAT"):
            shutil.copy(DATA / name, folder / name)
        path = folder / "parsec.in"
        path.write_text(text, encoding="utf-8")
        return path


class KeywordTests(_Inputs):
    def test_keywords_and_units(self) -> None:
        mixing = parse_parsec_input(self.write(_BULK.format(extra=""))).problem.mixing
        self.assertEqual(mixing.preconditioner, "none")
        self.assertEqual(mixing, MixingSettings())
        mixing = parse_parsec_input(
            self.write(_BULK.format(extra="Mixing_Preconditioner: Kerker\nKerker_Wavevector: 1.0 ang^-1"))
        ).problem.mixing
        self.assertEqual(mixing.preconditioner, "kerker")
        self.assertAlmostEqual(mixing.kerker_wavevector, 0.529177, places=6)
        self.assertEqual(mixing.kerker_dielectric_constant, float("inf"))
        mixing = parse_parsec_input(
            self.write(_BULK.format(extra="Mixing_Preconditioner: kerker\nKerker_Dielectric_Constant: 12"))
        ).problem.mixing
        self.assertEqual((mixing.kerker_wavevector, mixing.kerker_dielectric_constant), (1.0, 12.0))

    def test_auto_and_inert_keywords(self) -> None:
        translation = parse_parsec_input(self.write(_BULK.format(extra="Mixing_Preconditioner: auto\nKerker_Wavevector: 2")))
        self.assertEqual(translation.problem.mixing, MixingSettings())
        self.assertTrue(any(w.startswith("Kerker_Wavevector has no effect") for w in translation.warnings))
        translation = parse_parsec_input(self.write(_CLUSTER.format(extra="Mixing_Preconditioner: auto")))
        self.assertEqual(translation.problem.mixing.preconditioner, "none")

    def test_refusals(self) -> None:
        with self.assertRaises(ParsecInputError):
            parse_parsec_input(self.write(_BULK.format(extra="Mixing_Preconditioner: pulay")))
        with self.assertRaises(UnsupportedParsecOptionError):
            parse_parsec_input(self.write(_CLUSTER.format(extra="Mixing_Preconditioner: kerker")))
        with self.assertRaises(UnsupportedParsecOptionError):
            parse_parsec_input(
                self.write(_BULK.format(extra="Mixing_Preconditioner: kerker").replace("Correlation_Type: ca", "Correlation_Type: scan"))
            )
        with self.assertRaises(ParsecInputError):
            parse_parsec_input(self.write(_BULK.format(extra="Mixing_Preconditioner: kerker\nKerker_Wavevector: -1")))


class BulkSCFTests(_Inputs):
    def energy_and_iterations(self, extra: str) -> tuple[float, int]:
        from parsec_python.cli import main as cli_main

        path = self.write(_BULK.format(extra=extra))
        self.assertEqual(cli_main([str(path), "--quiet"]), 0)
        with np.load(path.parent / "parsec_python_results.npz") as archive:
            energy = float(archive["energy_total_ry"])
        text = (path.parent / "parsec.out").read_text(encoding="utf-8", errors="replace")
        iterations = text.count("SRE of pot. & charge weighted pot")
        return energy, iterations

    def test_the_preconditioned_scf_ends_at_the_same_energy(self) -> None:
        plain, _ = self.energy_and_iterations("")
        kerker, _ = self.energy_and_iterations("Mixing_Preconditioner: kerker\nKerker_Wavevector: 1.0")
        dielectric, _ = self.energy_and_iterations(
            "Mixing_Preconditioner: kerker\nKerker_Wavevector: 1.0\nKerker_Dielectric_Constant: 12"
        )
        # Convergence_Criterion 1e-6 Ry (SRE): the energies of two converged paths agree far below it
        self.assertAlmostEqual(kerker, plain, delta=2.0e-6)
        self.assertAlmostEqual(dielectric, plain, delta=2.0e-6)


if __name__ == "__main__":
    unittest.main()
