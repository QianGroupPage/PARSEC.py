"""``bands_spin.dat`` and the spin-texture plot: the expectation values of the spin along a band path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.models import BandStructureResult
from parsec_python.Output import bands as bands_module
from parsec_python.Output.bands import (
    degenerate_levels,
    level_averaged_spin,
    plot_spin_texture,
    save_band_structure_archive,
    write_spin_file,
)

ALPHA = 0.02  # Ry bohr: the Rashba splitting 2*alpha*k


def rashba_bands(points: int = 9) -> BandStructureResult:
    """Two Rashba branches of a free 2D electron along kx: E = k^2/2 -/+ alpha k, spin along -/+ y.

    The two branches are the ``-`` and ``+`` helicity states; at k = 0 they are degenerate and, as an
    eigensolver would return them, an arbitrary orthonormal mixture of the pair.
    """
    kx = np.linspace(-0.2, 0.2, points)
    k_points = np.column_stack((kx, np.zeros(points), np.zeros(points)))
    energies = np.empty((1, points, 2))
    spin = np.zeros((1, points, 2, 3))
    for i, k in enumerate(kx):
        low, high = 0.5 * k * k - ALPHA * abs(k), 0.5 * k * k + ALPHA * abs(k)
        energies[0, i] = [low, high]
        direction = np.sign(k)  # the helicity flips the spin with the sign of k
        spin[0, i, 0, 1] = -direction
        spin[0, i, 1, 1] = direction
        if k == 0.0:  # a degenerate pair: any rotation of the two states; here spins along +x and -x
            spin[0, i, 0] = [1.0, 0.0, 0.0]
            spin[0, i, 1] = [-1.0, 0.0, 0.0]
    return BandStructureResult(
        mode="soc",
        channels=("all",),
        segment_names=("-X-X",),
        segment_counts=np.array([points]),
        k_points=k_points,
        eigenvalues=energies,
        residual_norms=np.zeros((1, points)),
        fermi_level=0.0,
        spin_expectation=spin,
    )


class DegenerateLevelTests(unittest.TestCase):
    def test_levels_are_runs_of_states_within_the_tolerance(self) -> None:
        levels = degenerate_levels(np.array([0.0, 1e-6, 5e-5, 1.0, 1.00005, 3.0]), 1e-4)
        self.assertEqual([level.tolist() for level in levels], [[0, 1, 2], [3, 4], [5]])

    def test_mean_spin_of_a_level_is_gauge_independent(self) -> None:
        result = rashba_bands()
        averaged, sizes = level_averaged_spin(result)
        middle = len(result.k_points) // 2
        self.assertEqual(sizes[middle].tolist(), [2, 2])
        np.testing.assert_allclose(averaged[middle], 0.0, atol=1e-12)  # the arbitrary x spins cancel
        # away from k = 0 the branches are non-degenerate and keep their opposite in-plane spins
        self.assertEqual(sizes[-1].tolist(), [1, 1])
        np.testing.assert_allclose(averaged[-1, :, 1], [-1.0, 1.0])
        np.testing.assert_allclose(averaged[0, :, 1], [1.0, -1.0])  # and flip with -k

    def test_a_scalar_band_structure_has_no_spin_file(self) -> None:
        result = rashba_bands()
        result.spin_expectation = None
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(write_spin_file(directory, result))
        with self.assertRaises(ValueError):
            level_averaged_spin(result)


class SpinFileTests(unittest.TestCase):
    def test_file_has_a_block_per_band_in_gnuplot_layout(self) -> None:
        result = rashba_bands()
        with tempfile.TemporaryDirectory() as directory:
            path = write_spin_file(directory, result)
            text = path.read_text()
        header = [line for line in text.splitlines() if line.startswith("#")]
        self.assertIn("hbar/2", header[0])
        self.assertIn("S_x", header[-1])
        blocks = [block for block in text.split("\n \n") if any(not l.startswith("#") for l in block.splitlines())]
        self.assertEqual(len(blocks), 2)
        table = np.array(
            [line.split() for line in text.splitlines() if line.strip() and not line.startswith("#")], dtype=float
        )
        self.assertEqual(table.shape, (18, 7))
        lower = table[:9]
        np.testing.assert_allclose(lower[:, 2], result.eigenvalues[0, :, 0] * bands_module.RYDBERG_TO_EV, atol=1e-7)
        np.testing.assert_allclose(lower[:, 1], result.path_distance, atol=1e-7)
        self.assertEqual(lower[:, 6].tolist(), [1, 1, 1, 1, 2, 1, 1, 1, 1])
        np.testing.assert_allclose(lower[:, 4], [1, 1, 1, 1, 0, -1, -1, -1, -1], atol=1e-6)  # S_y of the lower branch


class SpinPlotTests(unittest.TestCase):
    def test_plot_and_command_line(self) -> None:
        result = rashba_bands()
        with tempfile.TemporaryDirectory() as directory:
            figure = plot_spin_texture(result, components="y", window_ev=None, output=Path(directory) / "y.png")
            self.assertEqual(len(figure.axes) - 1, 1)  # one panel and the colour bar
            self.assertTrue((Path(directory) / "y.png").exists())
            archive = save_band_structure_archive(Path(directory) / "parsec_python_bands.npz", result)
            output = Path(directory) / "texture.png"
            status = bands_module.main(["--spin", str(archive), "--components", "xyz", "-o", str(output)])
            self.assertEqual(status, 0)
            self.assertTrue(output.exists())
            with self.assertRaises(ValueError):
                plot_spin_texture(result, components="w")


if __name__ == "__main__":
    unittest.main()
