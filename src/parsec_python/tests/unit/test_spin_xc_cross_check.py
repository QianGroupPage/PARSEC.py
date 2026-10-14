"""The two spin-polarized exchange-correlation stacks agree.

The isolated-system stack (``ca_lda_spin``, ``pbe_spin``) is validated against
Fortran PARSEC; the periodic / k-point / spin-orbit stack uses
``ca_lda_spin_polarized`` and ``pbe_spin_polarized``.  On a cluster grid with
the same densities they must give the same energies and potentials.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.V_xc import (
    ca_lda_spin,
    ca_lda_spin_polarized,
    pbe_spin,
    pbe_spin_polarized,
)

from .test_pbe_spin_polarized import _cluster_grid, _spin_densities


class SpinXCStacksAgreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = _cluster_grid()
        self.up, self.down = _spin_densities(self.grid)

    def _compare(self, main, mine, rtol: float) -> None:
        self.assertAlmostEqual(main.total_energy, mine.total_energy, delta=rtol * abs(main.total_energy))
        scale = max(np.abs(main.potential_up).max(), np.abs(main.potential_down).max())
        np.testing.assert_allclose(main.potential_up, mine.potential_up, atol=rtol * scale)
        np.testing.assert_allclose(main.potential_down, mine.potential_down, atol=rtol * scale)

    def test_lda(self) -> None:
        volume = self.grid.volume_element
        self._compare(
            ca_lda_spin(self.up, self.down, volume),
            ca_lda_spin_polarized(self.up, self.down, volume),
            1e-8,
        )

    def test_pbe(self) -> None:
        self._compare(
            pbe_spin(self.up, self.down, self.grid),
            pbe_spin_polarized(self.up, self.down, self.grid),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
