"""UPF v2 reader and its conversion to the tables of PARSEC.py (Pseudopotential/upf.py).

A synthetic file with known analytic tables (two projectors per ``l`` and a non-diagonal ``D``) checks the algebra; the
PseudoDojo files are checked where the directory exists (``PARSEC_DOJO_DIR``, default
``~/Research/pseudopotentials/nc-sr-05_pbe_standard_upf``).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Pseudopotential.upf import (
    LOCAL_CHANNEL,
    kleinman_bylander_channels,
    parsec_logarithmic_grid,
    read_upf,
    read_upf_pseudopotential,
)

_Z = 4.0
_MESH = np.arange(0, 1001) * 0.01  # 0 ... 10 bohr, the linear mesh of a PseudoDojo file
_DIJ = np.array(
    [
        [6.0, 2.0, 0.0, 0.0, 0.0],
        [2.0, -1.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, -3.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.25],
    ]
)
_BETA_L = (0, 0, 1, 1, 2)


def _beta(index: int, l: int) -> np.ndarray:
    r = _MESH
    width = 1.1 + 0.2 * (index % 2)
    # a smooth taper to zero at the cutoff radius 2.6: a jump there would make the energy non-smooth in the position
    taper = np.where(r < 2.6, (1.0 - (r / 2.6) ** 2) ** 3, 0.0)
    return r ** (l + 1) * np.exp(-((r / width) ** 2)) * (1.0 + 0.3 * index) * taper


def _rows(values: np.ndarray) -> str:
    return "\n".join("  ".join(f"{v:.12E}" for v in values[i : i + 4]) for i in range(0, values.size, 4))


def _local() -> np.ndarray:
    from scipy.special import erf

    r = np.where(_MESH > 0, _MESH, 1.0)
    return np.where(_MESH > 0, -2.0 * _Z * erf(r / 1.6) / r, -2.0 * _Z * 2.0 / (1.6 * np.sqrt(np.pi)))


def _density() -> np.ndarray:
    r = _MESH
    shape = np.exp(-((r / 1.5) ** 2))
    rho = shape / np.trapezoid(4.0 * np.pi * r**2 * shape, r) * _Z
    return 4.0 * np.pi * r**2 * rho


def _upf_text(*, spin_orbit: bool = False, core: bool = True, dij: np.ndarray = _DIJ, element: str = "Xx") -> str:
    size = _MESH.size
    parts = [
        '<UPF version="2.0.1">',
        "<PP_INFO>\nsynthetic\n</PP_INFO>",
        f'<PP_HEADER element="{element}" pseudo_type="NC" relativistic="scalar" has_so="{"T" if spin_orbit else "F"}" '
        f'core_correction="{"T" if core else "F"}" functional="PBE" z_valence="{_Z:.2f}" mesh_size="{size}" '
        'number_of_wfc="2" number_of_proj="5" />',
        f'<PP_MESH>\n<PP_R type="real" size="{size}" columns="4">\n{_rows(_MESH)}\n</PP_R>\n</PP_MESH>',
        f'<PP_LOCAL type="real" size="{size}" columns="4">\n{_rows(_local())}\n</PP_LOCAL>',
        "<PP_NONLOCAL>",
    ]
    for index, l in enumerate(_BETA_L, start=1):
        parts.append(
            f'<PP_BETA.{index} type="real" size="{size}" columns="4" index="{index}" angular_momentum="{l}" '
            f'cutoff_radius_index="200" cutoff_radius="2.6">\n{_rows(_beta(index - 1, l))}\n</PP_BETA.{index}>'
        )
    parts.append(f'<PP_DIJ type="real" size="25" columns="4">\n{_rows(dij.reshape(-1))}\n</PP_DIJ>')
    parts.append("</PP_NONLOCAL>")
    parts.append("<PP_PSWFC>")
    for index, l in enumerate((0, 1), start=1):
        wave = _MESH ** (l + 1) * np.exp(-_MESH)
        parts.append(
            f'<PP_CHI.{index} type="real" size="{size}" columns="4" label="{index}{"SP"[l]}" l="{l}" '
            f'occupation="{2.0 + 2.0 * l}" pseudo_energy="-0.5">\n{_rows(wave)}\n</PP_CHI.{index}>'
        )
    parts.append("</PP_PSWFC>")
    if core:
        core_density = 0.4 * np.exp(-((_MESH / 1.0) ** 2))
        parts.append(f'<PP_NLCC type="real" size="{size}" columns="4">\n{_rows(core_density)}\n</PP_NLCC>')
    parts.append(f'<PP_RHOATOM type="real" size="{size}" columns="4">\n{_rows(_density())}\n</PP_RHOATOM>')
    if spin_orbit:
        parts.append("<PP_SPIN_ORB>\n</PP_SPIN_ORB>")
    parts.append("</UPF>")
    return "\n".join(parts)


class SyntheticUpfTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.path = Path(self._directory.name) / "Xx.upf"
        self.path.write_text(_upf_text(), encoding="utf-8")

    def test_the_file_is_read_as_written(self) -> None:
        upf = read_upf(self.path)
        self.assertEqual(upf.symbol, "Xx")
        self.assertEqual(upf.z_valence, _Z)
        self.assertEqual([p.angular_momentum for p in upf.projectors], list(_BETA_L))
        np.testing.assert_allclose(upf.mesh, _MESH)
        np.testing.assert_allclose(upf.dij, _DIJ)
        np.testing.assert_allclose(upf.projectors[3].values, _beta(3, 1), rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(upf.local_potential, _local(), rtol=1e-11)
        self.assertEqual([w.angular_momentum for w in upf.wavefunctions], [0, 1])
        self.assertIsNotNone(upf.core_density)

    def test_the_separable_form_reproduces_the_coupled_operator(self) -> None:
        """sum_ij D_ij beta_i(r) beta_j(r') = sum_k sign_k chi_k(r) chi_k(r') r r' for each l (D not diagonal)."""
        upf = read_upf(self.path)
        channels = kleinman_bylander_channels(upf)
        grid = np.arange(1, 400, 7)
        r = _MESH[grid]
        for l in (0, 1, 2):
            members = [i for i, value in enumerate(_BETA_L) if value == l]
            exact = sum(
                _DIJ[i, j] * np.outer(upf.projectors[i].values[grid], upf.projectors[j].values[grid])
                for i in members
                for j in members
            )
            separable = sum(
                c.sign * np.outer(c.values[grid] * r, c.values[grid] * r) for c in channels if c.angular_momentum == l
            )
            np.testing.assert_allclose(separable, exact, rtol=1e-9, atol=1e-11)

    def test_signs_and_counts_follow_the_eigenvalues_of_each_block(self) -> None:
        channels = kleinman_bylander_channels(read_upf(self.path))
        for l, members in ((0, [0, 1]), (1, [2, 3]), (2, [4])):
            expected = np.sort(np.linalg.eigvalsh(_DIJ[np.ix_(members, members)]))
            found = np.sort([c.coupling for c in channels if c.angular_momentum == l])
            np.testing.assert_allclose(found, expected, rtol=1e-12)
            for c in channels:
                if c.angular_momentum == l:
                    self.assertEqual(c.sign, 1.0 if c.coupling > 0 else -1.0)
        self.assertEqual([c.index for c in channels if c.angular_momentum == 0], [0, 1])
        # the strongest coupling of an l is index 0
        first, second = (c for c in channels if c.angular_momentum == 0)
        self.assertGreater(abs(first.coupling), abs(second.coupling))

    def test_a_coupling_between_angular_momenta_is_refused(self) -> None:
        dij = _DIJ.copy()
        dij[1, 2] = dij[2, 1] = 0.7
        self.path.write_text(_upf_text(dij=dij), encoding="utf-8")
        with self.assertRaises(ValueError):
            kleinman_bylander_channels(read_upf(self.path))

    def test_a_spin_orbit_file_is_refused(self) -> None:
        self.path.write_text(_upf_text(spin_orbit=True), encoding="utf-8")
        with self.assertRaises(ValueError):
            read_upf(self.path)

    def test_the_conversion_keeps_charge_tail_and_projectors(self) -> None:
        potential = read_upf_pseudopotential(self.path)
        r = potential.radii
        self.assertEqual(potential.ionic_charge, _Z)
        self.assertEqual(potential.local_channel, LOCAL_CHANNEL)
        self.assertEqual(sorted(potential.channel_potentials), [LOCAL_CHANNEL])
        self.assertAlmostEqual(4.0 * np.pi * np.trapezoid(r**2 * potential.valence_density, r), _Z, places=4)
        self.assertTrue(potential.has_nonlinear_core_correction)
        # the Coulomb tail beyond the range of the error function (erf(9/1.6) = 1 to 1e-15)
        np.testing.assert_allclose(r[r > 9.0] * potential.channel_potentials[LOCAL_CHANNEL][r > 9.0], -2.0 * _Z, rtol=1e-9)
        # values of the file's tables on the new grid
        reference = np.interp(r[(r > 0.05) & (r < 5.0)], _MESH, _local())
        np.testing.assert_allclose(
            potential.channel_potentials[LOCAL_CHANNEL][(r > 0.05) & (r < 5.0)], reference, rtol=2e-4, atol=1e-6
        )
        channels = {(c.angular_momentum, c.index): c for c in potential.kb_channels}
        self.assertEqual(sorted(channels), [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        self.assertEqual(sorted(potential.radial_wavefunctions), [0, 1, 2])
        self.assertEqual(potential.projector_count(0), 2)
        self.assertEqual(potential.nonlocal_channels(LOCAL_CHANNEL), [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])

    def test_radial_projector_returns_table_and_sign_by_index(self) -> None:
        potential = read_upf_pseudopotential(self.path)
        table, sign = potential.radial_projector(0, LOCAL_CHANNEL, 1)
        channel = next(c for c in potential.kb_channels if (c.angular_momentum, c.index) == (0, 1))
        np.testing.assert_array_equal(table, channel.values)
        self.assertEqual(sign, channel.sign)
        with self.assertRaises(ValueError):
            potential.radial_projector(0, LOCAL_CHANNEL, 2)
        with self.assertRaises(ValueError):
            potential.radial_projector(3, LOCAL_CHANNEL)
        with self.assertRaises(ValueError):
            potential.radial_projector(0, 1)  # an l is not a local potential here

    def test_a_file_without_a_core_charge_has_none(self) -> None:
        self.path.write_text(_upf_text(core=False), encoding="utf-8")
        potential = read_upf_pseudopotential(self.path)
        self.assertFalse(potential.has_nonlinear_core_correction)
        self.assertFalse(np.any(potential.core_density))

    def test_the_logarithmic_grid_is_parsecs(self) -> None:
        radii, a, b = parsec_logarithmic_grid(10.0)
        np.testing.assert_allclose(radii, a * np.exp(b * np.arange(1, radii.size + 1)) - a)
        self.assertGreaterEqual(radii[-1], 10.0)
        self.assertLess(radii[-2], 10.0)
        # as fine as the 0.01 bohr mesh of a PseudoDojo file where the projectors live (r ~ 1 ... 2)
        spacing = np.diff(radii)[np.searchsorted(radii, 1.5)]
        self.assertLess(spacing, 0.011)


_DOJO = Path(os.environ.get("PARSEC_DOJO_DIR", "~/Research/pseudopotentials/nc-sr-05_pbe_standard_upf")).expanduser()


@unittest.skipUnless((_DOJO / "O.upf").exists(), "PseudoDojo files are not available")
class PseudoDojoTests(unittest.TestCase):
    #: element -> (valence electrons, projectors per l, core charge present)
    EXPECTED = {
        "Li": (3.0, {0: 2, 1: 2}, False),
        "La": (11.0, {0: 2, 1: 2, 2: 2, 3: 2}, True),
        "Zr": (12.0, {0: 2, 1: 2, 2: 2}, True),
        "O": (6.0, {0: 2, 1: 2, 2: 1}, True),
    }

    def test_charge_tail_and_projectors_of_the_llzo_elements(self) -> None:
        for symbol, (charge, counts, has_core) in self.EXPECTED.items():
            with self.subTest(symbol=symbol):
                upf = read_upf(_DOJO / f"{symbol}.upf")
                potential = read_upf_pseudopotential(_DOJO / f"{symbol}.upf")
                r = potential.radii
                self.assertAlmostEqual(upf.z_valence, charge)
                self.assertAlmostEqual(np.trapezoid(upf.valence_charge, upf.mesh), charge, places=3)
                self.assertAlmostEqual(4.0 * np.pi * np.trapezoid(r**2 * potential.valence_density, r), charge, places=3)
                # the tail of the file is the Coulomb one to its printed precision
                self.assertAlmostEqual(upf.mesh[-1] * upf.local_potential[-1], -2.0 * charge, places=4)
                self.assertEqual({l: potential.projector_count(l) for l in counts}, counts)
                self.assertEqual(potential.has_nonlinear_core_correction, has_core)
                # the files come with a diagonal D, so no rotation: each projector keeps its own d
                self.assertLess(np.max(np.abs(upf.dij - np.diag(np.diag(upf.dij)))), 1.0e-10)

    def test_every_scalar_relativistic_dojo_file_is_read(self) -> None:
        failures = []
        for path in sorted(_DOJO.glob("*.upf")):
            try:
                potential = read_upf_pseudopotential(path)
                if not np.all(np.isfinite(potential.channel_potentials[LOCAL_CHANNEL])):
                    failures.append((path.name, "nan"))
            except Exception as error:  # noqa: BLE001 - the list names every file that fails
                failures.append((path.name, repr(error)[:80]))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
