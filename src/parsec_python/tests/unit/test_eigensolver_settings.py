from __future__ import annotations

import unittest

from parsec_python.models import EigensolverSettings


class EigensolverSettingsTests(unittest.TestCase):
    def test_defaults_are_parsec_chebff_controls(self) -> None:
        settings = EigensolverSettings()

        self.assertEqual(settings.method, "chebff")
        self.assertEqual(settings.first_filter_degree, 20)
        # FF_MaxIter of CHEBFF defaults to 3 (PARSEC 1.5) since 2026-11-04 (2 before).
        self.assertEqual(settings.first_filter_cycles, 3)
        self.assertEqual(settings.matvec_block_size, 6)
        self.assertEqual(settings.subspace_buffer, 6)
        self.assertEqual(settings.filter_degree, 15)
        self.assertEqual(settings.filter_degree_delta, 3)
        self.assertEqual(settings.lanczos_steps, 5)
        self.assertFalse(hasattr(settings, "fallback_to_lanczos"))

    def test_algorithm_names_are_explicit(self) -> None:
        self.assertEqual(EigensolverSettings(method="chebff").method, "chebff")
        self.assertEqual(EigensolverSettings(method="chebdav").method, "chebdav")
        self.assertEqual(
            EigensolverSettings(
                method="arpack",
                subspace_buffer=0,
                matvec_block_size=4,
            ).method,
            "arpack",
        )

        for collapsed_name in ("chebyshev", "lanczos"):
            with self.subTest(collapsed_name=collapsed_name):
                with self.assertRaisesRegex(ValueError, "eigensolver method"):
                    EigensolverSettings(method=collapsed_name)  # type: ignore[arg-type]

    def test_strict_filter_ranges_are_validated(self) -> None:
        invalid_values = (
            ({"first_filter_degree": 9}, "first_filter_degree"),
            ({"first_filter_cycles": 0}, "first_filter_cycles"),
            # PARSEC 1.5 accepts FF_MaxIter up to 50 (usrinputfile.F90:2303 of
            # 1.5; the earlier PARSEC 1..9), so 10 is valid since the 1.5 round.
            ({"first_filter_cycles": 51}, "first_filter_cycles"),
            ({"matvec_block_size": 0}, "matvec_block_size"),
            ({"subspace_buffer": 5}, "subspace buffer"),
        )
        for overrides, message in invalid_values:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    EigensolverSettings(**overrides)

    def test_first_filter_degree_minimum_depends_on_initial_method(self) -> None:
        # PARSEC's CHEBFF path permits degree 10, whereas CHEBDAV clamps its
        # first-filter polynomial to at least 15 before entering the solver.
        self.assertEqual(
            EigensolverSettings(
                method="chebff",
                first_filter_degree=10,
            ).first_filter_degree,
            10,
        )
        self.assertEqual(
            EigensolverSettings(
                method="chebdav",
                first_filter_degree=15,
            ).first_filter_degree,
            15,
        )
        with self.assertRaisesRegex(
            ValueError,
            "first_filter_degree must be at least 15 for chebdav",
        ):
            EigensolverSettings(
                method="chebdav",
                first_filter_degree=14,
            )

    def test_arpack_does_not_require_filter_buffer(self) -> None:
        settings = EigensolverSettings(
            method="arpack",
            subspace_buffer=0,
            matvec_block_size=1,
        )
        self.assertEqual(settings.subspace_buffer, 0)


if __name__ == "__main__":
    unittest.main()
