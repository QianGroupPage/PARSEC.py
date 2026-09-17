"""Density construction that keeps representation orbitals on the wedge."""

from __future__ import annotations

import gc
import os
from time import perf_counter
from typing import Any

import numpy as np

from ..Eigensolvers.symmetry import CuPySymmetryOrbitals
from ..SCF.symmetry_fields import SymmetrySCFReducer


class CuPySymmetryDensityBuilder:
    """Build the full scalar density without expanding every wavefunction.

    A real one-dimensional representation differs between symmetry images
    only by a sign.  Since the density contains ``|psi|**2``, all images have
    the same value.  The wrapped ordinary CuPy builder therefore evaluates
    the globally ordered, already-normalized wedge columns once, downloads a
    wedge-length density, and expands that scalar field by orbit lookup.
    """

    def __init__(
        self,
        device_builder: Any,
        timing_stats: Any | None = None,
        reducer: SymmetrySCFReducer | None = None,
    ) -> None:
        self.device_builder = device_builder
        self.timing_stats = timing_stats
        self.reducer = reducer

    def __call__(
        self,
        wavefunctions: CuPySymmetryOrbitals,
        occupations: np.ndarray,
        volume_element: float,
    ) -> np.ndarray:
        if not isinstance(wavefunctions, CuPySymmetryOrbitals):
            return self.device_builder(
                wavefunctions, occupations, volume_element
            )

        if wavefunctions.scaled_wedge_vectors is not None:
            wedge_density = self.device_builder(
                wavefunctions.scaled_wedge_vectors,
                occupations,
                volume_element,
            )
        else:
            if (
                wavefunctions.representation_columns is None
                or wavefunctions.sector_vectors is None
                or wavefunctions.sector_orbits is None
                or wavefunctions.sector_scales is None
                or wavefunctions.wedge_size is None
            ):
                raise RuntimeError("lazy symmetry orbitals are incomplete")
            wedge_density = np.zeros(
                wavefunctions.wedge_size, dtype=np.float64
            )
            # rho is invariant even when psi changes sign between symmetry
            # images.  Accumulate each representation directly from the Ritz
            # vectors the sector eigensolver already owns:
            #   rho_w = (2/h^3) sum_r sum_{n in r}
            #           f_n |q_{r,w,n}|^2 / |O_w|.
            # This is algebraically identical to first scattering all scaled
            # orbitals into a dense wedge matrix, but avoids that large copy.
            for representation, vectors in enumerate(
                wavefunctions.sector_vectors
            ):
                output_columns = np.flatnonzero(
                    wavefunctions.representations == representation
                )
                if output_columns.size == 0:
                    continue
                source_columns = wavefunctions.representation_columns[
                    output_columns
                ]
                # Eigenvalues within each representation are sorted, so the
                # globally selected states form a prefix.  Preserve a
                # zero-copy view in the common case; retain exact advanced
                # indexing for defensive support of a non-prefix selection.
                if np.array_equal(
                    source_columns,
                    np.arange(source_columns.size, dtype=source_columns.dtype),
                ):
                    selected = vectors[:, : source_columns.size]
                else:
                    selected = vectors[:, source_columns]
                sector_density = self.device_builder(
                    selected,
                    occupations[output_columns],
                    volume_element,
                )
                scales = wavefunctions.sector_scales[representation]
                wedge_density[
                    wavefunctions.sector_orbits[representation]
                ] += sector_density * scales * scales
            self._release_large_unused_pool(wavefunctions)
        if self.reducer is not None:
            return self.reducer.field(wedge_density)
        expansion_started = perf_counter()
        full_to_wedge = np.asarray(
            wavefunctions.full_to_wedge, dtype=np.int64
        )
        density = np.ascontiguousarray(wedge_density[full_to_wedge])
        if self.timing_stats is not None:
            self.timing_stats.density_seconds += (
                perf_counter() - expansion_started
            )
        return density

    @staticmethod
    def _release_large_unused_pool(
        wavefunctions: CuPySymmetryOrbitals,
    ) -> None:
        """Return cached ChebDav workspaces after a large sector solve.

        CuPy's default allocator deliberately caches freed blocks.  That is a
        speed win for ordinary systems, but a first ChebDav solve near the
        WDDM commit limit can retain many gigabytes of fragmented temporary
        workspaces.  A later SUBSPACE solve then fails despite the live Ritz
        vectors themselves fitting.  ``free_all_blocks`` touches only unused
        pool blocks; arrays referenced by the persistent eigensolver state are
        unaffected.
        """

        vectors = wavefunctions.sector_vectors
        if not vectors:
            return
        threshold = int(
            os.environ.get(
                "PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES",
                str(512 * 1024 * 1024),
            )
        )
        if threshold < 0:
            raise ValueError(
                "PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES must be nonnegative"
            )
        live_bytes = sum(int(getattr(item, "nbytes", 0)) for item in vectors)
        if live_bytes < threshold:
            return
        try:
            from ..backends.cupy import require_cupy

            cp, _ = require_cupy()
        except Exception:
            return
        # Device arrays created by the density-builder calls above are out of
        # scope before collection.  Release each participating device pool in
        # deterministic device order; pinned host staging uses one global pool.
        gc.collect()
        device_ids = sorted(
            {
                int(item.device.id)
                for item in vectors
                if isinstance(item, cp.ndarray)
            }
        )
        for device_id in device_ids:
            with cp.cuda.Device(device_id):
                cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


__all__ = ["CuPySymmetryDensityBuilder"]
