"""The FDET non-additive kinetic terms on a CUDA device (CuPy).

The counterpart of :mod:`parsec_python.Embedding.kinetic`: the same
functionals and pointwise formulas (:func:`parsec_python.Embedding.kinetic.kinetic_functional`
runs unchanged on CuPy arrays), with the gradient of ``gradmvs`` and the
Laplacian of the kinetic stencil applied by ``cupyx.scipy.ndimage.correlate1d``
to the zero-padded box of the cluster, as the host does with SciPy.  The
results equal the host ones to round-off (the order of the stencil sums and
of the reductions may differ).  The accelerated driver hands an instance to
its system (``kinetic_operators``) when an FDET run uses CuPy.
"""

from __future__ import annotations

import numpy as np

from parsec_python.Laplacian.finite_difference import second_derivative_coefficients
from parsec_python.V_xc.pbe import first_derivative_coefficients

from ..backends.cupy import require_cupy


class DeviceKineticOperators:
    """The grid operations of :func:`parsec_python.Embedding.kinetic.kinetic_functional` on one device.

    ``grid`` is an isolated cluster grid; ``device_id`` the device (``None``:
    the current one).  :meth:`to_host` brings a potential back.
    """

    def __init__(self, grid, device_id: int | None = None) -> None:
        cp, _ = require_cupy()
        from cupyx.scipy.ndimage import correlate1d

        self.cp = cp
        self.xp = cp
        self._correlate = correlate1d
        self.grid = grid
        self.device_id = device_id
        with self._device():
            local = np.asarray(grid.integer_coordinates - grid.index_min, dtype=np.int64)
            self._index = tuple(cp.asarray(local[:, axis]) for axis in range(3))
            self._shape = tuple(int(value) for value in grid.shape)
            order = grid.settings.expansion_order
            self._first = cp.asarray(first_derivative_coefficients(order) / grid.spacing)
            self._second = cp.asarray(second_derivative_coefficients(order) / grid.spacing**2)

    def _device(self):
        return self.cp.cuda.Device(self.device_id) if self.device_id is not None else self.cp.cuda.Device()

    def asarray(self, values):
        return self.cp.asarray(values, dtype=self.cp.float64)

    def _box(self, values):
        box = self.cp.zeros(self._shape, dtype=self.cp.float64)
        box[self._index] = values
        return box

    def gradient(self, values):
        box = self._box(values)
        return self.cp.stack(
            [self._correlate(box, self._first, axis=axis, mode="constant", cval=0.0)[self._index] for axis in range(3)]
        )

    def laplacian(self, values):
        box = self._box(values)
        total = self.cp.zeros(self.grid.size, dtype=self.cp.float64)
        for axis in range(3):
            total += self._correlate(box, self._second, axis=axis, mode="constant", cval=0.0)[self._index]
        return total

    def to_host(self, values) -> np.ndarray:
        return np.asarray(self.cp.asnumpy(values), dtype=np.float64)

    def run(self, function, *arguments, **keywords):
        """``function`` on the device of these operators."""

        with self._device():
            return function(*arguments, **keywords)


def device_kinetic_operators(reference, backend: str, device_id: int | None = None):
    """The device operators of an FDET system on CuPy, ``None`` otherwise.

    ``PARSEC_CUPY_FDET_KINETIC=host`` keeps the kinetic terms on the host.
    """

    import os

    field = getattr(reference, "embedding", None)
    if field is None or backend != "cupy" or field.functional.name == "none":
        return None
    setting = os.environ.get("PARSEC_CUPY_FDET_KINETIC", "device").strip().lower() or "device"
    if setting not in ("device", "host"):
        raise ValueError("PARSEC_CUPY_FDET_KINETIC must be device or host")
    if setting == "host":
        return None
    return DeviceKineticOperators(reference.grid, device_id)


__all__ = ["DeviceKineticOperators", "device_kinetic_operators"]
