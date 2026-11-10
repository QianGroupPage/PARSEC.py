"""Kerker-type preconditioning of the potential residual of a 3-D periodic cell.

``Mixing_Preconditioner: kerker`` (PARSEC.py; Fortran PARSEC has no
preconditioner of its mixers, and the isolated path needs none).

Why.  PARSEC mixes the effective potential.  Near the fixed point ``V*``
the output potential of an input ``V_in = V* + e`` is, to first order,
``V_out = V* + K chi0 e`` with the static response ``chi0`` of the
orbitals and the kernel ``K = v_H + f_xc``, ``v_H(G) = 8 pi/G^2`` (Ry).
The residual is therefore

``R = V_out - V_in = -(1 - K chi0) e = -eps e``,

with the dielectric operator ``eps`` of potential perturbations, and the
Newton step is ``V* = V_in + eps^-1 R``.  Simple and Anderson mixing take
``V_in + alpha R`` and its history average; they converge in a number of
steps set by the spread of the eigenvalues of ``eps``.  For a metal
``chi0(G -> 0)`` tends to a constant ``-q0^2/(8 pi)`` (Thomas-Fermi), so
``eps(G) = 1 + q0^2/G^2`` grows without bound at the small wavevectors of a
large cell: the long-wavelength "charge sloshing" whose iteration count
grows with the cell edge.  With the model ``eps(G)`` the ideal step filters
the residual by

``P(G) = eps(G)^-1 = G^2/(G^2 + q0^2)``,

Kerker's preconditioner (G. P. Kerker, Phys. Rev. B 23, 3082 (1981)).
Because the model is diagonal in G, the filter of the potential residual is
the same function as Kerker's filter of the density residual of density
mixing.  An insulator or semiconductor screens with a finite macroscopic
dielectric constant ``eps_r``: ``eps(G) = (G^2 + q0^2)/(G^2 + q0^2/eps_r)``
tends to ``eps_r`` at ``G -> 0`` and to 1 at large G, and

``P(G) = (G^2 + q0^2/eps_r)/(G^2 + q0^2)``          (G != 0)

(the model of DFTK's ``DielectricMixing``, M. F. Herbst and A. Levitt,
J. Phys.: Condens. Matter 32, 085503 (2020)); ``eps_r = inf`` is Kerker's
form.  ``q0`` is ``Kerker_Wavevector`` (bohr^-1), ``eps_r``
``Kerker_Dielectric_Constant``.

G = 0.  The mean of ``V_in`` is a gauge of the periodic cell: a constant
shifts every eigenvalue and changes neither the orbitals, the occupations,
the density nor the total energy (the band energy and ``-int rho V_in``
shift by the same ``c N``).  ``V_out`` does not depend on it, so its
residual component relaxes with the mixer's own step: ``P(0) = 1``, the mean
of the residual passes unchanged.  (Kerker's ``P(0) = 0`` would freeze the
mean of ``V_in``, and the SRE, which contains it, would never fall below the
mean of the first residual.)

How it enters the mixer.  The mixer of ``Mixing_Method`` receives
``V_in`` and ``V_in + P(V_out - V_in)``: it mixes the residual ``P R`` of the
preconditioned fixed-point problem ``P (V_out[V] - V) = 0`` (left
preconditioning: the history least squares of Anderson and Broyden
minimize ``|sum c_i P R_i|``, the step is ``alpha`` times the averaged ``P R``).
``P`` is invertible (``P(G) > 0`` for every G), so the fixed point
``V_out = V_in`` and with it the converged result are those of the
unpreconditioned loop; the convergence test (SRE) still reads the residual
``V_out - V_in`` itself.  With two spin channels the filter acts on the
charge part of the residual only: ``R_s -> R_s + (P - 1)(R_up + R_down)/2``
(the Hartree term couples to the total density; the magnetization part is
local).

Grid.  The cell's points ``i*a_1/n_1 + j*a_2/n_2 + k*a_3/n_3`` (row-major,
:class:`parsec_python.Grid.pbc.PeriodicRealSpaceGrid`, also Fortran's
general grid) are transformed by a real 3-D FFT; the coefficient of the
integer index ``m`` belongs to ``G = m_1 b_1 + m_2 b_2 + m_3 b_3`` with the
reciprocal vectors ``b_i`` of the cell (an oblique cell through its
reciprocal metric).  A grid shift multiplies the coefficients by phases the
inverse transform removes.  At an even count the Nyquist index stands for
``+-n/2``; on an oblique cell the filter takes the mean of its values at the
aliases, so that it stays even in ``m`` and the filtered field real.  The
FFT runs with the array module of the residual: CuPy on the device for a
device array, NumPy on the host otherwise.  Slabs, wires and clusters are
refused: their grids are not periodic along every axis (a slab would need a
2-D form, which is not validated here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools

import numpy as np

from ..Grid.cell import reciprocal_lattice


def _array_module(values):
    """``cupy`` for a CuPy array, else ``numpy``."""

    module = type(values).__module__
    if module.startswith("cupy"):
        import cupy

        return cupy
    return np


def bulk_periodic_grid(grid) -> bool:
    """Whether ``grid`` holds every point of a cell periodic along its three axes."""

    if getattr(grid, "is_confined_periodic", False):
        return False
    if tuple(getattr(grid, "boundary_modes", ())) != ("wrap", "wrap", "wrap"):
        return False
    shape = getattr(grid, "points_per_axis", None)
    if shape is None or getattr(grid, "cell", None) is None:
        return False
    return int(np.prod(np.asarray(shape, dtype=np.int64))) == int(grid.size)


def _fft_indices(count: int, half: bool) -> np.ndarray:
    """Integer FFT indices of one axis (``fftfreq`` order; ``rfftfreq`` for the halved axis)."""

    if half:
        return np.arange(count // 2 + 1, dtype=np.float64)
    return np.rint(np.fft.fftfreq(count, d=1.0 / count))


def kerker_filter(
    lattice_vectors: np.ndarray,
    shape: tuple[int, int, int],
    wavevector: float,
    dielectric_constant: float = float("inf"),
) -> np.ndarray:
    """``P(G)`` on the ``rfftn`` coefficients of a ``shape`` grid of the cell.

    ``P(G) = (G^2 + q0^2/eps_r)/(G^2 + q0^2)`` for ``G != 0`` and ``P(0) = 1``
    (module docstring), ``q0 = wavevector`` (bohr^-1), ``eps_r =
    dielectric_constant`` (``inf``: Kerker's ``G^2/(G^2 + q0^2)``).
    ``lattice_vectors`` are the rows ``a_i`` (bohr) that the grid's axes span
    with ``shape[i]`` points.  Returns float64 of shape
    ``(n_1, n_2, n_3//2 + 1)``; on an oblique cell the Nyquist indices hold
    the mean over their aliases ``+-n/2``.
    """

    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    shape = tuple(int(value) for value in shape)
    if lattice.shape != (3, 3) or len(shape) != 3 or min(shape) < 1:
        raise ValueError("a Kerker filter needs three lattice vectors and three grid counts")
    q0_squared = float(wavevector) ** 2
    floor = 0.0 if np.isinf(dielectric_constant) else q0_squared / float(dielectric_constant)
    reciprocal = reciprocal_lattice(lattice)
    metric = reciprocal @ reciprocal.T
    axes = [_fft_indices(count, half=(axis == 2)) for axis, count in enumerate(shape)]
    orthogonal = np.allclose(metric - np.diag(np.diag(metric)), 0.0, atol=1.0e-12 * np.max(np.abs(metric)))
    # Index sets of each axis: the index itself, and on an oblique cell also
    # its alias -n/2 <-> +n/2 at an even count (the mean of the two filters).
    choices = []
    for axis, (count, index) in enumerate(zip(shape, axes)):
        variants = [index]
        if not orthogonal and count % 2 == 0:
            alias = index.copy()
            nyquist = np.abs(np.abs(index) - count // 2) < 0.5
            alias[nyquist] = -index[nyquist]
            variants.append(alias)
        choices.append(variants)
    total = np.zeros(tuple(len(axis) for axis in axes), dtype=np.float64)
    combinations = list(itertools.product(*choices))
    for m1, m2, m3 in combinations:
        m1g = m1[:, None, None]
        m2g = m2[None, :, None]
        m3g = m3[None, None, :]
        squared = (
            metric[0, 0] * m1g * m1g
            + metric[1, 1] * m2g * m2g
            + metric[2, 2] * m3g * m3g
            + 2.0 * metric[0, 1] * m1g * m2g
            + 2.0 * metric[0, 2] * m1g * m3g
            + 2.0 * metric[1, 2] * m2g * m3g
        )
        total += (squared + floor) / (squared + q0_squared)
    values = total if len(combinations) == 1 else total / float(len(combinations))
    values[0, 0, 0] = 1.0
    return values


@dataclass
class KerkerPreconditioner:
    """The filter ``P(G)`` of the residual on one periodic grid (module docstring).

    ``apply`` takes one value per grid point (row-major, ``shape``), or
    ``channels`` such vectors end to end (spin up first), and returns the
    filtered vector in the array module of its argument.
    """

    lattice_vectors: np.ndarray
    shape: tuple[int, int, int]
    wavevector: float
    dielectric_constant: float = float("inf")
    _filters: dict = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def for_grid(cls, grid, wavevector: float, dielectric_constant: float = float("inf")) -> "KerkerPreconditioner":
        """The preconditioner of a 3-D periodic grid; slabs, wires and clusters are refused."""

        if not bulk_periodic_grid(grid):
            raise NotImplementedError(
                "Mixing_Preconditioner: kerker is implemented for 3-D periodic cells "
                "(Boundary_Conditions: bulk) only; clusters, slabs and wires mix without it"
            )
        shape = tuple(int(value) for value in grid.points_per_axis)
        # The vectors the grid axes span: n_i steps of s_i are the lattice vector a_i.
        lattice = np.asarray(grid.step_vectors, dtype=np.float64) * np.asarray(shape, dtype=np.float64)[:, None]
        cell = np.asarray(grid.cell.lattice_vectors, dtype=np.float64)
        if not np.allclose(lattice, cell, rtol=0.0, atol=1.0e-8 * max(1.0, float(np.max(np.abs(cell))))):
            raise NotImplementedError(
                "Mixing_Preconditioner: kerker needs a grid whose axes span the lattice vectors of the cell"
            )
        return cls(lattice, shape, float(wavevector), float(dielectric_constant))

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def filter(self, xp=np):
        """``P`` on the ``rfftn`` coefficients, as an array of ``xp`` (cached)."""

        key = xp.__name__
        values = self._filters.get(key)
        if values is None:
            host = self._filters.get("numpy")
            if host is None:
                host = self._filters["numpy"] = kerker_filter(
                    self.lattice_vectors, self.shape, self.wavevector, self.dielectric_constant
                )
            values = host if xp is np else xp.asarray(host)
            self._filters[key] = values
        return values

    def _filtered(self, field, xp):
        transform = xp.fft.rfftn(field.reshape(self.shape), axes=(0, 1, 2))
        transform *= self.filter(xp)
        return xp.fft.irfftn(transform, s=self.shape, axes=(0, 1, 2)).reshape(-1)

    def apply(self, residual, channels: int = 1):
        """``P R`` of one field, or of the charge part of ``channels`` = 2 fields end to end."""

        xp = _array_module(residual)
        residual = xp.asarray(residual, dtype=xp.float64)
        if residual.shape != (channels * self.size,) or channels not in (1, 2):
            raise ValueError("the residual does not match the grid of the Kerker preconditioner")
        if channels == 1:
            return self._filtered(residual, xp)
        up, down = residual[: self.size], residual[self.size :]
        charge = 0.5 * (up + down)
        change = self._filtered(charge, xp) - charge
        return xp.concatenate((up + change, down + change))


@dataclass
class PreconditionedMixer:
    """A mixer of :mod:`parsec_python.Mixer` that mixes the residual ``P (V_out - V_in)``.

    ``mix(V_in, V_out)`` returns ``inner.mix(V_in, V_in + P(V_out - V_in))``
    (module docstring).  ``adapter`` is the SCF loop's scalar-field adapter
    of a symmetry-reduced loop (``to_full``/``from_full``, e.g. the identity
    sector of a periodic cell): the residual is filtered on the full grid and
    the shifted output handed back in the loop's own representation.
    Everything else (``reset``, the safeguard counters, ``settings``) is the
    inner mixer's.
    """

    inner: object
    preconditioner: KerkerPreconditioner
    adapter: object | None = None
    channels: int = 1

    def preconditioned_output(self, input_potential, output_potential):
        """``V_in + P(V_out - V_in)`` in the representation of the arguments."""

        if self.adapter is None:
            xp = _array_module(input_potential)
            given = xp.asarray(input_potential, dtype=xp.float64)
            residual = xp.asarray(output_potential, dtype=xp.float64) - given
            return given + self.preconditioner.apply(residual, self.channels)
        given = np.asarray(self.adapter.to_full(input_potential), dtype=np.float64)
        residual = np.asarray(self.adapter.to_full(output_potential), dtype=np.float64) - given
        return self.adapter.from_full(given + self.preconditioner.apply(residual, self.channels))

    def mix(self, input_potential, output_potential, *, iteration: int | None = None):
        return self.inner.mix(
            input_potential,
            self.preconditioned_output(input_potential, output_potential),
            iteration=iteration,
        )

    def reset(self) -> None:
        self.inner.reset()

    def __getattr__(self, name):
        # Only reached for names the wrapper lacks (safeguard counters,
        # settings, set_weights of an inner mixer).
        if name == "inner" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)


def preconditioned_mixer(mixer, system, *, adapter=None, channels: int = 1):
    """``mixer`` itself, or wrapped in :class:`PreconditionedMixer` for ``Mixing_Preconditioner: kerker``.

    ``system`` supplies the mixing settings (``system.input.mixing``) and the
    grid; ``adapter`` is the loop's scalar-field adapter (``None`` for full-grid
    fields), ``channels`` 1, or 2 for the joined spin channels of a full-grid
    loop.  A loop without the preconditioner keeps its mixer object, so its
    arithmetic is unchanged.
    """

    settings = system.input.mixing
    if getattr(settings, "preconditioner", "none") == "none":
        return mixer
    if channels == 2 and adapter is not None:
        raise NotImplementedError(
            "Mixing_Preconditioner: kerker is not implemented for spin channels in symmetry sectors"
        )
    preconditioner = KerkerPreconditioner.for_grid(
        system.grid, settings.kerker_wavevector, settings.kerker_dielectric_constant
    )
    return PreconditionedMixer(mixer, preconditioner, adapter=adapter, channels=channels)


__all__ = [
    "KerkerPreconditioner",
    "PreconditionedMixer",
    "bulk_periodic_grid",
    "kerker_filter",
    "preconditioned_mixer",
]
