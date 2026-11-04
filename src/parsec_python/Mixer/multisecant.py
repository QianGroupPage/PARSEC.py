"""PARSEC's multisecant potential mixers (``msecant1-3.f90``, ``qinv.f90``).

H.-r. Fang and Y. Saad, "Two classes of multisecant methods for nonlinear
acceleration", report umsi-2007-100, Minnesota Supercomputer Institute
(2007).  The mixers solve ``f(x) = x_out(x) - x = 0`` with an approximate
inverse Jacobian ``G`` (``x_new = x - G f``), starting from
``G_0 = -alpha I`` (``alpha = Mixing_Param``).  The secant pairs
``(Dx_i, Df_i)`` of successive iterates are grouped in groups of
``Mixing_Group_Size``; a completed group ``g`` enters ``G`` as ``E_g V_g^T``
and the open last group is updated at every iteration:

* Type-I (``msecant1``, Broyden's good method for group size 1):
  ``G+ = G + E (N^T DF)^+ N^T`` with ``N = G^T DX``;
* Type-II (``msecant2``, Broyden's bad method for group size 1):
  ``G+ = G + E (DF^T DF)^+ DF^T``;
* ``msecant3`` chooses between them at every iteration by the secant errors
  of the previous group (``msecant3.f90:241-299``).

``E = DX - G DF`` and ``(.)^+`` is the basic-solution inverse of
:func:`quasi_inverse`.  ``Mixing_Group_Size: 0`` takes every pair since the
last restart in one group, recomputed at every iteration.  The history
restarts when the residual grows, ``||f_old|| < Mixing_Restart_Factor ||f||``,
from the previous iterate (``msecant1.f90:128-152``); ``Restart_Mixing`` is not
read.  ``Mixing_EN_Like: 1`` alternates two stages in the manner of the
Eirola-Nevanlinna method: one iteration forms a secant pair from the stored
base iterate and updates ``G`` without storing a new base, the next forms no
pair, reuses ``G`` and stores its iterate as the next base.

The storage is that of the Fortran mixer type (``structures.F90:2096-2116``):
column ``k`` of ``dx`` holds ``E_k`` of a formed pair, the base iterate ``x``
in column ``m = nsecant``; ``df`` holds ``V`` of the completed groups, the raw
``Df`` of the open group and the base residual in column ``m``.  The columns
grow by ``Mixing_Memory_Expand_Factor`` from ``Memory_Param`` when full and
never shrink.

The Fortran routines use plain dot products on the irreducible wedge, whose
orbits all have the size of the group there; the formulas do not change under
a uniform scale of the inner product, so that equals the full-grid product.
``weights`` gives every inner product the form ``sum_i w_i a_i b_i``: none on
the full grid, the orbit multiplicities on a symmetry wedge whose orbits
differ in size, which is what reproduces the full-grid result there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from ..models import MixingSettings
from .broyden import _validated_weights

# qinv.f90:39: the numerical-rank tolerance |R_11| * 2.2204e-16.  The literal
# has no kind in the Fortran source, so it is a single-precision constant.
_QINV_EPSILON = float(np.float32(2.2204e-16))


def quasi_inverse(matrix: np.ndarray) -> np.ndarray:
    """Return the basic-solution inverse of a square matrix (``qinv.f90``).

    With the Householder QR factorization with column pivoting ``A P = Q R``
    (LAPACK ``dgeqp3``, as :func:`scipy.linalg.qr` with ``pivoting=True``
    calls it) and the numerical rank ``r``, the number of leading diagonal
    entries with ``|R_ii| > |R_11| eps``, the result is
    ``P [R_11^-1 Q_1^T; 0]``: ``A^+ b`` is the basic solution of the
    least-squares problem ``A x = b`` that is zero on the ``n - r`` columns
    the pivoting puts last.  For a nonsingular matrix it is the inverse.
    """

    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("quasi_inverse needs a square matrix")
    size = array.shape[0]
    inverse = np.zeros((size, size))
    if size == 0:
        return inverse
    q, r, pivots = linalg.qr(array, pivoting=True)
    tolerance = abs(r[0, 0]) * _QINV_EPSILON
    rank = 0
    while rank < size and abs(r[rank, rank]) > tolerance:
        rank += 1
    if rank:
        r_inverse = linalg.solve_triangular(r[:rank, :rank], np.eye(rank))
        inverse[pivots[:rank], :] = r_inverse @ q[:, :rank].T
    return inverse


_KINDS = {"msecant1": 1, "msecant2": 2, "msecant3": 3}


@dataclass
class MultisecantMixer:
    """Potential mixer matching ``msecant1/2/3.f90``.

    The method comes from ``settings.method`` (``msecant1``, ``msecant2`` or
    ``msecant3``).  ``mix(x_in, x_out, iteration=k)`` returns the next input
    potential; ``iteration == 1`` empties the secant history (``nsecant = 0``)
    as at the start of every PARSEC SCF loop.  The vectors are
    one-dimensional: the local potential on the grid, or ``[V_up; V_down]``
    of a spin-polarized calculation (``spinmix.f90``).

    ``restarts`` counts the restarts on residual growth and ``update_type``
    is the update of the last ``msecant3`` iteration (1 Type-I, 2 Type-II).
    """

    settings: MixingSettings = field(
        default_factory=lambda: MixingSettings(method="msecant1")
    )
    weights: np.ndarray | None = None
    restarts: int = field(default=0, init=False)
    update_type: int = field(default=0, init=False)
    _kind: int = field(default=1, init=False)
    _capacity: int = field(default=0, init=False)
    _nsecant: int = field(default=0, init=False)
    _en_stage: int = field(default=0, init=False)
    # Rows are the columns of the Fortran arrays: dx, df (capacity rows),
    # n (group size rows, or capacity for one group), and for msecant3 dx1,
    # dx2 and df2 (group size rows); x0 and f0 are the previous iterate.
    _dx: np.ndarray | None = field(default=None, init=False)
    _df: np.ndarray | None = field(default=None, init=False)
    _n: np.ndarray | None = field(default=None, init=False)
    _dx1: np.ndarray | None = field(default=None, init=False)
    _dx2: np.ndarray | None = field(default=None, init=False)
    _df2: np.ndarray | None = field(default=None, init=False)
    _x0: np.ndarray | None = field(default=None, init=False)
    _f0: np.ndarray | None = field(default=None, init=False)
    _calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.settings.method not in _KINDS:
            raise ValueError(
                f"MultisecantMixer needs an msecant method, not {self.settings.method!r}"
            )
        self.weights = _validated_weights(self.weights)
        kind = _KINDS[self.settings.method]
        if kind == 3 and self.settings.group_size == 0:
            # msecant3.f90:62-70: without groups there is no secant error to
            # compare, and the preferred update runs alone.
            kind = 1 if self.settings.preferred_type == 1 else 2
        self._kind = kind
        self.reset()

    def reset(self) -> None:
        self.restarts = 0
        self.update_type = 0
        self._capacity = self.settings.memory
        self._nsecant = 0
        self._en_stage = self.settings.en_like
        self._dx = self._df = self._n = None
        self._dx1 = self._dx2 = self._df2 = None
        self._x0 = self._f0 = None
        self._calls = 0

    # Inner products of rows: a @ (w b).
    def _dot(self, left: np.ndarray, right: np.ndarray) -> float:
        if self.weights is None:
            return float(np.dot(left, right))
        return float(np.dot(left, self.weights * right))

    def _products(self, rows: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return rows @ vector
        return rows @ (self.weights * vector)

    def _gram(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return left @ right.T
        return left @ (right * self.weights).T

    def _allocate(self, size: int) -> None:
        group_size = self.settings.group_size
        self._dx = np.zeros((self._capacity, size))
        self._df = np.zeros((self._capacity, size))
        if self._kind in (1, 3):
            rows = self._capacity if group_size == 0 else group_size
            self._n = np.zeros((rows, size))
        if self._kind == 3:
            self._dx1 = np.zeros((group_size, size))
            self._dx2 = np.zeros((group_size, size))
            self._df2 = np.zeros((group_size, size))

    def _expand(self) -> None:
        """``msecant1.f90:68-97``: grow the columns of dx, df (and N)."""

        old = self._capacity
        self._capacity = max(old + 1, int(old * self.settings.expand_factor))

        def grown(array: np.ndarray) -> np.ndarray:
            larger = np.zeros((self._capacity, array.shape[1]))
            larger[:old] = array[:old]
            return larger

        self._dx = grown(self._dx)
        self._df = grown(self._df)
        if self._kind == 1 and self.settings.group_size == 0:
            self._n = grown(self._n)

    def _advance_en_stage(self) -> None:
        if self._en_stage == 1:
            self._en_stage = 2
        elif self._en_stage == 2:
            self._en_stage = 1

    def _hybrid_update_type(
        self, open_df: np.ndarray, res: int, size: int, ngroup: int
    ) -> None:
        """``msecant3.f90:241-299``: Type-I or Type-II by the secant errors.

        With the Frobenius norms of the open group and of the previous
        completed group (``dx2``, ``df2``, its last ``res`` columns),
        ``||DF^T DF_prev||^2 ||N^T DF||^2`` larger than
        ``||DF^T DF||^2 ||DX^T DX_prev||^2`` selects Type-I, smaller Type-II,
        equal the preferred type.  An EN stage 1 keeps the type of the
        stage before it.
        """

        if self._en_stage != 1 and ngroup > 0:
            previous = slice(size - res, size)
            lhs = float(np.sum(self._gram(open_df, self._df2[previous]) ** 2)) * float(
                np.sum(self._gram(self._n[:res], open_df) ** 2)
            )
            rhs = float(np.sum(self._gram(open_df, open_df) ** 2)) * float(
                np.sum(self._gram(self._dx1[:res], self._dx2[previous]) ** 2)
            )
            if lhs > rhs:
                self.update_type = 1
            elif lhs < rhs:
                self.update_type = 2
            else:
                self.update_type = self.settings.preferred_type
        elif ngroup == 0:
            self.update_type = self.settings.preferred_type

    def mix(
        self,
        input_potential: np.ndarray,
        output_potential: np.ndarray,
        *,
        iteration: int | None = None,
    ) -> np.ndarray:
        x = np.array(input_potential, dtype=float)
        y = np.asarray(output_potential, dtype=float)
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError(
                "input and output potentials must be identical one-dimensional vectors"
            )
        if self.weights is not None and self.weights.size != x.size:
            raise ValueError("mixer weights do not match the mixed vector")
        if iteration is None:
            iteration = self._calls + 1
        if iteration < 1:
            raise ValueError("SCF iteration numbers start at one")
        self._calls = iteration
        settings = self.settings
        alpha = settings.parameter

        f = y - x
        if iteration == 1:
            self._nsecant = 0
        if self._dx is None or self._dx.shape[1] != x.size:
            self._allocate(x.size)
        m = self._nsecant
        if m == self._capacity:
            self._expand()
        size = m + 1 if settings.group_size == 0 else settings.group_size

        if m == 0:
            # msecant1.f90:110-121: store the iterate, mix linearly.
            self._dx[0] = x
            self._df[0] = f
            self._nsecant = 1
            if self._en_stage == 1:
                self._en_stage = 2
            return x + alpha * f

        # msecant1.f90:128-152: restart from the previous iterate when the
        # residual grew; x0 and f0 stay those of the previous iterate.
        if (m == 1 and self._en_stage == 1) or m >= 2:
            factor = settings.restart_factor
            if self._dot(self._f0, self._f0) < factor * factor * self._dot(f, f):
                self._dx[0] = self._x0
                self._df[0] = self._f0
                self._nsecant = 1
                if self._en_stage == 1:
                    self._en_stage = 2
                self.restarts += 1
                return self._x0 + alpha * self._f0

        # res is the size of the open last group, ngroup the number of
        # completed groups before it (msecant1.f90:154-157).
        res = (m + size - 1) % size + 1
        ngroup = (m - res) // size
        groups = [slice(g * size, (g + 1) * size) for g in range(ngroup)]
        last = m - 1
        if self._en_stage != 1:
            # msecant1.f90:158-181: the new pair and E_m = Dx - G Df.
            secant_x = x - self._dx[last]
            self._df[last] = f - self._df[last]
            secant_f = self._df[last]
            if self._kind == 3:
                self._dx1[res - 1] = secant_x
            new_e = secant_x + alpha * secant_f
            for columns in groups:
                new_e -= self._products(self._df[columns], secant_f) @ self._dx[columns]
            self._dx[last] = new_e

        # msecant1.f90:184-202: x + alpha f - sum_g E_g (V_g^T f).
        mixed = x + alpha * f
        for columns in groups:
            mixed -= self._products(self._df[columns], f) @ self._dx[columns]

        if self._en_stage != 1 and self._kind in (1, 3):
            # msecant1.f90:205-223: N_res = G^T Dx.
            n_column = -alpha * secant_x
            for columns in groups:
                n_column += self._products(self._dx[columns], secant_x) @ self._df[columns]
            self._n[res - 1] = n_column

        open_group = slice(ngroup * size, m)
        if self._en_stage == 1 and res == size:
            # msecant1.f90:224-239: the group was completed by the stage
            # before; df holds its V already.
            mixed -= self._products(self._df[open_group], f) @ self._dx[open_group]
        else:
            open_df = self._df[open_group]
            if self._kind == 3:
                self._hybrid_update_type(open_df, res, size, ngroup)
                update = 1 if self.update_type == 1 else 2
            else:
                update = self._kind
            if update == 1:
                # msecant1.f90:240-291: V = N (N^T DF)^+T.
                n_rows = self._n[:res]
                v = quasi_inverse(self._gram(n_rows, open_df)) @ n_rows
            else:
                # msecant2.f90:209-260: V = DF (DF^T DF)^+.
                v = quasi_inverse(self._gram(open_df, open_df)).T @ open_df
            mixed -= self._products(v, f) @ self._dx[open_group]
            if res == size:
                if self._kind == 3:
                    self._dx2 = self._dx1.copy()
                    self._df2 = open_df.copy()
                self._df[open_group] = v

        self._x0 = x
        self._f0 = f
        if self._en_stage != 2:
            self._dx[m] = x
            self._df[m] = f
            self._nsecant = m + 1
        self._advance_en_stage()
        return mixed


__all__ = ["MultisecantMixer", "quasi_inverse"]
