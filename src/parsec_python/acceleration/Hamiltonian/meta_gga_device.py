"""The orbital term of a meta-GGA on the devices of a CuPy calculation.

The device counterpart of :class:`parsec_python.V_xc.meta_gga.KineticPotentialOperator`
and :func:`parsec_python.V_xc.meta_gga.kinetic_energy_density`: the term

``H_tau = sum_c G_c^T diag(vtau) G_c``   (Rydberg)

in the CuPy Hamiltonians the eigensolvers apply, and ``tau`` of the orbitals
where they are, on the full grid or in the symmetry sectors.

Symmetry sectors.  The sector groups are commuting involutions with real
one-dimensional characters (:mod:`acceleration.Symmetry.representations`).
Their 3x3 operations ``O_g`` commute and square to one, so they have common
real eigenvectors ``q_a`` (the Cartesian axes for reflections):
``O_g q_a = chi_a(g) q_a``.  The derivative ``D'_a = sum_c q_ac G_c`` along
``q_a`` then maps a function of sector ``Gamma`` to one of sector
``Gamma chi_a``, and with the sector bases ``U`` (orthonormal columns)

``B_{Gamma,a} = U_{Gamma chi_a}^T D'_a U_Gamma``

is a sparse matrix between the two sectors, assembled once from
representative rows as the sector Laplacian is
(:func:`reduce_between_sectors`).  Since ``sum_a D'_a^T V D'_a =
sum_c G_c^T V G_c`` and an invariant ``V`` is ``diag(v_w)`` in every sector,

``U_Gamma^T H_tau U_Gamma = sum_a B_{Gamma,a}^T diag(vtau on the orbits of Gamma chi_a) B_{Gamma,a}``

exactly, and ``tau`` of the sector orbitals ``x_n`` on orbit ``w`` is

``tau_w = (d/2h^3) sum_n f_n sum_a (B_{Gamma_n,a} x_n)_w^2 / |O_w|``:

no full-grid field is formed.  On the full grid ``B_a = G_a``.

The orbital form of the density gradient adds ``2 sum_c [W_c G_c + G_c^T W_c]``
of ``W = de/dg``.  ``W`` turns with the operations (``W(R r) = R W(r)``), so
its component ``W'_a`` along ``q_a`` has the character ``chi_a``; on the
orbits shared by ``Gamma`` and ``Gamma chi_a`` the products ``W'_a (D'_a q)``
and ``W'_a q`` lie in ``Gamma`` and ``Gamma chi_a``, and the sector term is
``P^T diag(W'_a) B + B^T diag(W'_a) P`` with ``P`` the restriction to the
shared orbits.  ``g'_a`` of sector orbitals on those orbits is
``(d/h^3) sum_n f_n 2 x_w (B x)_w / |O_w|``.  The fields leave and enter as
Cartesian components at the representatives (:class:`OrbitVectorFields`).

Each bound term reports Gershgorin's bound of its eigenvalues
(``spectral_margin``), which the filter's upper bound takes in
(:func:`parsec_python.acceleration.Eigensolvers.spectral_bounds.with_orbital_term`):
a narrow spike of ``vtau`` (low density) has localized eigenvalues that a
short Lanczos run does not see.

The products run in one CUDA kernel of a CSR matrix with a strided block
(:class:`DeviceCSR`), the transposes stored as CSR as well (deterministic
sums).  Memory: the matrices hold ``2 x 3 x Expansion_Order`` nonzeros per
grid point (12 bytes each), about 0.86 KiB per point at order 12; the
orbital form adds ``2 W'`` per derivative on the shared, target and source
rows and the int32 maps between them, at most 120 bytes per point
(``flux_device_bytes``).  The first-order part runs inside the two CSR
passes of each derivative (``DeviceCSR.product`` with ``secondary`` and
``add``).

Units Ry, bohr.  Nothing here prints or opens files.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import scipy.sparse as sp

from parsec_python.V_xc.meta_gga import grid_gradient

from ..backends.cupy import require_cupy


# ----- host: the derivatives between sectors ---------------------------------------


def cartesian_operations(reduction) -> np.ndarray:
    """The 3x3 matrices of the operations of a sector reduction, ``(order, 3, 3)``."""

    operations = getattr(reduction, "operations", None)
    if operations is not None:
        return np.asarray(operations, dtype=np.float64)
    signs = np.asarray(reduction.signs, dtype=np.float64)
    return np.stack([np.diag(row) for row in signs])


def derivative_directions(reduction) -> tuple[np.ndarray, np.ndarray]:
    """``(Q, chi)``: orthonormal common eigenvectors ``Q[:, a]`` and their characters ``chi[a, g]``.

    The Cartesian axes for a group of reflections; otherwise the eigenvectors
    of a fixed generic combination of the operations, each checked.
    """

    operations = cartesian_operations(reduction)
    diagonal = all(np.array_equal(operation, np.diag(np.diag(operation))) for operation in operations)
    if diagonal:
        directions = np.eye(3)
    else:
        weights = np.cos(1.0 + 0.7 * np.arange(operations.shape[0]))
        combination = np.tensordot(weights, operations, axes=1)
        combination = 0.5 * (combination + combination.T)
        _, directions = np.linalg.eigh(combination)
    characters = np.empty((3, operations.shape[0]))
    for axis in range(3):
        vector = directions[:, axis]
        for index, operation in enumerate(operations):
            image = operation @ vector
            if np.allclose(image, vector, atol=1e-12):
                characters[axis, index] = 1.0
            elif np.allclose(image, -vector, atol=1e-12):
                characters[axis, index] = -1.0
            else:
                raise RuntimeError("the operations of the sectors have no common eigenvectors")
    return directions, characters


def reduce_between_sectors(decomposition, matrix: sp.csr_matrix, source: int, target: int) -> sp.csr_matrix:
    """``U_target^T M U_source`` of an operator ``M`` that maps sector ``source`` into sector ``target``.

    For an admitted orbit ``w'`` of the target and ``w`` of the source,
    ``B[w', w] = sum_{j in O_w} M[rep(w'), j] phase_source(j) sqrt(m_w' / m_w)``:
    ``M U_source x`` lies in the target sector, whose coefficient on ``w'`` is
    ``sqrt(m_w')`` times its value at the representative row.
    """

    reduction = decomposition.reduction
    target_orbits = decomposition.sector_orbit_indices(target)
    selected = sp.csr_matrix(matrix)[reduction.representative_rows[target_orbits], :].tocoo()
    column_orbits = reduction.full_to_wedge[selected.col]
    columns = decomposition.orbit_to_sector[source, column_orbits]
    phases = decomposition.phases[source, selected.col]
    keep = (columns >= 0) & (phases != 0)
    multiplicities = np.asarray(reduction.multiplicities, dtype=np.float64)
    data = (
        selected.data[keep]
        * phases[keep]
        * np.sqrt(multiplicities[target_orbits[selected.row[keep]]] / multiplicities[column_orbits[keep]])
    )
    reduced = sp.coo_matrix(
        (data, (selected.row[keep], columns[keep])),
        shape=(target_orbits.size, decomposition.sector_size(source)),
    ).tocsr()
    reduced.sum_duplicates()
    reduced.eliminate_zeros()
    reduced.sort_indices()
    return reduced


@dataclass(frozen=True)
class SectorDerivative:
    """One ``B_{Gamma,a}``: the host CSR matrix and its target sector (``None`` on the full grid)."""

    matrix: sp.csr_matrix
    target: int | None


def sector_derivatives(grid, decomposition, representations=None, *, check: bool = True) -> dict:
    """``{representation: (SectorDerivative x 3)}`` of the sectors (``{None: ...}`` without sectors)."""

    gradient = grid_gradient(grid).matrices
    if decomposition is None:
        return {None: tuple(SectorDerivative(sp.csr_matrix(matrix), None) for matrix in gradient)}
    directions, characters = derivative_directions(decomposition.reduction)
    rotated = []
    for axis in range(3):
        matrix = None
        for component in range(3):
            weight = float(directions[component, axis])
            if weight == 0.0:
                continue
            term = weight * gradient[component]
            matrix = term if matrix is None else matrix + term
        rotated.append(sp.csr_matrix(matrix))
    table = np.asarray(decomposition.characters, dtype=np.float64)
    wanted = range(decomposition.representation_count) if representations is None else representations
    rng = np.random.default_rng(20261107)
    result = {}
    for representation in wanted:
        pieces = []
        for axis in range(3):
            character = table[representation] * characters[axis]
            matches = np.flatnonzero(np.all(table == character[None, :], axis=1))
            if matches.size != 1:
                raise RuntimeError("no sector takes the derivative of this sector")
            target = int(matches[0])
            matrix = reduce_between_sectors(decomposition, rotated[axis], representation, target)
            if check:
                vector = rng.standard_normal(decomposition.sector_size(representation))
                full = rotated[axis] @ decomposition.expand_vector(vector, representation)
                expected = decomposition.reduce_vector(full, target)
                error = float(np.max(np.abs(matrix @ vector - expected), initial=0.0))
                if error > 1e-10 * max(1.0, float(np.max(np.abs(expected), initial=0.0))):
                    raise RuntimeError(f"the reduced derivative of sector {representation} is not exact ({error:.2e})")
            pieces.append(SectorDerivative(matrix, target))
        result[int(representation)] = tuple(pieces)
    return result


class OrbitVectorFields:
    """Vector fields that turn with the operations of a sector reduction, kept at the orbit representatives.

    The orbital density gradient ``g`` of orbitals of one-dimensional
    characters and ``W = de/dg`` satisfy ``g(R r) = R g(r)`` for every
    operation ``R`` of the group (``q(R r) = chi q(r)`` and the stencil is
    mapped onto itself).  The three Cartesian components at the
    representative point of every orbit therefore hold the whole field: the
    layout of ``g`` and ``W`` in a meta-GGA loop that keeps its fields on
    orbits.  :meth:`expand` builds the full-grid components, :meth:`reduce`
    takes them at the representatives.  Where two operations carry a
    representative onto the same row the field is invariant under their
    quotient there, and either gives the same value.
    """

    def __init__(self, grid, reduction) -> None:
        from ..Symmetry.axis_reflection import _grid_row_mapping_operation

        operations = cartesian_operations(reduction)
        rows = np.asarray(reduction.representative_rows, dtype=np.int64)
        full_to_wedge = np.asarray(reduction.full_to_wedge, dtype=np.int64)
        operation_of_row = np.full(full_to_wedge.size, -1, dtype=np.int16)
        order = sorted(range(len(operations)), key=lambda index: (not np.allclose(operations[index], np.eye(3)), index))
        for index in order:
            images = _grid_row_mapping_operation(grid, np.rint(operations[index]).astype(np.int8), 1.0e-10, selected_rows=rows)
            if images is None or not np.array_equal(full_to_wedge[images], np.arange(rows.size)):
                raise RuntimeError("the symmetry operations do not map the orbit representatives onto their orbits")
            images = np.asarray(images, dtype=np.int64)
            fresh = images[operation_of_row[images] < 0]
            operation_of_row[fresh] = index
        if np.any(operation_of_row < 0):
            raise RuntimeError("a grid row is the image of no orbit representative")
        self.operations = operations
        self.operation_of_row = operation_of_row
        self.full_to_wedge = full_to_wedge
        self.representative_rows = rows

    def expand(self, components) -> np.ndarray:
        """The full-grid Cartesian components ``(3, n)`` of the representatives' ``(3, wedge)``."""

        values = np.asarray(components, dtype=np.float64)
        at_rows = values[:, self.full_to_wedge]
        result = np.empty_like(at_rows)
        for index, operation in enumerate(self.operations):
            selected = self.operation_of_row == index
            if np.any(selected):
                result[:, selected] = operation @ at_rows[:, selected]
        return result

    def reduce(self, components) -> np.ndarray:
        """The components ``(3, wedge)`` at the representatives of full-grid ``(3, n)``."""

        return np.ascontiguousarray(np.asarray(components, dtype=np.float64)[:, self.representative_rows])


_ORBIT_VECTOR_FIELDS: dict = {}


def orbit_vector_fields(grid, reduction) -> OrbitVectorFields:
    """:class:`OrbitVectorFields` of a grid and reduction (cached for the last few)."""

    key = (id(grid), id(reduction))
    cached = _ORBIT_VECTOR_FIELDS.get(key)
    if cached is not None and cached[0] is grid and cached[1] is reduction:
        return cached[2]
    fields = OrbitVectorFields(grid, reduction)
    if len(_ORBIT_VECTOR_FIELDS) >= 4:
        _ORBIT_VECTOR_FIELDS.pop(next(iter(_ORBIT_VECTOR_FIELDS)))
    _ORBIT_VECTOR_FIELDS[key] = (grid, reduction, fields)
    return fields


# ----- device: CSR products ----------------------------------------------------------

_KERNEL_SOURCE = r"""
extern "C" __global__
void csr_block_product(
    const long long rows,
    const long long columns,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const double* __restrict__ values,
    const double* __restrict__ block,
    const long long block_row,
    const long long block_column,
    const double* __restrict__ row_scale,
    const int scaled,
    double* __restrict__ output,
    const long long output_row,
    const long long output_column,
    const int accumulate,
    const int column_fastest,
    const int absolute,
    double* __restrict__ secondary,
    const long long secondary_row,
    const long long secondary_column,
    const int has_secondary,
    const int* __restrict__ add_index,
    const double* __restrict__ add_weight,
    const double* __restrict__ add_block,
    const long long add_row,
    const long long add_column,
    const int has_add,
    const int add_identity
) {
    const long long index = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (index >= rows * columns) {
        return;
    }
    long long row;
    long long column;
    if (column_fastest) {
        row = index / columns;
        column = index - row * columns;
    } else {
        column = index / rows;
        row = index - column * rows;
    }
    double sum = 0.0;
    const int end = indptr[row + 1];
    for (int entry = indptr[row]; entry < end; ++entry) {
        const double value = absolute ? fabs(values[entry]) : values[entry];
        sum += value * block[static_cast<long long>(indices[entry]) * block_row + column * block_column];
    }
    if (has_secondary) {
        secondary[row * secondary_row + column * secondary_column] = sum;
    }
    if (scaled) {
        sum *= row_scale[row];
    }
    if (has_add) {
        const long long source = add_identity ? row : static_cast<long long>(add_index[row]);
        if (source >= 0) {
            sum += add_weight[row] * add_block[source * add_row + column * add_column];
        }
    }
    double* target = output + row * output_row + column * output_column;
    if (accumulate) {
        *target += sum;
    } else {
        *target = sum;
    }
}
"""
_KERNELS: dict[int, Any] = {}


def _kernel(cp):
    device = int(cp.cuda.Device().id)
    kernel = _KERNELS.get(device)
    if kernel is None:
        kernel = cp.RawKernel(_KERNEL_SOURCE, "csr_block_product", options=("--std=c++11", "--fmad=false"))
        kernel.compile()
        _KERNELS[device] = kernel
    return kernel


class DeviceCSR:
    """A float64 CSR matrix on the current device (int32 indices) with a strided block product."""

    def __init__(self, cp, matrix: sp.csr_matrix) -> None:
        matrix = sp.csr_matrix(matrix, dtype=np.float64)
        matrix.sort_indices()
        if matrix.nnz >= 2**31:
            raise ValueError("a meta-GGA derivative matrix has too many entries for int32 indices")
        self.cp = cp
        self.shape = tuple(int(value) for value in matrix.shape)
        self.indptr = cp.asarray(matrix.indptr.astype(np.int32))
        self.indices = cp.asarray(matrix.indices.astype(np.int32))
        self.values = cp.asarray(matrix.data.astype(np.float64))
        self.nbytes = int(self.indptr.nbytes + self.indices.nbytes + self.values.nbytes)

    def product(
        self,
        block,
        output=None,
        *,
        row_scale=None,
        accumulate: bool = False,
        absolute: bool = False,
        secondary=None,
        add=None,
    ):
        """``output (+)= diag(row_scale) M block [+ diag(w) A[index]]`` for a block of shape ``(columns of M, k)``.

        ``|M|`` with ``absolute``.  ``secondary`` receives ``M block`` before
        the scale; ``add = (index, weight, other)`` adds ``weight[r] *
        other[index[r], :]`` to row ``r`` (``index`` ``None``: the same row;
        ``index[r] < 0``: nothing), in the same pass.
        """

        cp = self.cp
        if block.ndim != 2 or block.shape[0] != self.shape[1]:
            raise ValueError("the block does not match the matrix")
        rows, count = self.shape[0], int(block.shape[1])
        if output is None:
            output = cp.empty((rows, count), dtype=cp.float64, order="F" if block.flags.f_contiguous and not block.flags.c_contiguous else "C")
            accumulate = False
        itemsize = 8
        column_fastest = int(output.strides[1] // itemsize == 1 and count > 1)
        total = rows * count
        if total == 0:
            return output
        threads = 256
        blocks = (total + threads - 1) // threads
        scale = self.values if row_scale is None else row_scale
        dummy_index = self.indices
        if secondary is not None and secondary.shape != (rows, count):
            raise ValueError("the secondary output does not match the product")
        add_index, add_weight, add_block = (None, None, None) if add is None else add
        if add_block is not None and add_block.shape[1] != count:
            raise ValueError("the added block does not match the product")
        secondary_array = output if secondary is None else secondary
        added = self.values if add_block is None else add_block
        _kernel(cp)(
            (blocks,),
            (threads,),
            (
                np.int64(rows),
                np.int64(count),
                self.indptr,
                self.indices,
                self.values,
                block,
                np.int64(block.strides[0] // itemsize),
                np.int64(block.strides[1] // itemsize),
                scale,
                np.int32(row_scale is not None),
                output,
                np.int64(output.strides[0] // itemsize),
                np.int64(output.strides[1] // itemsize),
                np.int32(bool(accumulate)),
                np.int32(column_fastest),
                np.int32(bool(absolute)),
                secondary_array,
                np.int64(secondary_array.strides[0] // itemsize),
                np.int64(secondary_array.strides[1] // itemsize if secondary_array.ndim > 1 else 0),
                np.int32(secondary is not None),
                dummy_index if add_index is None else add_index,
                self.values if add_weight is None else add_weight,
                added,
                np.int64(added.strides[0] // itemsize),
                np.int64(added.strides[1] // itemsize if added.ndim > 1 else 0),
                np.int32(add is not None),
                np.int32(add is not None and add_index is None),
            ),
        )
        return output


class DeviceKineticTerm:
    """``sum_a B_a^T diag(v_a) B_a`` of one device operator (full grid or one sector).

    ``derivatives`` are the :class:`SectorDerivative` of the operator,
    ``target_orbits`` per derivative the wedge orbits of its target sector
    (``None`` on the full grid, where ``vtau`` is a full-grid vector) and
    ``source_orbits`` those of the operator's own sector.

    In the orbital form of the density gradient the term also holds
    ``2 sum_a [W'_a D'_a + D'_a^T W'_a]`` of ``W'_a``, the component of
    ``W = de/dg`` along the derivative direction ``a``, a field of character
    ``chi_a``: on the orbits shared by the sector and its target,
    ``W'_a (D'_a q)`` has the sector's character and ``W'_a q`` the target's,
    so in the sector bases the term is ``P^T diag(W'_a) B_a + B_a^T diag(W'_a) P``
    with ``P`` the restriction of the sector's rows to the shared orbits.
    """

    def __init__(self, cp, derivatives, target_orbits, device_id: int, source_orbits=None) -> None:
        self.cp = cp
        self.device_id = int(device_id)
        with cp.cuda.Device(self.device_id):
            self.forward = [DeviceCSR(cp, item.matrix) for item in derivatives]
            self.backward = [DeviceCSR(cp, sp.csr_matrix(item.matrix.T)) for item in derivatives]
            self.orbits = [None if orbits is None else cp.asarray(np.asarray(orbits, dtype=np.int64)) for orbits in target_orbits]
        self.potentials: list = [None] * len(self.forward)
        # 2 W'_a on the shared rows (the orbital form), None otherwise.
        self.doubled_fluxes: list = [None] * len(self.forward)
        self.rows = int(self.forward[0].shape[1])
        # The target sector of each derivative (None on the full grid).
        self.targets = [item.target for item in derivatives]
        self._target_orbits = [None if orbits is None else np.asarray(orbits, dtype=np.int64) for orbits in target_orbits]
        self._source_orbits = None if source_orbits is None else np.asarray(source_orbits, dtype=np.int64)
        self._shared = None
        # Per derivative the wedge orbits of the shared rows (None on the full grid).
        self.shared_orbits = None
        # The fused orbital form (apply): per derivative the source row of every
        # target row and the target row of every source row (-1: not shared),
        # and 2 W'_a on those rows.
        self._row_maps = None
        self.forward_weights: list = [None] * len(self.forward)
        self.backward_weights: list = [None] * len(self.forward)
        # A bound of the eigenvalues of the bound term (Ry, _row_sum_bound) and its row sums.
        self.spectral_margin = 0.0
        self.row_sums = None

    def row_maps(self) -> list:
        """Per derivative ``(source row of each target row, target row of each source row)`` (int32, -1 if none)."""

        if self._row_maps is None:
            cp = self.cp
            shared = self.shared_rows()
            maps = []
            with cp.cuda.Device(self.device_id):
                for index, rows in enumerate(shared):
                    if rows is None:
                        maps.append(None)
                        continue
                    target_rows, source_rows = rows
                    source_of_target = cp.full(self.forward[index].shape[0], -1, dtype=cp.int32)
                    source_of_target[target_rows] = source_rows
                    target_of_source = cp.full(self.forward[index].shape[1], -1, dtype=cp.int32)
                    target_of_source[source_rows] = target_rows
                    maps.append((source_of_target, target_of_source))
            self._row_maps = maps
        return self._row_maps

    def shared_rows(self) -> list:
        """Per derivative ``(target rows, source rows)`` of the orbits of both sectors (``None``: the same rows)."""

        if self._shared is None:
            cp = self.cp
            shared, orbits = [], []
            with cp.cuda.Device(self.device_id):
                for target in self._target_orbits:
                    if target is None or self._source_orbits is None:
                        shared.append(None)
                        orbits.append(None)
                        continue
                    common, target_rows, source_rows = np.intersect1d(
                        target, self._source_orbits, assume_unique=True, return_indices=True
                    )
                    shared.append((cp.asarray(target_rows.astype(np.int32)), cp.asarray(source_rows.astype(np.int32))))
                    orbits.append(common)
            self._shared = shared
            self.shared_orbits = orbits
        return self._shared

    @property
    def device_bytes(self) -> int:
        total = sum(matrix.nbytes for matrix in self.forward + self.backward)
        total += sum(0 if orbits is None else int(orbits.nbytes) for orbits in self.orbits)
        total += sum(8 * matrix.shape[0] for matrix in self.forward)
        return int(total)

    @property
    def block_arrays(self) -> int:
        """Row-sized arrays per column that :meth:`apply` allocates (result, scaled block, orbital form's derivative)."""

        largest = max(int(matrix.shape[0]) for matrix in self.forward)
        ratio = max(1.0, largest / max(1, self.rows))
        orbital = any(flux is not None for flux in self.doubled_fluxes)
        return int(np.ceil(1.0 + ratio * (2.0 if orbital else 1.0)))

    @property
    def flux_device_bytes(self) -> int:
        """What the orbital form adds: ``2 W'_a`` on the target and source rows, the shared rows and the row maps."""

        total = 0
        for index, matrix in enumerate(self.forward):
            targets, sources = int(matrix.shape[0]), int(matrix.shape[1])
            # 2 W'_a (shared rows), its forward and backward weights
            total += 8 * (targets + sources + targets)
            if self._target_orbits[index] is not None and self._source_orbits is not None:
                shared = int(np.intersect1d(self._target_orbits[index], self._source_orbits, assume_unique=True).size)
                # the shared-row pairs and the two row maps (int32)
                total += 8 * shared + 4 * (targets + sources)
        return int(total)

    def bind(self, potential, fluxes=None) -> None:
        """``vtau`` on this device (full grid, or wedge orbit values in the sectors) and the ``W'_a`` of the orbital form.

        ``fluxes`` are the three components along the derivative directions,
        in the layout of ``potential``; ``None`` leaves the first-order term out.
        """

        cp = self.cp
        with cp.cuda.Device(self.device_id):
            values = cp.asarray(potential, dtype=cp.float64)
            for index, orbits in enumerate(self.orbits):
                selected = values if orbits is None else values[orbits]
                if selected.shape != (self.forward[index].shape[0],):
                    raise ValueError("vtau does not match the derivative of the operator")
                self.potentials[index] = cp.ascontiguousarray(selected)
            if fluxes is None:
                self.doubled_fluxes = [None] * len(self.forward)
                self.forward_weights = [None] * len(self.forward)
                self.backward_weights = [None] * len(self.forward)
                self.spectral_margin = self._row_sum_bound()
                return
            if len(fluxes) != len(self.forward):
                raise ValueError("W needs one component per derivative direction")
            shared = self.shared_rows()
            for index, orbits in enumerate(self.orbits):
                component = cp.asarray(fluxes[index], dtype=cp.float64)
                selected = component if orbits is None else component[orbits]
                if selected.shape != (self.forward[index].shape[0],):
                    raise ValueError("W does not match the derivative of the operator")
                if shared[index] is not None:
                    selected = selected[shared[index][0]]
                self.doubled_fluxes[index] = cp.ascontiguousarray(2.0 * selected)
            maps = self.row_maps()
            for index, doubled in enumerate(self.doubled_fluxes):
                if shared[index] is None:
                    self.forward_weights[index] = doubled
                    self.backward_weights[index] = doubled
                    continue
                target_rows, source_rows = shared[index]
                forward = cp.zeros(self.forward[index].shape[0], dtype=cp.float64)
                forward[target_rows] = doubled
                backward = cp.zeros(self.forward[index].shape[1], dtype=cp.float64)
                backward[source_rows] = doubled
                self.forward_weights[index] = forward
                self.backward_weights[index] = backward
            self.spectral_margin = self._row_sum_bound()

    def _row_sum_bound(self) -> float:
        """Gershgorin's bound of the term: the largest row sum of ``|M|`` (a bound of ``|lambda|`` of every eigenvalue).

        Row sums of ``sum_a |B_a|^T (|v_a| |B_a| 1) + 2 P^T (|W'_a| |B_a| 1) + 2 |B_a|^T (|W'_a| P 1)``,
        which bound those of ``|M|``.  Local: a narrow spike of ``vtau``
        (low density, where ``alpha`` of SCAN is a ratio of small numbers)
        gives its own localized eigenvalues, which a short Lanczos run from
        a random vector does not see, and this bound does.
        """

        cp = self.cp
        total = None
        for index, (forward, backward, potential) in enumerate(zip(self.forward, self.backward, self.potentials)):
            ones = cp.ones((forward.shape[1], 1), dtype=cp.float64)
            row_sums = forward.product(ones, absolute=True)[:, 0]
            weights = cp.abs(potential) * row_sums
            doubled = self.doubled_fluxes[index]
            shared = None if doubled is None else self._shared[index]
            if doubled is not None:
                flux = cp.abs(doubled)
                if shared is None:
                    weights = weights + flux
                else:
                    weights[shared[0]] += flux
            contribution = backward.product(weights[:, None], absolute=True)[:, 0]
            if doubled is not None:
                if shared is None:
                    contribution += flux * row_sums
                else:
                    contribution[shared[1]] += flux * row_sums[shared[0]]
            total = contribution if total is None else total + contribution
        # The start vector of the targeted Lanczos bound (spectral_bounds.with_orbital_term).
        self.row_sums = total
        return float(total.max()) if total is not None and total.size else 0.0

    @property
    def bound(self) -> bool:
        return self.potentials[0] is not None

    def apply(self, block):
        """``H_tau X`` (a new array of the layout of ``X``)."""

        cp = self.cp
        was_vector = block.ndim == 1
        columns = block[:, None] if was_vector else block
        result = None
        for index, (forward, backward, potential) in enumerate(zip(self.forward, self.backward, self.potentials)):
            doubled = self.doubled_fluxes[index]
            if doubled is None:
                scaled = forward.product(columns, row_scale=potential)
                if result is None:
                    result = backward.product(scaled)
                else:
                    backward.product(scaled, result, accumulate=True)
                continue
            # One pass each way: s = vtau (B x) + 2W' P x (keeping B x), then
            # B^T s + P^T 2W' (B x) into the result.
            maps = self._row_maps[index] if self._row_maps is not None else None
            derivative = cp.empty((forward.shape[0], columns.shape[1]), dtype=cp.float64)
            scaled = forward.product(
                columns,
                row_scale=potential,
                secondary=derivative,
                add=(None if maps is None else maps[0], self.forward_weights[index], columns),
            )
            add = (None if maps is None else maps[1], self.backward_weights[index], derivative)
            if result is None:
                result = backward.product(scaled, add=add)
            else:
                backward.product(scaled, result, accumulate=True, add=add)
        return result[:, 0] if was_vector else result

    def kinetic_sums(self, block, occupations, gradient: bool = False):
        """Per derivative, ``sum_n f_n (B_a x_n)^2`` on the rows of its target (device arrays).

        With ``gradient`` also ``sum_n f_n x_n (B_a x_n)`` on the shared rows
        (:attr:`shared_orbits`): ``(squares, products)``.
        """

        cp = self.cp
        weights = cp.asarray(np.asarray(occupations, dtype=np.float64))
        shared = self.shared_rows() if gradient else None
        sums, products = [], []
        for index, forward in enumerate(self.forward):
            values = forward.product(block)
            sums.append((values * values) @ weights)
            if gradient:
                rows = shared[index]
                if rows is None:
                    products.append((block * values) @ weights)
                else:
                    target_rows, source_rows = rows
                    products.append((block[source_rows] * values[target_rows]) @ weights)
        return (sums, products) if gradient else sums


# ----- the system's piece --------------------------------------------------------------


def _device_of(array) -> int | None:
    device = getattr(array, "device", None)
    return None if device is None else int(getattr(device, "id", device))


class DeviceMetaGGA:
    """The meta-GGA orbital term in the device Hamiltonians of a prepared CuPy system.

    ``backend`` is the CuPy backend (its full-grid operator), ``sectors`` the
    symmetry eigensolver or ``None``, ``reducer`` the orbit-field adapter of
    the sectors (``SymmetrySCFReducer``) or ``None``.  The terms are built on
    the first :meth:`bind` with a potential (or by :meth:`prepare`).

    The vector fields of the orbital form (``g`` and ``W``) enter and leave
    as their three Cartesian components in the layout of the scalar fields
    (values at the orbit representatives in the sectors,
    :class:`OrbitVectorFields`); inside, the sectors take their components
    along the derivative directions.
    """

    def __init__(self, *, backend, sectors, reducer, grid) -> None:
        self.backend = backend
        self.sectors = sectors
        self.reducer = reducer
        self.grid = grid
        self.terms: dict = {}
        self.directions = np.eye(3)
        self.statistics = dict(build_seconds=0.0, bind_seconds=0.0, tau_seconds=0.0, tau_calls=0, binds=0)
        self._bound_none = True

    def operators(self) -> dict:
        """The device operators that hold the term: representation (``None``: full grid) -> operator."""

        if self.sectors is not None:
            return {
                representation: operator
                for representation, operator in enumerate(self.sectors._operators)
                if operator is not None
            }
        operator = getattr(self.backend, "device_operator", None)
        if operator is None:
            operator = self.backend.ensure_device_operator()
        return {None: operator}

    def _check_route(self) -> None:
        sectors = self.sectors
        if sectors is None:
            return
        if getattr(sectors, "mpi_context", None) is not None:
            raise NotImplementedError("a meta-GGA is not implemented for the symmetry sectors of several MPI ranks")
        groups = getattr(sectors, "sector_device_groups", None) or {}
        if groups:
            raise NotImplementedError(
                "a meta-GGA is not implemented for a sector basis shared by several devices "
                "(sectors fewer than devices); run with as many devices as sectors or fewer"
            )

    def prepare(self) -> None:
        if self.terms:
            return
        self._check_route()
        cp, _ = require_cupy()
        started = perf_counter()
        operators = self.operators()
        decomposition = None if self.sectors is None else self.sectors.decomposition
        wanted = [representation for representation in operators if representation is not None]
        derivatives = sector_derivatives(self.grid, decomposition, wanted if decomposition is not None else None)
        if decomposition is not None:
            self.directions = derivative_directions(decomposition.reduction)[0]
        for representation, operator in operators.items():
            pieces = derivatives[representation]
            orbits = [
                None if item.target is None else decomposition.sector_orbit_indices(item.target) for item in pieces
            ]
            source = None if representation is None else decomposition.sector_orbit_indices(representation)
            device_id = int(operator.effective_potential.device.id)
            term = DeviceKineticTerm(cp, pieces, orbits, device_id, source_orbits=source)
            self.terms[representation] = term
            # The operator keeps the Lanczos bound of its passes without the
            # term (Eigensolvers.spectral_bounds.with_orbital_term).
            if not hasattr(operator, "bare_upper_bound"):
                operator.bare_upper_bound = None
        self.statistics["build_seconds"] += perf_counter() - started

    @property
    def device_bytes(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for term in self.terms.values():
            counts[term.device_id] = counts.get(term.device_id, 0) + term.device_bytes
        return counts

    @property
    def flux_device_bytes(self) -> dict[int, int]:
        """The device memory the orbital form adds to :attr:`device_bytes` (``W'`` and the shared-row maps)."""

        counts: dict[int, int] = {}
        for term in self.terms.values():
            counts[term.device_id] = counts.get(term.device_id, 0) + term.flux_device_bytes
        return counts

    def _values(self, field) -> np.ndarray:
        if self.reducer is not None:
            return np.ascontiguousarray(self.reducer.wedge_values(field), dtype=np.float64)
        return np.ascontiguousarray(field, dtype=np.float64)

    def _layout(self, values):
        return self.reducer.field(values) if self.reducer is not None else values

    def bind(self, kinetic_potential, gradient_flux=None) -> None:
        """Write ``vtau`` (and ``W`` of the orbital form) into the operators; ``None`` removes the term.

        ``vtau`` is a full-grid vector or an orbit field of the sectors,
        ``gradient_flux`` the three Cartesian components of ``W`` in the
        same layout.
        """

        if kinetic_potential is None:
            if gradient_flux is not None:
                raise ValueError("W of the orbital form comes with vtau")
            if not self._bound_none:
                for operator in self.operators().values():
                    operator.bind_kinetic_term(None)
                self._bound_none = True
            return
        self.prepare()
        cp, _ = require_cupy()
        started = perf_counter()
        values = self._values(kinetic_potential)
        rotated = None
        if gradient_flux is not None:
            components = np.stack([self._values(component) for component in gradient_flux])
            if components.shape != (3, values.size):
                raise ValueError("W needs three components in the layout of vtau")
            # Along the derivative directions: W'_a = sum_c Q[c, a] W_c.
            rotated = np.ascontiguousarray(self.directions.T @ components)
        uploaded: dict[int, Any] = {}
        for representation, operator in self.operators().items():
            term = self.terms[representation]
            if term.device_id not in uploaded:
                with cp.cuda.Device(term.device_id):
                    uploaded[term.device_id] = (
                        cp.asarray(values),
                        None if rotated is None else [cp.asarray(row) for row in rotated],
                    )
            potential, fluxes = uploaded[term.device_id]
            term.bind(potential, fluxes)
            operator.bind_kinetic_term(term)
        self._bound_none = False
        self.statistics["bind_seconds"] += perf_counter() - started
        self.statistics["binds"] += 1

    def tau(self, wavefunctions, occupations, degeneracy: float, gradient: bool = False):
        """``tau`` (Hartree) of the eigensolver's orbitals: a host full-grid vector, or orbit field of the sectors.

        With ``gradient`` also the orbital form of the density gradient,
        ``(tau, (g_x, g_y, g_z))``, ``g = (d/h^3) sum_n f_n 2 q_n (G q_n)``
        (:func:`parsec_python.V_xc.meta_gga.orbital_density_gradient`), its
        Cartesian components in the layout of ``tau``.
        """

        self.prepare()
        cp, _ = require_cupy()
        started = perf_counter()
        factor = 0.5 * float(degeneracy) / float(self.grid.volume_element)
        gradient_factor = 2.0 * float(degeneracy) / float(self.grid.volume_element)
        occupations = np.asarray(occupations, dtype=np.float64)
        from ..Eigensolvers.symmetry import CuPySymmetryOrbitals

        if isinstance(wavefunctions, CuPySymmetryOrbitals):
            tau, along = self._sector_tau(cp, wavefunctions, occupations, gradient)
            tau = factor * tau
            if gradient:
                cartesian = self.directions @ (gradient_factor * along)
        else:
            term = self.terms[None]
            with cp.cuda.Device(term.device_id):
                block = cp.asarray(wavefunctions, dtype=cp.float64)
                if block.ndim == 1:
                    block = block[:, None]
                sums = term.kinetic_sums(block, occupations, gradient=gradient)
                if gradient:
                    sums, products = sums
                    cartesian = gradient_factor * np.stack([cp.asnumpy(value) for value in products])
                total = None
                for value in sums:
                    total = value if total is None else total + value
                tau = factor * cp.asnumpy(total)
            if self.reducer is not None:
                tau = self.reducer.wedge_values(self.reducer.from_full(tau))
                if gradient:
                    rows = np.asarray(self.reducer.reduction.representative_rows, dtype=np.int64)
                    cartesian = cartesian[:, rows]
        result = self._layout(np.ascontiguousarray(tau))
        self.statistics["tau_seconds"] += perf_counter() - started
        self.statistics["tau_calls"] += 1
        if not gradient:
            return result
        return result, tuple(self._layout(np.ascontiguousarray(row)) for row in cartesian)

    def _sector_tau(self, cp, wavefunctions, occupations, gradient: bool):
        """``(sum, along)``: per orbit ``sum_n f_n sum_a (B x)^2 / m`` and (``gradient``) ``sum_n f_n x (B_a x) / m``."""

        decomposition = self.sectors.decomposition
        multiplicities = np.asarray(decomposition.reduction.multiplicities, dtype=np.float64)
        wedge = np.zeros(decomposition.wedge_size, dtype=np.float64)
        along = np.zeros((3, decomposition.wedge_size), dtype=np.float64) if gradient else None
        handovers = getattr(wavefunctions, "sector_handovers", None) or ()
        if any(record is not None and not getattr(record, "rotated", True) for record in handovers):
            ensure = getattr(self.sectors, "ensure_ritz_vectors", None)
            if not callable(ensure):
                raise NotImplementedError("tau of a sector basis that holds a Ritz hand-over")
            ensure()
        for representation in range(decomposition.representation_count):
            output_columns = np.flatnonzero(wavefunctions.representations == representation)
            if output_columns.size == 0:
                continue
            term = self.terms.get(representation)
            if term is None:
                raise RuntimeError(f"sector {representation} has no meta-GGA term on this process")
            weights = occupations[output_columns]
            with cp.cuda.Device(term.device_id):
                if wavefunctions.sector_vectors is not None:
                    vectors = wavefunctions.sector_vectors[representation]
                    if hasattr(vectors, "group"):
                        raise NotImplementedError("tau of a sector basis shared by several devices")
                    source = wavefunctions.representation_columns[output_columns]
                    prefix = np.array_equal(source, np.arange(source.size, dtype=source.dtype))
                    block = vectors[:, : source.size] if prefix else vectors[:, source]
                    block = cp.asarray(block, dtype=cp.float64)
                else:
                    orbits = decomposition.sector_orbit_indices(representation)
                    scaled = wavefunctions.scaled_wedge_vectors
                    gathered = scaled[cp.asarray(orbits)][:, cp.asarray(output_columns)]
                    block = cp.asarray(gathered) * cp.asarray(np.sqrt(multiplicities[orbits]))[:, None]
                sums = term.kinetic_sums(block, weights, gradient=gradient)
                if gradient:
                    sums, products = sums
                    for axis, (orbits, values) in enumerate(zip(term.shared_orbits, products)):
                        along[axis, orbits] += cp.asnumpy(values) / multiplicities[orbits]
                for target, values in zip(term.targets, sums):
                    orbits = decomposition.sector_orbit_indices(target)
                    wedge[orbits] += cp.asnumpy(values) / multiplicities[orbits]
        return wedge, along


__all__ = [
    "DeviceCSR",
    "DeviceKineticTerm",
    "DeviceMetaGGA",
    "HostKineticHamiltonian",
    "OrbitVectorFields",
    "SectorDerivative",
    "cartesian_operations",
    "derivative_directions",
    "orbit_vector_fields",
    "reduce_between_sectors",
    "sector_derivatives",
]


class HostKineticHamiltonian:
    """A bound CPU-backend Hamiltonian plus the host orbital term of a meta-GGA.

    The native and SciPy backends apply their own kinetic, local and KB
    terms; :class:`parsec_python.V_xc.meta_gga.KineticPotentialOperator` is
    added to every application, as :class:`parsec_python.Hamiltonian.KohnShamHamiltonian`
    adds it.
    """

    def __init__(self, bound, term) -> None:
        self.bound = bound
        self.term = term

    @property
    def shape(self):
        return self.bound.shape

    @property
    def spectral_margin(self) -> float:
        return float(self.term.spectral_margin)

    def without_orbital_term(self):
        """The bound CPU Hamiltonian without the term (the bare spectral bound)."""

        return self.bound

    def apply(self, vectors):
        return self.bound.apply(vectors) + self.term.apply(np.asarray(vectors, dtype=np.float64))

    def apply_kinetic(self, vectors):
        return self.bound.apply_kinetic(vectors) + self.term.apply(np.asarray(vectors, dtype=np.float64))

    def apply_local(self, vectors):
        return self.bound.apply_local(vectors)

    def apply_nonlocal(self, vectors):
        return self.bound.apply_nonlocal(vectors)

    def as_linear_operator(self):
        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(self.shape, matvec=self.apply, matmat=self.apply, rmatvec=self.apply, dtype=np.float64)

    def as_eigensolver_operator(self):
        return self.as_linear_operator()

    def as_sparse(self) -> sp.csr_matrix:
        return (self.bound.as_sparse() + self.term.as_sparse()).tocsr()

    def __matmul__(self, vectors):
        return self.apply(vectors)
