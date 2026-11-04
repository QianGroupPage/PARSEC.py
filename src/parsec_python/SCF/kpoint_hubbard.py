"""LDA+U in the k-point loops (Bloch projectors, k-summed occupations).

PARSEC builds the occupation matrices of every iteration but the first from
the orbitals of the iteration before (``parsec.F90:1135``, ``calcden.f90z``)
and adds ``sum_(m,m') |phi_m> D_(m,m') <phi_m'|`` to the Hamiltonian of every
k-point (``plusu.f90z``).  Its k-point forms are wrong (the projector offset
of ``plusu.f90z`` never advances past the first atom and ``calcden.f90z``
projects without the Bloch phases), so the k-point loops of this port use the
correct Bloch form of :mod:`parsec_python.Hamiltonian.hubbard`:

* the projectors of k-point ``k`` carry ``exp(-i k.(r + L))``
  (:func:`~parsec_python.Hamiltonian.hubbard.build_hubbard_projectors` with
  ``k_point``), built once per k-point;
* ``N^s = sum_k w_k sum_n f_nks <phi|psi_nks><psi_nks|phi>`` is summed over the
  k-points of every channel, its real part kept (time reversal of collinear
  spin) and, for an irreducible wedge of a space group, averaged over the
  group with the rotation of the d harmonics;
* one ``D^s`` from that ``N`` serves every k-point of the channel.

A k-point Hamiltonian with the term is a :class:`BlochHubbardHamiltonian`.
The device workers of the acceleration package apply it on their device
(``acceleration/Hamiltonian/device_bloch.py``: ``P_k D P_k^H`` beside the KB
term), and :meth:`KPointHubbard.occupation` forms ``N_k`` where the orbitals
are (``xp``), so LDA+U takes the backend every k-point loop takes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .kpoint_solver import whole_orbitals
from ..Hamiltonian.hubbard import (
    HubbardOperator,
    HubbardProjectors,
    HubbardTerms,
    build_hubbard_projectors,
    hubbard_atom_permutations,
    hubbard_replica_lattice,
    hubbard_symmetry_lattice,
    hubbard_terms,
    kpoint_occupation_matrix,
    symmetrize_occupations,
    time_reversal_occupations,
)


@dataclass(frozen=True)
class BlochHubbardHamiltonian:
    """A k-point (or band) Hamiltonian plus its LDA+U term ``P_k D P_k^H``."""

    base: object
    hubbard: HubbardOperator

    @property
    def shape(self) -> tuple[int, int]:
        return self.base.shape

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        result = np.asarray(self.base.apply(vectors))
        return result + self.hubbard.apply(vectors)

    def as_linear_operator(self):
        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(
            self.shape,
            matvec=self.apply,
            matmat=self.apply,
            rmatvec=self.apply,
            dtype=complex,
        )


def kpoint_backend(system) -> str | None:
    """The k-point backend a loop of ``system`` asks for: ``None``, the default one.

    LDA+U once kept its loops on the CPU, when the device workers had no
    LDA+U term; they apply it now (see the module text).
    """

    return None


@dataclass
class KPointHubbard:
    """The LDA+U state of one k-point loop.

    ``k_points`` are the Cartesian k-points of the loop; ``operations`` the
    group average of :func:`~parsec_python.Hamiltonian.hubbard.symmetrize_occupations`
    (``None``: none).  The Bloch projectors are built on first use (in the
    worker that owns the k-point) and kept.
    """

    system: object
    k_points: np.ndarray
    operations: list | None = None
    _projectors: dict = field(default_factory=dict, repr=False)
    # Per k-point, the support rows and values of every atom's five
    # projectors (host), and their copies on the devices that hold orbitals.
    _blocks: dict = field(default_factory=dict, repr=False)
    _device_blocks: dict = field(default_factory=dict, repr=False)

    @classmethod
    def for_system(cls, system, k_points: np.ndarray) -> "KPointHubbard | None":
        """The state of ``system``'s LDA+U atoms, ``None`` without LDA+U."""

        projectors: HubbardProjectors | None = getattr(system, "hubbard", None)
        if projectors is None:
            return None
        problem = system.input
        operations = None
        group = getattr(problem, "space_group", None)
        if (
            getattr(problem, "monkhorst_pack_shift", None) is not None
            and group is not None
            and group.order > 1
        ):
            # An irreducible wedge of the space group (irrbz.f90): complete
            # the stars by the group average of the occupation matrices (the
            # group of a wire or slab is that of alatt).
            operations = hubbard_atom_permutations(
                system.atoms,
                projectors.atom_indices,
                hubbard_symmetry_lattice(system),
                group,
            )
        return cls(system=system, k_points=np.asarray(k_points, dtype=float), operations=operations)

    @property
    def template(self) -> HubbardProjectors:
        return self.system.hubbard

    def projectors(self, k_index: int) -> HubbardProjectors:
        cached = self._projectors.get(k_index)
        if cached is None:
            system = self.system
            # Replicas along the periodic vectors (a wire or slab: those
            # alone, nreplica(1:pbc%per) = 1 of upot.F90).
            cached = build_hubbard_projectors(
                system.grid,
                system.atoms,
                system.pseudopotentials,
                system.input.pseudopotentials,
                hubbard_replica_lattice(system),
                k_point=self.k_points[k_index],
            )
            self._projectors[k_index] = cached
        return cached

    def hamiltonian(self, base, k_index: int, potential_matrices: np.ndarray | None):
        """``base`` plus the term of ``potential_matrices`` (``None``: ``base``)."""

        if potential_matrices is None:
            return base
        return BlochHubbardHamiltonian(
            base, HubbardOperator(self.projectors(k_index), potential_matrices)
        )

    def occupation(
        self, k_index: int, vectors, occupations, weight: float, xp=np
    ) -> np.ndarray:
        """``w_k N_k`` of one k-point and channel (complex, host ``(atoms, 5, 5)``).

        ``vectors`` are the orbitals of the whole grid where the worker keeps
        them: host arrays (``xp`` NumPy) or device arrays of the array module
        ``xp``, where ``N_k`` is formed on the device from the support rows of
        every atom (:meth:`_device_occupation`) and only the matrices come back.
        """

        grid_size = int(self.system.grid.size)
        if xp is np:
            return kpoint_occupation_matrix(
                self.projectors(k_index),
                whole_orbitals(vectors, grid_size, "LDA+U"),
                occupations,
                weight,
            )
        if int(vectors.shape[0]) != grid_size:
            whole_orbitals(vectors, grid_size, "LDA+U")  # refuses a block of grid rows
        return float(weight) * self._device_occupation(k_index, vectors, occupations, xp)

    def _atom_blocks(self, k_index: int):
        """``(rows, values)`` of every LDA+U atom at k-point ``k_index``: its support rows and the
        ``(rows, 5)`` values of its five Bloch projectors there."""

        blocks = self._blocks.get(k_index)
        if blocks is None:
            matrix = self.projectors(k_index).projectors.tocsc()
            blocks = []
            for atom in range(self.template.atom_count):
                columns = matrix[:, 5 * atom : 5 * atom + 5].tocsr()
                rows = np.flatnonzero(np.diff(columns.indptr))
                blocks.append((rows, np.asarray(columns[rows].toarray(), dtype=np.complex128)))
            self._blocks[k_index] = blocks
        return blocks

    def _device_occupation(self, k_index: int, vectors, occupations, xp) -> np.ndarray:
        """``N_k = sum_n f_n <phi|u_n><u_n|phi>`` of device orbitals (index order of
        :func:`~parsec_python.Hamiltonian.hubbard.occupation_matrices`)."""

        occupations = np.asarray(occupations, dtype=np.float64)
        occupied = np.flatnonzero(occupations != 0.0)
        atoms = self.template.atom_count
        if occupied.size == 0:
            return np.zeros((atoms, 5, 5), dtype=np.complex128)
        device = getattr(getattr(vectors, "device", None), "id", None)
        key = (k_index, device)
        blocks = self._device_blocks.get(key)
        if blocks is None:
            blocks = [
                (xp.asarray(rows), xp.asarray(values.conj().T))
                for rows, values in self._atom_blocks(k_index)
            ]
            self._device_blocks[key] = blocks
        selected = vectors[:, : int(occupied[-1]) + 1]
        weights = xp.asarray(occupations[: int(occupied[-1]) + 1])
        dots = xp.stack([adjoint @ selected[rows] for rows, adjoint in blocks])
        matrices = xp.einsum("amn,akn->akm", dots * weights[None, None, :], dots.conj())
        host = getattr(matrices, "get", None)
        return np.asarray(host() if host is not None else matrices, dtype=np.complex128)

    def terms(self, summed: Sequence[np.ndarray]) -> HubbardTerms:
        """``D`` and ``E_U`` of the k-summed matrices of every channel, up first."""

        matrices = np.stack([time_reversal_occupations(block) for block in summed])
        if self.operations is not None:
            matrices = symmetrize_occupations(matrices, self.operations)
        return hubbard_terms(self.template, list(matrices))


__all__ = ["BlochHubbardHamiltonian", "KPointHubbard", "kpoint_backend"]
