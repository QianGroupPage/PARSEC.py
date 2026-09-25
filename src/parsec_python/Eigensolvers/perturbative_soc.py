"""PARSEC's default perturbative spin-orbit correction (``pls.F90``).

Given a converged collinear spin-polarized ground state
(:class:`~parsec_python.models.SpinPolarizedSinglePointResult`), this module
builds and diagonalizes the small dense ``2N x 2N`` complex Hermitian matrix

``H_spin[i,sigma; j,sigma'] = delta_ij*delta_sigma,sigma'*eps_i,sigma
                               + <psi_i,sigma | L.S | psi_j,sigma'>``

in the space of the ``N`` converged eigenstates from *each* spin channel,
giving SOC-split eigenvalues and 2-component spinor eigenvectors -- without
touching the real-space grid Hamiltonian, eigensolver, or SCF loop.  This is
Fortran PARSEC's default (``SCF_SO=false``) spin-orbit treatment; it is
strictly cheaper than the self-consistent (``SO_from_scratch``) path because
diagonalization happens once, after SCF, in a basis of only ``2*N`` vectors.

The ``L.S`` matrix elements are built from PARSEC's separable Kleinman-Bylander
spin-orbit projector pair (:meth:`~parsec_python.Pseudopotential.ParsecPseudopotential.spin_orbit_projectors`,
which returns each species' radial ``V_ion``/``V_so`` functions) evaluated on
the grid with *complex* spherical harmonics ``Y_l^ml`` -- unlike the ordinary
scalar KB projectors (real harmonics), the ladder-operator structure of
``L.S`` requires the complex ``|l,ml>`` basis.  Angular-momentum matrix
elements (``lz``, the ``L+-`` ladder coefficients) and their combination into
``lzsz``/``lsxy`` follow ``pls.F90`` exactly, including its slightly
non-obvious three-term ``{|Vion><Vso|+h.c.} - (1/2)|Vso><Vso|`` structure
plus a constant ``(1/4)*l(l+1)|Vso><Vso|`` piece (``lzsz`` only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..Grid import RealSpaceGrid
from ..models import Atom, SpeciesPotential, SpinPolarizedSinglePointResult
from ..Pseudopotential import ParsecPseudopotential
from ..V_ion.ionic_potential import _projector_support_radius

# Angular-momentum channels PARSEC supports for spin-orbit: l=1 (p, 3
# m_l components) then l=2 (d, 5 m_l components), 8 (l,m_l) slots total,
# matching pls.F90's fixed lm=1..8 loop (do lj=1,2: do ml=-lj,lj).
_SOC_ANGULAR_MOMENTA: tuple[int, ...] = (1, 2)


def complex_spherical_harmonics(
    angular_momentum: int, relative_coordinates: np.ndarray
) -> np.ndarray:
    """Complex spherical harmonics ``Y_l^ml``, ``ml=-l..l`` (Condon-Shortley).

    Returned columns are ordered ``ml=-l,...,l``, matching ``pls.F90``'s
    ``do ml=-lj,lj`` loop.  Supports ``l=1`` and ``l=2``, PARSEC's spin-orbit
    angular momenta.  Direction cosines are computed the same way as
    :func:`~parsec_python.V_ion.ionic_potential.real_spherical_harmonics`
    (zero at the exact atomic center).
    """
    xyz = np.asarray(relative_coordinates, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("relative_coordinates must have shape (n, 3)")
    radius = np.linalg.norm(xyz, axis=1)
    unit = np.zeros_like(xyz)
    nonzero = radius > 0
    unit[nonzero] = xyz[nonzero] / radius[nonzero, None]
    x, y, z = unit.T

    if angular_momentum == 1:
        y1m1 = np.sqrt(3.0 / (8.0 * np.pi)) * (x - 1j * y)
        y10 = np.sqrt(3.0 / (4.0 * np.pi)) * z.astype(complex)
        y1p1 = -np.sqrt(3.0 / (8.0 * np.pi)) * (x + 1j * y)
        return np.column_stack((y1m1, y10, y1p1))
    if angular_momentum == 2:
        xpiy = x + 1j * y
        xmiy = x - 1j * y
        y2m2 = np.sqrt(15.0 / (32.0 * np.pi)) * xmiy * xmiy
        y2m1 = np.sqrt(15.0 / (8.0 * np.pi)) * z * xmiy
        y20 = np.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * z * z - 1.0).astype(complex)
        y2p1 = -np.sqrt(15.0 / (8.0 * np.pi)) * z * xpiy
        y2p2 = np.sqrt(15.0 / (32.0 * np.pi)) * xpiy * xpiy
        return np.column_stack((y2m2, y2m1, y20, y2p1, y2p2))
    raise ValueError("spin-orbit complex harmonics support only l=1 and l=2")


@dataclass(frozen=True)
class AtomSpinOrbitProjectors:
    """One SOC atom's grid-sampled ``V_ion``/``V_so`` projector pair.

    ``support_rows`` indexes the grid points where these projectors are
    nonzero.  ``v_ion``/``v_so`` have shape ``(len(support_rows), 8)``: the 8
    columns are, in order, l=1's ``m_l=-1,0,1`` then l=2's
    ``m_l=-2,-1,0,1,2``, matching ``pls.F90``'s fixed ``lm=1..8`` indexing so
    the ladder-operator index shift ``lm+m`` (``m=+/-1``) stays meaningful.
    ``sign`` holds each l's KB sign, indexed ``[0]`` for l=1 and ``[1]`` for
    l=2 (``nloc_p_pot%cc(lj,iso)``).
    """

    atom_index: int
    support_rows: np.ndarray
    v_ion: np.ndarray
    v_so: np.ndarray
    sign: tuple[float, float]


def build_spin_orbit_projectors(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
) -> list[AtomSpinOrbitProjectors]:
    """Build the complex-harmonic SOC projector pair for every SOC atom.

    Only atoms whose species has ``SpeciesPotential.spin_orbit=True`` (and
    therefore, per :func:`~parsec_python.V_ion.load_pseudopotentials`'s
    validation, a relativistic POTRE file with matching SO channels)
    contribute.  Mirrors ``nonloc.F90``'s SOC branch: one projector pair per
    ``(atom, l)`` spanning ``2l+1`` complex spherical harmonics, sampled
    directly on the active grid (``Double_Grid_order=1``).
    """
    sqrt_dv = np.sqrt(grid.volume_element)
    result: list[AtomSpinOrbitProjectors] = []
    for atom_index, atom in enumerate(atoms):
        specification = specifications[atom.symbol]
        if not specification.spin_orbit:
            continue
        potential = potentials[atom.symbol]
        local_l = specification.local_angular_momentum

        support_radius = _projector_support_radius(potential)
        position = np.asarray(atom.position, dtype=np.float64)
        relative = grid.coordinates - position
        radius = np.linalg.norm(relative, axis=1)
        support = radius <= support_radius
        support_rows = np.flatnonzero(support)
        if support_rows.size == 0:
            continue
        relative_support = relative[support]
        radius_support = radius[support]

        v_ion = np.zeros((support_rows.size, 8), dtype=complex)
        v_so = np.zeros((support_rows.size, 8), dtype=complex)
        sign_by_l: dict[int, float] = {}
        column = 0
        for angular_momentum in _SOC_ANGULAR_MOMENTA:
            width = 2 * angular_momentum + 1
            if angular_momentum not in potential.spin_orbit_channel_potentials:
                column += width
                continue
            radial_ion, radial_so, sign = potential.spin_orbit_projectors(
                angular_momentum, local_l
            )
            interpolated_ion = np.interp(
                radius_support,
                potential.radii,
                radial_ion,
                left=radial_ion[0],
                right=0.0,
            )
            interpolated_so = np.interp(
                radius_support,
                potential.radii,
                radial_so,
                left=radial_so[0],
                right=0.0,
            )
            harmonics = complex_spherical_harmonics(angular_momentum, relative_support)
            v_ion[:, column : column + width] = (
                sqrt_dv * interpolated_ion[:, None] * harmonics
            )
            v_so[:, column : column + width] = (
                sqrt_dv * interpolated_so[:, None] * harmonics
            )
            sign_by_l[angular_momentum] = sign
            column += width

        result.append(
            AtomSpinOrbitProjectors(
                atom_index=atom_index,
                support_rows=support_rows,
                v_ion=v_ion,
                v_so=v_so,
                sign=(sign_by_l.get(1, 0.0), sign_by_l.get(2, 0.0)),
            )
        )
    return result


# lm=1..8 -> (l, m_l): l=1 gives m_l=-1,0,1 (columns 0-2); l=2 gives
# m_l=-2,...,2 (columns 3-7).  Matches pls.F90's `do lj=1,2: do ml=-lj,lj`.
_LM_TO_L = np.array([1, 1, 1, 2, 2, 2, 2, 2])
_LM_TO_ML = np.array([-1, 0, 1, -2, -1, 0, 1, 2])


def _projector_overlaps(
    projectors: Sequence[AtomSpinOrbitProjectors], wavefunctions: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-atom ``<v_ion,lm|psi_n>``/``<v_so,lm|psi_n>`` overlap matrices.

    Returns two lists (one entry per SOC atom) of complex arrays shaped
    ``(n_states, 8)``, each already scaled by that atom-l's KB sign
    (``nloc_p_pot%cc``), matching ``lzsz``/``lsxy``'s ``dotio``/``dotso``.
    """
    dotio_by_atom: list[np.ndarray] = []
    dotso_by_atom: list[np.ndarray] = []
    for atom_projectors in projectors:
        rows = atom_projectors.support_rows
        block = np.asarray(wavefunctions[rows, :], dtype=complex)
        # <v|psi> = sum_i conj(v_i)*psi_i, matching lzsz/lsxy's conjg(v)*p.
        dotio = atom_projectors.v_ion.conj().T @ block
        dotso = atom_projectors.v_so.conj().T @ block
        sign = np.where(_LM_TO_L == 1, atom_projectors.sign[0], atom_projectors.sign[1])
        dotio_by_atom.append((dotio * sign[:, None]).T)
        dotso_by_atom.append((dotso * sign[:, None]).T)
    return dotio_by_atom, dotso_by_atom


def _lzsz_block(
    projectors: Sequence[AtomSpinOrbitProjectors],
    dotio: Sequence[np.ndarray],
    dotso: Sequence[np.ndarray],
    spin_sign: int,
) -> np.ndarray:
    """``H[i,j] = conj(<psi_i,sigma| lzsz_sigma |psi_j,sigma>)``.

    Directly transcribes ``lzsz``'s grid computation plus ``pls.F90``'s
    ``h_spin(i,j) += conjg(dot_product(psi_i, q_tmp))``, without building any
    grid-space vector.  ``dotio``/``dotso`` (from :func:`_projector_overlaps`)
    already carry the per-``(atom,l)`` KB sign baked in once, matching how
    ``lzsz`` uses them directly as ``q_tmp``'s coefficients on the *ket*
    side.  Reading the *same* stored value as a *bra*,
    ``<psi_i|v_ion,lm> = conj(<v_ion,lm|psi_i>) = conj(dotio_i(lm)/sign(lm))
    = sign(lm)*conj(dotio_i(lm))`` (since ``sign(lm) = +/-1``) -- an extra
    factor of ``sign(lm)`` that is easy to drop by eye and was in fact
    missing from an earlier version of this function (caught only by
    cross-checking against the independently-verified matrix-free grid
    operator in ``Hamiltonian.spinor_operator``, not by this function's own
    Hermiticity, which a self-consistently-missing sign still satisfies).
    ``spin_sign`` is +1 for the up channel, -1 for down (``lzsz``'s ``m``).
    """
    n_states = dotio[0].shape[0]
    h = np.zeros((n_states, n_states), dtype=complex)
    lz = spin_sign * 0.5 * _LM_TO_ML
    quarter_ll1 = 0.25 * _LM_TO_L * (_LM_TO_L + 1)
    for atom, atom_dotio, atom_dotso in zip(projectors, dotio, dotso):
        sign_by_lm = np.where(_LM_TO_L == 1, atom.sign[0], atom.sign[1])
        for lm in range(8):
            dotio_col = atom_dotio[:, lm]
            dotso_col = atom_dotso[:, lm]
            term = lz[lm] * (
                np.outer(np.conj(dotio_col), dotso_col)
                + np.outer(np.conj(dotso_col), dotio_col)
                - 0.5 * np.outer(np.conj(dotso_col), dotso_col)
            ) + quarter_ll1[lm] * np.outer(np.conj(dotso_col), dotso_col)
            h += sign_by_lm[lm] * np.conj(term)
    return h


def _lsxy_block(
    projectors: Sequence[AtomSpinOrbitProjectors],
    dotio_in: Sequence[np.ndarray],
    dotso_in: Sequence[np.ndarray],
    dotio_out: Sequence[np.ndarray],
    dotso_out: Sequence[np.ndarray],
    spin_sign: int,
) -> np.ndarray:
    """``H[i,j] = conj(<psi_i,sigma'| lsxy_m |psi_j,sigma>)``.

    ``dotio_in``/``dotso_in`` are the *input*-channel sigma overlaps (state
    index ``j``, unshifted ``lm``).  ``dotio_out``/``dotso_out`` are the
    *output*-channel sigma' overlaps (state index ``i``): ``lsxy`` samples
    the output-channel projector overlap at the ladder-shifted index
    ``lm+m``, so ``<psi_i,sigma'|v_ion,lm+m> = sign(lm+m)*conj(dotio_out_i(lm+m))``
    -- see :func:`_lzsz_block`'s docstring for why the extra ``sign`` factor
    is needed when reading a sign-included stored overlap as a bra.
    ``spin_sign`` is ``pls.F90``'s ``m`` argument to ``lsxy`` -- the ladder
    direction, not sigma' itself: ``m=-1`` (down input) produces the up-row
    block, ``m=+1`` (up input) produces the down-row block.
    """
    n_out = dotio_out[0].shape[0]
    n_in = dotio_in[0].shape[0]
    h = np.zeros((n_out, n_in), dtype=complex)
    for atom, atom_in_io, atom_in_so, atom_out_io, atom_out_so in zip(
        projectors, dotio_in, dotso_in, dotio_out, dotso_out
    ):
        for lm in range(8):
            target = lm + spin_sign
            if target < 0 or target > 7 or _LM_TO_L[target] != _LM_TO_L[lm]:
                continue
            l = int(_LM_TO_L[lm])
            m_l = int(_LM_TO_ML[lm])
            number = l * (l + 1) - m_l * (spin_sign + m_l)
            if number <= 0:
                continue
            weight = 0.5 * np.sqrt(float(number))
            target_sign = atom.sign[0] if _LM_TO_L[target] == 1 else atom.sign[1]

            dotso_j = atom_in_so[:, lm]
            dotio_j = atom_in_io[:, lm]
            v_ion_overlap_i = atom_out_io[:, target]
            v_so_overlap_i = atom_out_so[:, target]
            term = target_sign * weight * (
                np.outer(np.conj(v_ion_overlap_i), dotso_j)
                + np.outer(np.conj(v_so_overlap_i), dotio_j)
                - 0.5 * np.outer(np.conj(v_so_overlap_i), dotso_j)
            )
            h += np.conj(term)
    return h


@dataclass(frozen=True)
class PerturbativeSpinOrbitResult:
    """SOC-split spinor eigenpairs from PARSEC's ``pls.F90`` diagonalization."""

    eigenvalues: np.ndarray
    """Length ``2*n_states``, ascending."""
    spinor_coefficients_up: np.ndarray
    """Shape ``(n_states, 2*n_states)``: column k's linear-combination
    coefficients over the converged spin-up basis states."""
    spinor_coefficients_down: np.ndarray
    """Shape ``(n_states, 2*n_states)``, spin-down basis coefficients."""
    magnetic_moment: np.ndarray
    """Length ``2*n_states``: ``<S_z>`` (up minus down grid-norm) per
    SOC-split spinor eigenstate, matching ``elec_st%magmom``."""


def perturbative_spin_orbit_correction(
    grid: RealSpaceGrid,
    scf_result: SpinPolarizedSinglePointResult,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
) -> PerturbativeSpinOrbitResult:
    """Diagonalize PARSEC's perturbative spin-orbit ``2N x 2N`` matrix.

    ``scf_result`` must come from a converged
    :func:`~parsec_python.SCF.spin_polarized.run_scf_spin_polarized` call on
    the same ``atoms``/``potentials``/``specifications``.  Both channels must
    supply the same number of states (``scf_result.eigenvalues_up.size ==
    scf_result.eigenvalues_down.size``), matching ``pls.F90``'s single
    ``elec_st%nstate`` per channel.
    """
    n_states = scf_result.eigenvalues_up.size
    if scf_result.eigenvalues_down.size != n_states:
        raise ValueError(
            "perturbative spin-orbit correction requires equal state counts "
            "in both spin channels"
        )
    projectors = build_spin_orbit_projectors(grid, atoms, potentials, specifications)
    if not projectors:
        raise ValueError(
            "no atom has SpeciesPotential.spin_orbit=True with matching "
            "spin-orbit pseudopotential channels"
        )

    dotio_up, dotso_up = _projector_overlaps(projectors, scf_result.wavefunctions_up)
    dotio_down, dotso_down = _projector_overlaps(projectors, scf_result.wavefunctions_down)

    nmax = 2 * n_states
    h_spin = np.zeros((nmax, nmax), dtype=complex)
    h_spin[:n_states, :n_states] = np.diag(scf_result.eigenvalues_up.astype(complex))
    h_spin[n_states:, n_states:] = np.diag(scf_result.eigenvalues_down.astype(complex))

    h_spin[:n_states, :n_states] += _lzsz_block(projectors, dotio_up, dotso_up, spin_sign=1)
    h_spin[n_states:, n_states:] += _lzsz_block(
        projectors, dotio_down, dotso_down, spin_sign=-1
    )
    # up-down block: lsxy(m=-1) maps a down input to an up-indexed row.
    h_spin[:n_states, n_states:] += _lsxy_block(
        projectors, dotio_down, dotso_down, dotio_up, dotso_up, spin_sign=-1
    )
    # down-up block: lsxy(m=+1) maps an up input to a down-indexed row.
    h_spin[n_states:, :n_states] += _lsxy_block(
        projectors, dotio_up, dotso_up, dotio_down, dotso_down, spin_sign=1
    )

    # Hermitize: pls.F90 builds each block independently via separate lzsz/
    # lsxy calls and never explicitly symmetrizes off-diagonal round-off
    # between the (up,down) and (down,up) blocks it assembles from two
    # different lsxy(m=-1)/lsxy(m=+1) evaluations that are only equal in
    # exact arithmetic.
    h_spin = 0.5 * (h_spin + h_spin.conj().T)

    eigenvalues, eigenvectors = np.linalg.eigh(h_spin)
    spinor_up = eigenvectors[:n_states, :]
    spinor_down = eigenvectors[n_states:, :]
    magnetic_moment = np.sum(np.abs(spinor_up) ** 2, axis=0) - np.sum(
        np.abs(spinor_down) ** 2, axis=0
    )
    return PerturbativeSpinOrbitResult(
        eigenvalues=eigenvalues,
        spinor_coefficients_up=spinor_up,
        spinor_coefficients_down=spinor_down,
        magnetic_moment=magnetic_moment,
    )


__all__ = [
    "AtomSpinOrbitProjectors",
    "PerturbativeSpinOrbitResult",
    "build_spin_orbit_projectors",
    "complex_spherical_harmonics",
    "perturbative_spin_orbit_correction",
]
