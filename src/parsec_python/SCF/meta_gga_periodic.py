"""The meta-GGA pieces of the k-point loops of a periodic cell.

:func:`parsec_python.SCF.kpoints.run_scf_kpoints` and
:func:`~parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized` run a meta-GGA
with the loop of :mod:`parsec_python.SCF.meta_gga` (PBE in the first
iteration, ``(V, vtau)`` mixed together, the double counting
``2 h^3 sum vtau_in tau`` removed from the energy).  What the k-points add:

* ``tau`` of Bloch factors ``u_nk`` (``psi_nk = e^{ik.r} u_nk``), whose
  derivative is ``(G + ik) u``:

  ``tau(r) = sum_k w_k (d/2h^3) sum_n f_nk |(G + ik) u_nk|^2``   (Hartree),

  ``d`` = 2 unpolarized, 1 per spin channel, ``w_k`` the normalized k-point
  weights, ``G`` the first-derivative stencil of the cell (``system.gradient``,
  the one of the Bloch kinetic term).  Time reversal gives ``u_-k = conj(u_k)``
  and the same ``tau`` pointwise, so a k and -k merged with twice the weight
  is exact; the points of an irreducible wedge are completed by the
  space-group average of ``tau``, as of the density (``density_symmetry``).
* in the orbital form of the density gradient
  (:func:`parsec_python.V_xc.meta_gga.meta_gga_gradient`) the gradient of the
  Bloch factors, ``g = sum_k w_k (d/h^3) sum_n f_nk 2 Re[conj(u_nk) (G + ik) u_nk]``
  ``= sum_k w_k (d/h^3) sum_n f_nk 2 Re[conj(u_nk) G u_nk]`` (the ``ik`` part is
  imaginary; time reversal keeps ``g`` too);
* the orbital terms ``(G + ik)^H diag(vtau) (G + ik)`` (and ``2 sum_c [W_c G_c +
  G_c^T W_c]`` of ``W = de/dg``) of each Bloch Hamiltonian
  (:class:`parsec_python.Hamiltonian.kpoint_operator.KPointKohnShamHamiltonian`
  with ``kinetic_potential`` and ``gradient_flux``).

``tau`` is formed where a k-point worker keeps the orbitals: on the host, or
on its device (:mod:`parsec_python.acceleration.Hamiltonian.bloch_meta_gga_device`).
"""

from __future__ import annotations

import numpy as np


def kpoint_meta_gga_loop(system, *, channels: int, warm_start=None):
    """The :class:`~parsec_python.SCF.meta_gga.MetaGGALoop` of a k-point loop, ``None`` for another functional.

    ``warm_start`` holds the ``kinetic_potential`` (one field, or a pair for
    spin) and, in the orbital form of the density gradient, the
    ``gradient_flux`` a loop starts from (the structure before a move);
    without them the loop starts with PBE and no orbital terms.
    """

    if not getattr(system.input.scf, "is_meta_gga", False):
        return None
    from .single_point import meta_gga_loop

    return meta_gga_loop(system, channels=channels, adapter=None, warm_start=warm_start)


def bloch_orbital_fields(
    system, k_point, vectors, occupations, weight: float, degeneracy: float, xp=np, *, with_gradient: bool = False
) -> dict:
    """The meta-GGA fields of one task's Bloch factors: ``{"tau": ..., ["gradient": (3, n)]}`` (Hartree, bohr).

    ``tau = w_k (d/2h^3) sum_n f_n |(G + ik) u_n|^2`` and, with
    ``with_gradient``, ``g = w_k (d/h^3) sum_n f_n 2 Re[conj(u_n) G u_n]``.
    ``vectors`` are the Euclidean-normalized grid vectors of the task; on a
    device (``xp`` = CuPy) the fields are formed there and returned as device
    arrays, which the k-point worker brings to the host with the density.
    """

    if int(vectors.shape[0]) != int(system.grid.size):
        raise NotImplementedError(
            "the kinetic-energy density of a meta-GGA needs the orbitals on every grid point; the "
            "MPI-distributed k-point solver holds a block of grid rows on each rank (run without --mpi)"
        )
    scale = float(weight) * 0.5 * float(degeneracy) / float(system.grid.volume_element)
    if xp is not np and not isinstance(vectors, np.ndarray):
        from ..acceleration.Hamiltonian.bloch_meta_gga_device import device_bloch_fields

        fields = device_bloch_fields(
            xp, system.gradient, vectors, occupations, k_point, scale, with_gradient=with_gradient
        )
    else:
        fields = _host_bloch_fields(
            system.gradient, np.asarray(vectors), occupations, k_point, scale, with_gradient=with_gradient
        )
    if with_gradient:
        return {"tau": fields[0], "gradient": fields[1]}
    return {"tau": fields}


def _host_bloch_fields(gradient, vectors, occupations, k_point, scale: float, *, with_gradient: bool):
    """The host form of :func:`parsec_python.acceleration.Hamiltonian.bloch_meta_gga_device.device_bloch_fields`."""

    weights = np.asarray(occupations, dtype=np.float64)
    size = int(vectors.shape[0])
    tau = np.zeros(size, dtype=np.float64)
    density_gradient = np.zeros((3, size), dtype=np.float64) if with_gradient else None
    selected = np.flatnonzero(weights != 0.0)
    if selected.size:
        block = np.ascontiguousarray(vectors[:, selected], dtype=np.complex128)
        occupied = weights[selected]
        k = np.zeros(3) if k_point is None else np.asarray(k_point, dtype=np.float64).reshape(3)
        for axis, matrix in enumerate(gradient):
            derivative = (matrix @ block.view(np.float64)).view(np.complex128)
            if with_gradient:
                density_gradient[axis] = (block.real * derivative.real + block.imag * derivative.imag) @ occupied
            if k[axis] != 0.0:
                derivative = derivative + (1j * float(k[axis])) * block
            tau += (derivative.real * derivative.real + derivative.imag * derivative.imag) @ occupied
    tau *= float(scale)
    if not with_gradient:
        return tau
    return tau, (4.0 * float(scale)) * density_gradient


def symmetrized_tau(tau, density_symmetry, volume_element):
    """``tau`` averaged over the space group of ``Symmetrize_Charge_Density`` (as the density), or unchanged."""

    if density_symmetry is None:
        return tau
    from ..Grid.point_group import symmetrize_density

    averaged, _ = symmetrize_density(tau, density_symmetry, volume_element)
    return averaged


def symmetrized_gradient(gradient, density_symmetry, lattice_vectors):
    """The orbital density gradient ``g`` (3, n) averaged over the space group, as a vector field.

    An operation maps fractional coordinates as ``rw' = M (rw - t)``
    (:func:`parsec_python.Grid.space_group.periodic_grid_symmetry`); with the
    lattice vectors as rows ``L`` it is ``r' = r C - c`` in Cartesian row
    vectors, ``C = L^-1 M^T L``, and the gradient of an invariant field obeys
    ``g(r) = C g(r')``.  So ``g_sym(r) = (1/|G|) sum_g C_g g(image_g(r))``, the
    vector counterpart of ``symm_scalar``.  Unchanged without a group.
    """

    if density_symmetry is None:
        return gradient
    values = np.asarray(gradient, dtype=np.float64)
    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    inverse = np.linalg.inv(lattice)
    total = np.zeros_like(values)
    for matrix, images in zip(np.asarray(density_symmetry.matrices, dtype=np.float64), density_symmetry.images):
        cartesian = inverse @ matrix.T @ lattice
        if not np.allclose(cartesian @ cartesian.T, np.eye(3), atol=1.0e-8):
            raise NotImplementedError(
                "an operation of the symmetrized density is not a Cartesian rotation of this cell; the orbital "
                "density gradient of a meta-GGA cannot be symmetrized with it"
            )
        total += cartesian @ values[:, images]
    return total / float(len(density_symmetry.images))


__all__ = [
    "bloch_orbital_fields",
    "kpoint_meta_gga_loop",
    "symmetrized_gradient",
    "symmetrized_tau",
]
