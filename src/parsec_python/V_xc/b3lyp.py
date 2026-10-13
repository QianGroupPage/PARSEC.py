"""Semilocal parts of the B3LYP exchange-correlation energy on the grid.

B3LYP is evaluated here as an *energy only*, for a density that another
functional made self-consistent (the PP-PBE(B3LYP) refinement of Xu,
Prendergast and Qian, J. Chem. Theory Comput. 18, 5471 (2022); ``Lrefine``
in ARES).  No potential is returned: nothing in this module enters a
Hamiltonian.  The definition is the one of Gaussian, libxc
(``HYB_GGA_XC_B3LYP``) and ARES,

``E_xc = 0.20 E_x[HF] + 0.08 E_x[Slater] + 0.72 E_x[B88]
        + 0.19 E_c[VWN-RPA] + 0.81 E_c[LYP]``,

where ``E_x[B88]`` is the complete Becke 1988 exchange (Slater plus its
gradient correction) and the local correlation is the Vosko-Wilk-Nusair fit
to the *RPA* correlation energy of the uniform gas ("VWN1RPA" in ARES,
functional III of Gaussian), not the fit to the Ceperley-Alder data (VWN5).
The exact (Hartree-Fock) exchange is the subject of
:mod:`parsec_python.V_xc.exact_exchange`; this module provides the four
density functionals.

All four are written for a spin-unpolarized density ``n = n_up + n_down``
with ``n_up = n_down``, and again, as the functions named ``*_spin``, for two
different spin densities.  Their gradient variables are ``sigma = |D n|**2``
of the total and of each spin density, with the same centered
finite-difference first derivative ``D`` of the active grid, extended by
zero outside the domain, that :mod:`parsec_python.V_xc.pbe` uses.

Densities are in electrons/bohr**3.  Pointwise functions return energies per
volume in Hartree atomic units; :func:`b3lyp_semilocal_energies` integrates
them and converts to the Rydberg convention of the solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import correlate1d

from ..Grid import RealSpaceGrid
from .pbe import _embed_active, _gather_active, first_derivative_coefficients


# Mixing coefficients of B3LYP (Stephens, Devlin, Chabalowski, Frisch 1994).
B3LYP_EXACT_EXCHANGE = 0.20
B3LYP_SLATER_EXCHANGE = 0.08
B3LYP_B88_EXCHANGE = 0.72
B3LYP_VWN_CORRELATION = 0.19
B3LYP_LYP_CORRELATION = 0.81

# Slater (Dirac) exchange of the unpolarized gas: e_x = -C_x n**(4/3).
_SLATER_CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
# Becke 1988: beta of the gradient correction, fitted to the rare-gas atoms.
_B88_BETA = 0.0042
# Lee-Yang-Parr 1988 (Colle-Salvetti) constants.
_LYP_A = 0.04918
_LYP_B = 0.132
_LYP_C = 0.2533
_LYP_D = 0.349
_LYP_CF = 0.3 * (3.0 * np.pi * np.pi) ** (2.0 / 3.0)
# Vosko-Wilk-Nusair 1980, paramagnetic gas, Hartree: (A, x0, b, c).  "rpa"
# is their fit to the RPA correlation energy (the B3LYP of Gaussian, libxc
# and ARES), "vwn5" their fit to the Ceperley-Alder Monte Carlo energies.
_VWN_PARAMETERS = {
    "rpa": (0.0310907, -0.409286, 13.0720, 42.7198),
    "vwn5": (0.0310907, -0.10498, 3.72744, 12.9352),
}
# The same fits for the ferromagnetic (fully polarized) gas, and the fit of
# the spin stiffness alpha_c that the Monte Carlo functional interpolates
# with; f''(0) of the interpolation function.
_VWN_FERROMAGNETIC = {
    "rpa": (0.01554535, -0.743294, 20.1231, 101.578),
    "vwn5": (0.01554535, -0.32500, 7.06042, 18.0578),
}
_VWN_SPIN_STIFFNESS = (-1.0 / (6.0 * np.pi * np.pi), -0.0047584, 1.13107, 13.0045)
_VWN_F_SECOND = 4.0 / (9.0 * (2.0 ** (1.0 / 3.0) - 1.0))
# The vacuum branch of pbe.py: below it every energy density is zero.
_DENSITY_THRESHOLD = 1.0e-20


def _checked(density: np.ndarray, sigma: np.ndarray | None = None):
    density = np.asarray(density, dtype=np.float64)
    if np.any(density < -1.0e-14):
        raise ValueError("the functional requires a nonnegative density")
    if sigma is None:
        return density, None
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape != density.shape:
        raise ValueError("density and squared gradient must have the same shape")
    if np.any(sigma < -1.0e-14):
        raise ValueError("the functional requires a nonnegative squared gradient")
    return density, np.maximum(sigma, 0.0)


def slater_exchange_energy_density(density: np.ndarray) -> np.ndarray:
    """Slater exchange energy per volume, ``-(3/4)(3/pi)**(1/3) n**(4/3)``."""

    density, _ = _checked(density)
    energy = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    n = density[positive]
    energy[positive] = -_SLATER_CX * n * np.cbrt(n)
    return energy


def b88_exchange_energy_density(density: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Becke 1988 exchange energy per volume, Slater part included.

    For each spin density ``n_s = n/2`` with ``x = |grad n_s| / n_s**(4/3)``,

    ``e_s = -(3/2)(3/(4 pi))**(1/3) n_s**(4/3)
            - beta n_s**(4/3) x**2 / (1 + 6 beta x asinh(x))``,

    and the unpolarized energy density is ``2 e_s``
    (A. D. Becke, Phys. Rev. A 38, 3098 (1988)).
    """

    density, sigma = _checked(density, sigma)
    energy = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    n = density[positive]
    spin_density_43 = (0.5 * n) * np.cbrt(0.5 * n)
    # |grad n_s| = |grad n| / 2.
    x = 0.5 * np.sqrt(sigma[positive]) / spin_density_43
    correction = _B88_BETA * x * x / (1.0 + 6.0 * _B88_BETA * x * np.arcsinh(x))
    energy[positive] = -_SLATER_CX * n * np.cbrt(n) - 2.0 * spin_density_43 * correction
    return energy


def vwn_correlation_energy_density(
    density: np.ndarray,
    parametrization: str = "rpa",
) -> np.ndarray:
    """Vosko-Wilk-Nusair correlation energy per volume of the unpolarized gas.

    With ``x = sqrt(r_s)``, ``X(x) = x**2 + b x + c`` and
    ``Q = sqrt(4c - b**2)`` the energy per electron is (S. H. Vosko, L. Wilk
    and M. Nusair, Can. J. Phys. 58, 1200 (1980), Eq. 4.4)

    ``eps_c = A { ln(x**2/X) + (2b/Q) atan(Q/(2x+b))
                 - (b x0/X(x0)) [ ln((x-x0)**2/X)
                                  + (2(b+2 x0)/Q) atan(Q/(2x+b)) ] }``.

    ``parametrization`` is ``"rpa"`` (the B3LYP of Gaussian, libxc and ARES)
    or ``"vwn5"``.
    """

    try:
        parameters = _VWN_PARAMETERS[parametrization]
    except KeyError:
        raise ValueError("VWN parametrization must be 'rpa' or 'vwn5'") from None
    density, _ = _checked(density)
    energy = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    n = density[positive]
    x = np.sqrt(np.cbrt(3.0 / (4.0 * np.pi * n)))
    energy[positive] = n * _vwn_epsilon(x, parameters)
    return energy


def _vwn_epsilon(x: np.ndarray, parameters: tuple[float, float, float, float]) -> np.ndarray:
    """VWN Eq. 4.4 for one set ``(A, x0, b, c)``, with ``x = sqrt(r_s)``."""

    a, x0, b, c = parameters
    big_x = x * x + b * x + c
    big_x0 = x0 * x0 + b * x0 + c
    q = np.sqrt(4.0 * c - b * b)
    arctangent = np.arctan(q / (2.0 * x + b))
    return a * (
        np.log(x * x / big_x)
        + (2.0 * b / q) * arctangent
        - (b * x0 / big_x0)
        * (
            np.log((x - x0) * (x - x0) / big_x)
            + (2.0 * (b + 2.0 * x0) / q) * arctangent
        )
    )


def _spin_checked(density_up: np.ndarray, density_down: np.ndarray):
    up, _ = _checked(density_up)
    down, _ = _checked(density_down)
    if up.shape != down.shape:
        raise ValueError("the two spin densities must have the same shape")
    return np.maximum(up, 0.0), np.maximum(down, 0.0)


def slater_exchange_energy_density_spin(
    density_up: np.ndarray, density_down: np.ndarray
) -> np.ndarray:
    """Slater exchange energy per volume of two spin densities.

    Exchange obeys ``E_x[n_up, n_down] = (E_x[2 n_up] + E_x[2 n_down]) / 2``
    with the unpolarized functional on the right.
    """

    up, down = _spin_checked(density_up, density_down)
    return 0.5 * (
        slater_exchange_energy_density(2.0 * up)
        + slater_exchange_energy_density(2.0 * down)
    )


def b88_exchange_energy_density_spin(
    density_up: np.ndarray,
    density_down: np.ndarray,
    sigma_up: np.ndarray,
    sigma_down: np.ndarray,
) -> np.ndarray:
    """Becke 1988 exchange energy per volume of two spin densities.

    ``sigma_up`` and ``sigma_down`` are ``|grad n_up|**2`` and
    ``|grad n_down|**2``.  The spin scaling of the exchange turns each
    channel into an unpolarized density ``2 n_s`` with gradient
    ``4 sigma_ss``.
    """

    up, down = _spin_checked(density_up, density_down)
    return 0.5 * (
        b88_exchange_energy_density(2.0 * up, 4.0 * np.asarray(sigma_up))
        + b88_exchange_energy_density(2.0 * down, 4.0 * np.asarray(sigma_down))
    )


def vwn_correlation_energy_density_spin(
    density_up: np.ndarray,
    density_down: np.ndarray,
    parametrization: str = "rpa",
) -> np.ndarray:
    """Vosko-Wilk-Nusair correlation energy per volume of two spin densities.

    With ``zeta = (n_up - n_down)/n`` and
    ``f(zeta) = [(1+zeta)**(4/3) + (1-zeta)**(4/3) - 2] / (2**(4/3) - 2)``:

    * ``"rpa"``, the local correlation of B3LYP, interpolates the RPA fits
      of the paramagnetic and the ferromagnetic gas with ``f`` alone,
      ``eps = eps_P + (eps_F - eps_P) f(zeta)`` (``LDA_C_VWN_RPA`` of libxc,
      the VWN of Gaussian's B3LYP);
    * ``"vwn5"`` is their Eq. 4.15 with the fit of the spin stiffness,
      ``eps = eps_P + alpha_c f (1 - zeta**4)/f''(0)
      + (eps_F - eps_P) f zeta**4``.
    """

    if parametrization not in _VWN_PARAMETERS:
        raise ValueError("VWN parametrization must be 'rpa' or 'vwn5'")
    up, down = _spin_checked(density_up, density_down)
    density = up + down
    energy = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    n = density[positive]
    zeta = (up[positive] - down[positive]) / n
    f = (
        np.maximum(1.0 + zeta, 0.0) ** (4.0 / 3.0)
        + np.maximum(1.0 - zeta, 0.0) ** (4.0 / 3.0)
        - 2.0
    ) / (2.0 ** (4.0 / 3.0) - 2.0)
    x = np.sqrt(np.cbrt(3.0 / (4.0 * np.pi * n)))
    paramagnetic = _vwn_epsilon(x, _VWN_PARAMETERS[parametrization])
    ferromagnetic = _vwn_epsilon(x, _VWN_FERROMAGNETIC[parametrization])
    if parametrization == "rpa":
        epsilon = paramagnetic + (ferromagnetic - paramagnetic) * f
    else:
        z4 = zeta**4
        epsilon = (
            paramagnetic
            + _vwn_epsilon(x, _VWN_SPIN_STIFFNESS) * f * (1.0 - z4) / _VWN_F_SECOND
            + (ferromagnetic - paramagnetic) * f * z4
        )
    energy[positive] = n * epsilon
    return energy


def lyp_correlation_energy_density_spin(
    density_up: np.ndarray,
    density_down: np.ndarray,
    sigma_up: np.ndarray,
    sigma_down: np.ndarray,
    sigma_total: np.ndarray,
) -> np.ndarray:
    """Lee-Yang-Parr correlation energy per volume of two spin densities.

    Eq. 2 of Miehlich, Savin, Stoll and Preuss with ``a`` and ``b`` the two
    spin densities, ``n`` their sum and the squared gradients of all three:

    ``e_c = -4 a_LYP a b / (n (1 + d n**(-1/3)))
            - a_LYP b_LYP omega { a b [ 2**(11/3) C_F (a**(8/3) + b**(8/3))
                + (47/18 - 7 delta/18) |grad n|**2
                - (5/2 - delta/18) (|grad a|**2 + |grad b|**2)
                - (delta - 11)/9 (a |grad a|**2 + b |grad b|**2)/n ]
              - (2/3) n**2 |grad n|**2
              + (2/3 n**2 - a**2) |grad b|**2
              + (2/3 n**2 - b**2) |grad a|**2 }``,

    ``omega = exp(-c n**(-1/3)) n**(-11/3) / (1 + d n**(-1/3))``.  A channel
    without electrons has no LYP correlation.
    """

    up, down = _spin_checked(density_up, density_down)
    gradients = [np.asarray(value, dtype=np.float64) for value in (sigma_up, sigma_down, sigma_total)]
    if any(value.shape != up.shape for value in gradients):
        raise ValueError("densities and squared gradients must have the same shape")
    if any(np.any(value < -1.0e-14) for value in gradients):
        raise ValueError("the functional requires nonnegative squared gradients")
    density = up + down
    energy = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    n = density[positive]
    a = up[positive]
    b = down[positive]
    grad_a, grad_b, grad_n = (np.maximum(value[positive], 0.0) for value in gradients)
    inverse_cube_root = 1.0 / np.cbrt(n)
    denominator = 1.0 + _LYP_D * inverse_cube_root
    delta = _LYP_C * inverse_cube_root + _LYP_D * inverse_cube_root / denominator
    omega = (
        np.exp(-_LYP_C * inverse_cube_root) / denominator * inverse_cube_root**11
    )
    bracket = (
        2.0 ** (11.0 / 3.0) * _LYP_CF * (a ** (8.0 / 3.0) + b ** (8.0 / 3.0))
        + (47.0 / 18.0 - 7.0 * delta / 18.0) * grad_n
        - (2.5 - delta / 18.0) * (grad_a + grad_b)
        - (delta - 11.0) / 9.0 * (a * grad_a + b * grad_b) / n
    )
    energy[positive] = -_LYP_A * 4.0 / denominator * a * b / n - _LYP_A * _LYP_B * omega * (
        a * b * bracket
        - (2.0 / 3.0) * n * n * grad_n
        + ((2.0 / 3.0) * n * n - a * a) * grad_b
        + ((2.0 / 3.0) * n * n - b * b) * grad_a
    )
    return energy


def lyp_correlation_energy_density(density: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Lee-Yang-Parr correlation energy per volume of a closed shell.

    The gradient-only form of B. Miehlich, A. Savin, H. Stoll and H. Preuss,
    Chem. Phys. Lett. 157, 200 (1989), Eq. 2, with ``n_up = n_down = n/2``
    and ``|grad n_up|**2 = |grad n_down|**2 = sigma/4`` reduces to

    ``e_c = -a n / (1 + d n**(-1/3))
            - a b exp(-c n**(-1/3)) / (1 + d n**(-1/3))
              * [ C_F n - n**(-5/3) sigma (1/24 + 7 delta/72) ]``,

    ``delta = c n**(-1/3) + d n**(-1/3) / (1 + d n**(-1/3))`` and
    ``C_F = (3/10)(3 pi**2)**(2/3)``.
    """

    density, sigma = _checked(density, sigma)
    energy = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    n = density[positive]
    inverse_cube_root = 1.0 / np.cbrt(n)
    denominator = 1.0 + _LYP_D * inverse_cube_root
    delta = _LYP_C * inverse_cube_root + _LYP_D * inverse_cube_root / denominator
    gradient_term = (
        inverse_cube_root**5 * sigma[positive] * (1.0 / 24.0 + 7.0 * delta / 72.0)
    )
    energy[positive] = (
        -_LYP_A * n / denominator
        - _LYP_A
        * _LYP_B
        * np.exp(-_LYP_C * inverse_cube_root)
        / denominator
        * (_LYP_CF * n - gradient_term)
    )
    return energy


def density_gradient_squared(density: np.ndarray, grid: RealSpaceGrid) -> np.ndarray:
    """Return ``sigma = |D n|**2`` on the active grid.

    ``D`` is the centered first derivative of the order of the grid, with the
    density extended by zero outside the active domain: the gradient of
    :func:`parsec_python.V_xc.pbe`.
    """

    density = np.asarray(density, dtype=np.float64)
    if density.shape != (grid.size,):
        raise ValueError("density does not match the active real-space grid")
    weights = first_derivative_coefficients(grid.settings.expansion_order)
    weights = weights / grid.spacing
    full_density = _embed_active(density, grid)
    sigma = np.zeros(grid.size, dtype=np.float64)
    for axis in range(3):
        derivative_box = correlate1d(
            full_density, weights, axis=axis, mode="constant", cval=0.0
        )
        gradient = _gather_active(derivative_box, grid)
        sigma += gradient * gradient
    return sigma


@dataclass(frozen=True)
class B3LYPSemilocalEnergies:
    """Integrated density-functional parts of B3LYP, in Rydberg.

    ``b88_exchange`` is the complete B88 exchange, Slater part included.
    ``vwn_correlation`` belongs to ``vwn_parametrization`` (``"rpa"`` for
    B3LYP).
    """

    slater_exchange: float
    b88_exchange: float
    vwn_correlation: float
    lyp_correlation: float
    vwn_parametrization: str = "rpa"


def b3lyp_semilocal_energies(
    valence_density: np.ndarray,
    grid: RealSpaceGrid,
    core_density: np.ndarray | None = None,
    *,
    vwn_parametrization: str = "rpa",
) -> B3LYPSemilocalEnergies:
    """Integrate Slater, B88, VWN and LYP for one density of the grid.

    ``core_density`` is the frozen nonlinear-core-correction density.  It is
    added to the valence density exactly as the self-consistent functional
    adds it (:func:`parsec_python.V_xc.pbe`), so that the refinement replaces
    the exchange-correlation energy of one and the same density.
    """

    valence = np.asarray(valence_density, dtype=np.float64)
    if valence.shape != (grid.size,):
        raise ValueError("density does not match the active real-space grid")
    if core_density is None:
        density = valence
    else:
        core = np.asarray(core_density, dtype=np.float64)
        if core.shape != valence.shape:
            raise ValueError("core and valence densities must have the same shape")
        density = valence + core
    if np.any(density < -1.0e-14):
        raise ValueError("the functional requires a nonnegative density")
    density = np.maximum(density, 0.0)
    sigma = density_gradient_squared(density, grid)
    # Hartree -> Rydberg at the public boundary, as in pbe.py.
    scale = 2.0 * grid.volume_element

    def integral(values: np.ndarray) -> float:
        return float(scale * np.sum(values))

    return B3LYPSemilocalEnergies(
        slater_exchange=integral(slater_exchange_energy_density(density)),
        b88_exchange=integral(b88_exchange_energy_density(density, sigma)),
        vwn_correlation=integral(
            vwn_correlation_energy_density(density, vwn_parametrization)
        ),
        lyp_correlation=integral(lyp_correlation_energy_density(density, sigma)),
        vwn_parametrization=vwn_parametrization,
    )


def b3lyp_spin_semilocal_energies(
    valence_up: np.ndarray,
    valence_down: np.ndarray,
    grid: RealSpaceGrid,
    core_density: np.ndarray | None = None,
    *,
    vwn_parametrization: str = "rpa",
) -> B3LYPSemilocalEnergies:
    """Integrate Slater, B88, VWN and LYP for two spin densities of the grid.

    Half of the frozen core density is added to each channel, as the
    self-consistent spin functionals add it
    (:func:`parsec_python.V_xc.pbe_spin`).  For equal spin densities the
    energies are those of :func:`b3lyp_semilocal_energies`.
    """

    up = np.asarray(valence_up, dtype=np.float64)
    down = np.asarray(valence_down, dtype=np.float64)
    if up.shape != (grid.size,) or down.shape != (grid.size,):
        raise ValueError("spin densities do not match the active real-space grid")
    if core_density is not None:
        core = np.asarray(core_density, dtype=np.float64)
        if core.shape != up.shape:
            raise ValueError("core and valence densities must have the same shape")
        up = up + 0.5 * core
        down = down + 0.5 * core
    if np.any(up < -1.0e-14) or np.any(down < -1.0e-14):
        raise ValueError("the functional requires nonnegative spin densities")
    up = np.maximum(up, 0.0)
    down = np.maximum(down, 0.0)
    sigma_up = density_gradient_squared(up, grid)
    sigma_down = density_gradient_squared(down, grid)
    # The gradient is linear: this is |D n_up + D n_down|**2.
    sigma_total = density_gradient_squared(up + down, grid)
    scale = 2.0 * grid.volume_element

    def integral(values: np.ndarray) -> float:
        return float(scale * np.sum(values))

    return B3LYPSemilocalEnergies(
        slater_exchange=integral(slater_exchange_energy_density_spin(up, down)),
        b88_exchange=integral(
            b88_exchange_energy_density_spin(up, down, sigma_up, sigma_down)
        ),
        vwn_correlation=integral(
            vwn_correlation_energy_density_spin(up, down, vwn_parametrization)
        ),
        lyp_correlation=integral(
            lyp_correlation_energy_density_spin(
                up, down, sigma_up, sigma_down, sigma_total
            )
        ),
        vwn_parametrization=vwn_parametrization,
    )


def b3lyp_exchange_correlation_energy(
    exact_exchange: float,
    semilocal: B3LYPSemilocalEnergies,
) -> float:
    """Combine the five parts with the B3LYP weights (same unit as the input)."""

    if semilocal.vwn_parametrization != "rpa":
        raise ValueError(
            "B3LYP is defined with the RPA parametrization of VWN, not "
            f"{semilocal.vwn_parametrization!r}"
        )
    return float(
        B3LYP_EXACT_EXCHANGE * exact_exchange
        + B3LYP_SLATER_EXCHANGE * semilocal.slater_exchange
        + B3LYP_B88_EXCHANGE * semilocal.b88_exchange
        + B3LYP_VWN_CORRELATION * semilocal.vwn_correlation
        + B3LYP_LYP_CORRELATION * semilocal.lyp_correlation
    )


__all__ = [
    "B3LYP_B88_EXCHANGE",
    "B3LYP_EXACT_EXCHANGE",
    "B3LYP_LYP_CORRELATION",
    "B3LYP_SLATER_EXCHANGE",
    "B3LYP_VWN_CORRELATION",
    "B3LYPSemilocalEnergies",
    "b3lyp_exchange_correlation_energy",
    "b3lyp_semilocal_energies",
    "b3lyp_spin_semilocal_energies",
    "b88_exchange_energy_density",
    "b88_exchange_energy_density_spin",
    "density_gradient_squared",
    "lyp_correlation_energy_density",
    "lyp_correlation_energy_density_spin",
    "slater_exchange_energy_density",
    "slater_exchange_energy_density_spin",
    "vwn_correlation_energy_density",
    "vwn_correlation_energy_density_spin",
]
