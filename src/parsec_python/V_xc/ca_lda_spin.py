"""Spin-polarized CA/PZ local-density exchange and correlation.

This is the ``Correlation_Type=ca`` branch of PARSEC's ``exc_spn.f90``: the
local spin density approximation with Dirac exchange for each spin channel
and the Ceperley-Alder correlation in the Perdew-Zunger parametrization,
interpolated between the unpolarized and the fully polarized electron gas
with the von Barth-Hedin function

``f(zeta) = [(1+zeta)**(4/3) + (1-zeta)**(4/3) - 2] / (2**(4/3) - 2)``,

``zeta = (rho_up - rho_down) / (rho_up + rho_down)``.

Each spin channel sees its own density plus half of the frozen core density
of a nonlinear core correction (``rho_hart.F90``).  Densities are in
electrons/bohr^3, energies and potentials in Rydberg.

One point of the Fortran is not translated.  ``exc_spn.f90`` evaluates the
functional only where both spin densities are positive and leaves its work
array, which holds the densities, where one of them is zero: the potentials
of such a point are then the densities themselves and the point adds nothing
to the energy.  A channel without electrons (the down channel of a hydrogen
atom) is zero everywhere, and that calculation does not converge in
Fortran.  Here the functional is evaluated wherever the total density is
positive; a channel whose density vanishes has ``zeta = +-1`` and no
exchange energy of its own, which is the limit of the formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Constants of exc_spn.f90, with the digits of the source.  ``_AX`` is the
# exchange coefficient -(3/4)(3/pi)**(1/3) in Hartree; the correlation
# parameters are Perdew and Zunger's in Rydberg.
_AX = -0.738558766382022405884230032680836
_FOUR_THIRDS = 4.0 / 3.0
_GAMMA = 0.5198421
# Unpolarized gas: high density (r_s < 1), then low density.
_C1, _C2, _C3, _C4 = 0.0622, 0.0960, 0.0040, 0.0232
_G, _B1, _B2 = -0.2846, 1.0529, 0.3334
# Fully polarized gas.
_AS, _BS, _CS, _DS = 0.0311, -0.0538, 0.0014, -0.0096
_GS, _B1S, _B2S = -0.1686, 1.3981, 0.2611


@dataclass(frozen=True)
class SpinXCResult:
    """Spin-resolved exchange-correlation potentials and their energy.

    ``potential_up`` and ``potential_down`` are the functional derivatives
    of the energy with respect to the density of each channel and go on the
    diagonal of the Hamiltonian of that channel.  ``energy_density`` is the
    energy per volume in Rydberg/bohr^3 and ``total_energy`` its integral
    over the grid in Rydberg.
    """

    potential_up: np.ndarray
    potential_down: np.ndarray
    energy_density: np.ndarray
    total_energy: float

    @property
    def potentials(self) -> tuple[np.ndarray, np.ndarray]:
        """The two potentials, up first."""

        return (self.potential_up, self.potential_down)


def spin_densities_with_core(
    valence_up: np.ndarray,
    valence_down: np.ndarray,
    core_density: np.ndarray | None,
    *,
    functional: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two densities a spin functional is evaluated at.

    ``rho_hart.F90`` adds half of the frozen core density to each channel.
    A density below ``-1e-14`` is refused, as the unpolarized functionals
    do; roundoff below zero is set to zero.
    """

    up = np.asarray(valence_up, dtype=np.float64)
    down = np.asarray(valence_down, dtype=np.float64)
    if up.shape != down.shape:
        raise ValueError("the two spin densities must have the same shape")
    if core_density is not None:
        core = np.asarray(core_density, dtype=np.float64)
        if core.shape != up.shape:
            raise ValueError("core and valence densities must have the same shape")
        up = up + 0.5 * core
        down = down + 0.5 * core
    if np.any(up < -1.0e-14) or np.any(down < -1.0e-14):
        raise ValueError(f"{functional} requires nonnegative spin densities")
    return np.maximum(up, 0.0), np.maximum(down, 0.0)


def ca_lda_spin(
    valence_up: np.ndarray,
    valence_down: np.ndarray,
    volume_element: float,
    core_density: np.ndarray | None = None,
) -> SpinXCResult:
    """Evaluate the CA/PZ local spin density functional on a uniform grid.

    With ``rho = rho_up + rho_down`` and ``r_s = [3/(4 pi rho)]**(1/3)``,

    ``E_xc = dV * sum[ 2 (rho_up e_x(2 rho_up) + rho_down e_x(2 rho_down))
                       + rho (e_u + f(zeta) (e_p - e_u)) ]``,

    ``e_x(d) = -(3/4)(3/pi)**(1/3) d**(1/3)`` in Hartree, hence the factor
    two, and ``e_u``, ``e_p`` the Perdew-Zunger correlation energies per
    electron of the unpolarized and the fully polarized gas in Rydberg.  The
    potentials are

    ``V_sigma = 2 (4/3) e_x(2 rho_sigma) + v_u + f (v_p - v_u)
               +- (e_p - e_u) (1 -+ zeta) f'(zeta)``

    with the upper signs for the up channel.
    """

    up, down = spin_densities_with_core(
        valence_up, valence_down, core_density, functional="CA-LDA"
    )
    potential_up = np.zeros_like(up)
    potential_down = np.zeros_like(up)
    energy_density = np.zeros_like(up)
    density = up + down
    positive = density > 0.0
    if np.any(positive):
        rho_up = up[positive]
        rho_down = down[positive]
        rho = density[positive]

        # Exchange of each channel is that of an unpolarized gas of twice
        # its density.  ``_AX`` is in Hartree; the factors two below convert.
        exchange_up = _AX * np.cbrt(2.0 * rho_up)
        exchange_down = _AX * np.cbrt(2.0 * rho_down)

        rs = np.cbrt((0.75 / np.pi) / rho)
        zeta = (rho_up - rho_down) / rho
        # Roundoff must not carry 1 +- zeta below zero under the roots.
        plus = np.maximum(1.0 + zeta, 0.0)
        minus = np.maximum(1.0 - zeta, 0.0)
        interpolation = (
            plus**_FOUR_THIRDS + minus**_FOUR_THIRDS - 2.0
        ) / _GAMMA
        interpolation_slope = _FOUR_THIRDS * (np.cbrt(plus) - np.cbrt(minus)) / _GAMMA

        unpolarized = np.empty_like(rs)
        unpolarized_potential = np.empty_like(rs)
        polarized = np.empty_like(rs)
        polarized_potential = np.empty_like(rs)
        high_density = rs < 1.0
        if np.any(high_density):
            r = rs[high_density]
            log_r = np.log(r)
            eu = _C1 * log_r - _C2 + (_C3 * log_r - _C4) * r
            unpolarized[high_density] = eu
            unpolarized_potential[high_density] = (
                eu - (_C1 + (_C3 * log_r + _C3 - _C4) * r) / 3.0
            )
            es = _AS * log_r + _BS + (_CS * log_r + _DS) * r
            polarized[high_density] = es
            polarized_potential[high_density] = (
                es - (_AS + (_CS * log_r + _CS + _DS) * r) / 3.0
            )
        low_density = ~high_density
        if np.any(low_density):
            r = rs[low_density]
            sqrt_r = np.sqrt(r)
            eu = _G / (1.0 + _B1 * sqrt_r + _B2 * r)
            unpolarized[low_density] = eu
            unpolarized_potential[low_density] = (
                eu * eu * (1.0 + 7.0 * _B1 * sqrt_r / 6.0 + 4.0 / 3.0 * _B2 * r) / _G
            )
            es = _GS / (1.0 + _B1S * sqrt_r + _B2S * r)
            polarized[low_density] = es
            polarized_potential[low_density] = (
                es * es * (1.0 + 7.0 * _B1S * sqrt_r / 6.0 + 4.0 / 3.0 * _B2S * r) / _GS
            )

        mixed = unpolarized_potential + interpolation * (
            polarized_potential - unpolarized_potential
        )
        difference = polarized - unpolarized
        correlation_up = mixed + difference * (1.0 - zeta) * interpolation_slope
        correlation_down = mixed - difference * (1.0 + zeta) * interpolation_slope
        correlation = unpolarized + interpolation * difference

        potential_up[positive] = 2.0 * _FOUR_THIRDS * exchange_up + correlation_up
        potential_down[positive] = 2.0 * _FOUR_THIRDS * exchange_down + correlation_down
        energy_density[positive] = (
            2.0 * (rho_up * exchange_up + rho_down * exchange_down)
            + correlation * rho
        )
    return SpinXCResult(
        potential_up=potential_up,
        potential_down=potential_down,
        energy_density=energy_density,
        total_energy=float(np.sum(energy_density) * volume_element),
    )


__all__ = ["SpinXCResult", "ca_lda_spin", "spin_densities_with_core"]
