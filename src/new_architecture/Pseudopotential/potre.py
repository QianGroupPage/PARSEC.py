"""Reader and radial operations for PARSEC ``*_POTRE.DAT`` files.

The supported layout is the ``MARTINS_NEW``/``potre`` format read by
``pseudo.f90``.  The file stores ``r*V_l(r)`` and radial charge distributions;
this module converts them to the volume-density and potential conventions used
by the rest of the Python solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .radial_quadrature import parsec_radial_integral
from .radial_spline import ParsecRadialSpline


_FLOAT_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
)


def _numbers(text: str) -> list[float]:
    return [float(value.replace("D", "E").replace("d", "e")) for value in _FLOAT_RE.findall(text)]


def _read_values(lines: list[str], start: int, count: int) -> tuple[np.ndarray, int]:
    values: list[float] = []
    index = start
    while len(values) < count and index < len(lines):
        values.extend(_numbers(lines[index]))
        index += 1
    if len(values) != count:
        raise ValueError(f"expected {count} radial values, found {len(values)}")
    return np.asarray(values, dtype=float), index


@dataclass(frozen=True)
class ParsecPseudopotential:
    """One norm-conserving PARSEC pseudopotential.

    Attributes use PARSEC units: bohr for radii and Rydberg for potentials.
    ``radial_wavefunctions`` contain the file's reduced radial functions
    ``u_l(r)``.  Densities are converted to electrons/bohr**3.
    """

    symbol: str
    correlation: str
    relativity: str
    core_marker: str
    comment: str
    configuration: str
    number_of_channels: int
    number_of_spin_orbit_channels: int
    radial_parameter_a: float
    radial_parameter_b: float
    ionic_charge: float
    radii: np.ndarray
    channel_potentials: dict[int, np.ndarray]
    valence_density: np.ndarray
    core_density: np.ndarray
    radial_wavefunctions: dict[int, np.ndarray]
    channel_occupations: dict[int, float]
    channel_cutoffs: dict[int, float]
    source: Path

    @property
    def has_nonlinear_core_correction(self) -> bool:
        return self.core_marker.strip().lower() not in {"nc", "none", "no"}

    @property
    def interpolation_cutoff(self) -> float:
        """Largest radius used by PARSEC before analytic/zero continuation."""
        if self.radii.size < 2:
            return float(self.radii[-1])
        return float(self.radii[-2])

    def validate_local_channel(self, local_l: int) -> None:
        if local_l not in self.channel_potentials:
            available = ", ".join(str(value) for value in sorted(self.channel_potentials))
            raise ValueError(
                f"local channel l={local_l} is absent from {self.source}; "
                f"available channels: {available}"
            )

    def local_potential(
        self,
        distances: np.ndarray,
        local_l: int,
        *,
        use_spline: bool = False,
        spline_padding_width: int | None = None,
    ) -> np.ndarray:
        """Evaluate the selected local potential with PARSEC's Coulomb tail.

        When ``use_spline`` is true, ``spline_padding_width`` must equal the
        real-space stencil half-width ``Expansion_Order/2``.  PARSEC uses it
        to extend the radial spline through the atomic origin.
        """
        self.validate_local_channel(local_l)
        radius = np.asarray(distances, dtype=float)
        if np.any(radius < 0):
            raise ValueError("radial distances cannot be negative")
        values = self.channel_potentials[local_l]
        result = np.empty_like(radius)
        # ``ionpot.f90`` switches at ``dist >= rs(ns-1)``.
        inside = radius < self.interpolation_cutoff
        if use_spline:
            if spline_padding_width is None:
                raise ValueError(
                    "spline_padding_width is required when use_spline=True"
                )
            if np.any(inside):
                spline = ParsecRadialSpline.from_positive_grid(
                    self.radii,
                    values,
                    spline_padding_width,
                )
                result[inside] = spline(radius[inside])
        else:
            near_zero = inside & (radius <= self.radii[0])
            result[near_zero] = values[0]
            interpolate = inside & ~near_zero
            if np.any(interpolate):
                sample_r = radius[interpolate]
                radial_potential = self.radii * values
                interpolated = np.interp(sample_r, self.radii, radial_potential)
                result[interpolate] = interpolated / sample_r

        outside = ~inside
        if np.any(outside):
            result[outside] = -2.0 * self.ionic_charge / radius[outside]
        return result

    def interpolate_density(
        self,
        distances: np.ndarray,
        *,
        core: bool = False,
        use_spline: bool = False,
        spline_padding_width: int | None = None,
    ) -> np.ndarray:
        """Interpolate a stored atomic volume density and set its tail to zero.

        The optional spline uses PARSEC's origin padding, whose width is the
        finite-difference stencil half-width ``Expansion_Order/2``.
        """
        radius = np.asarray(distances, dtype=float)
        values = self.core_density if core else self.valence_density
        result = np.zeros_like(radius)
        # ``initchrg.f90``/``corecd.f90`` drop the density at and beyond
        # ``rs(ns-1)``.
        inside = radius < self.interpolation_cutoff
        if use_spline and spline_padding_width is None:
            raise ValueError(
                "spline_padding_width is required when use_spline=True"
            )
        if not np.any(inside):
            return result
        sample = radius[inside]
        if use_spline:
            spline = ParsecRadialSpline.from_positive_grid(
                self.radii,
                values,
                spline_padding_width,
            )
            result[inside] = spline(sample)
        else:
            result[inside] = np.interp(sample, self.radii, values, left=values[0])
        return result

    def radial_projector(self, angular_momentum: int, local_l: int) -> tuple[np.ndarray, float]:
        """Return PARSEC's normalized KB radial projector and denominator sign.

        The unnormalized projector is
        ``(V_l - V_local) * u_l / r``.  PARSEC divides it by
        ``sqrt(abs(integral u_l**2 * (V_l - V_local) dr))`` and stores the
        denominator sign separately.
        """
        self.validate_local_channel(local_l)
        if angular_momentum == local_l:
            raise ValueError("the selected local channel has no KB projector")
        if angular_momentum not in self.radial_wavefunctions:
            raise ValueError(f"no pseudo-wavefunction for l={angular_momentum}")
        if angular_momentum not in self.channel_potentials:
            raise ValueError(f"no potential channel for l={angular_momentum}")

        delta_v = (
            self.channel_potentials[angular_momentum]
            - self.channel_potentials[local_l]
        )
        radial_wave = self.radial_wavefunctions[angular_momentum]
        denominator = parsec_radial_integral(
            self.radii,
            radial_wave * radial_wave * delta_v,
        )
        if abs(denominator) < 1.0e-18:
            raise ValueError(f"nearly zero Kleinman-Bylander denominator for l={angular_momentum}")
        projector = delta_v * radial_wave / self.radii
        projector /= np.sqrt(abs(denominator))
        # PARSEC's 18th-order radial derivative setup (mor=9) replaces
        # internal indices through mor with the value at 2*mor.  POTRE file
        # index zero is internal index two, so this is Python 0:8 <- 16.
        if projector.size > 16:
            projector[:8] = projector[16]
        return projector, float(np.sign(denominator))


def read_parsec_pseudopotential(path: str | Path) -> ParsecPseudopotential:
    """Read a PARSEC ``MARTINS_NEW``/``*_POTRE.DAT`` pseudopotential."""
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise ValueError(f"{source} is too short to be a PARSEC potential")

    identity = lines[0].split()
    if len(identity) < 4:
        raise ValueError(f"invalid PARSEC pseudopotential identity line in {source}")
    symbol, correlation, relativity, core_marker = identity[:4]

    header = _numbers(lines[3])
    if len(header) < 6:
        raise ValueError(f"invalid numeric pseudopotential header in {source}")
    number_of_channels = int(header[0])
    number_of_spin_orbit_channels = int(header[1])
    number_of_grid_points = int(header[2])
    radial_parameter_a, radial_parameter_b, ionic_charge = header[3:6]
    # POTRE stores ``number_of_grid_points`` positive-radius samples.  PARSEC
    # inserts an additional origin entry internally and then reads indices
    # 2..ns, hence the easy-to-misread Fortran bounds.
    radial_count = number_of_grid_points

    marker = next(
        (index for index, line in enumerate(lines) if "radial grid follows" in line.lower()),
        None,
    )
    if marker is None:
        raise ValueError(f"missing radial grid marker in {source}")
    radii, _ = _read_values(lines, marker + 1, radial_count)
    if np.any(radii <= 0) or np.any(np.diff(radii) <= 0):
        raise ValueError(f"radial grid in {source} must be positive and increasing")

    channel_potentials: dict[int, np.ndarray] = {}
    radial_wavefunctions: dict[int, np.ndarray] = {}
    channel_occupations: dict[int, float] = {}
    channel_cutoffs: dict[int, float] = {}
    valence_radial = np.zeros(radial_count)
    core_radial = np.zeros(radial_count)

    for index, line in enumerate(lines):
        lower = line.lower()
        compact_marker = re.sub(r"[^a-z]", "", lower)
        if "pseudopotential follows" in lower:
            l_values = _numbers(lines[index + 1])
            if not l_values:
                raise ValueError(f"missing angular momentum after line {index + 1} in {source}")
            angular_momentum = int(l_values[0])
            radial_v, _ = _read_values(lines, index + 2, radial_count)
            channel_potentials[angular_momentum] = radial_v / radii
        elif "core charge follows" in lower:
            core_radial, _ = _read_values(lines, index + 1, radial_count)
        elif "valence charge follows" in lower:
            valence_radial, _ = _read_values(lines, index + 1, radial_count)
        elif "pseudowavefunctionfollows" in compact_marker:
            wave_header = _numbers(lines[index + 1])
            if len(wave_header) < 3:
                raise ValueError(f"invalid pseudo-wavefunction header in {source}")
            angular_momentum = int(wave_header[0])
            channel_occupations[angular_momentum] = float(wave_header[1])
            channel_cutoffs[angular_momentum] = float(wave_header[2])
            wave, _ = _read_values(lines, index + 2, radial_count)
            radial_wavefunctions[angular_momentum] = wave

    if len(channel_potentials) != number_of_channels:
        raise ValueError(
            f"{source} declares {number_of_channels} channels but contains "
            f"{len(channel_potentials)}"
        )
    missing_waves = sorted(
        set(channel_potentials).difference(radial_wavefunctions)
    )
    if missing_waves:
        channels = ", ".join(f"l={value}" for value in missing_waves)
        raise ValueError(
            f"{source} is missing pseudo-wavefunctions for potential "
            f"channels {channels}"
        )

    radial_to_volume = 1.0 / (4.0 * np.pi * radii * radii)
    return ParsecPseudopotential(
        symbol=symbol,
        correlation=correlation,
        relativity=relativity,
        core_marker=core_marker,
        comment=lines[1].strip(),
        configuration=lines[2].strip(),
        number_of_channels=number_of_channels,
        number_of_spin_orbit_channels=number_of_spin_orbit_channels,
        radial_parameter_a=float(radial_parameter_a),
        radial_parameter_b=float(radial_parameter_b),
        ionic_charge=float(ionic_charge),
        radii=radii,
        channel_potentials=channel_potentials,
        valence_density=valence_radial * radial_to_volume,
        core_density=core_radial * radial_to_volume,
        radial_wavefunctions=radial_wavefunctions,
        channel_occupations=channel_occupations,
        channel_cutoffs=channel_cutoffs,
        source=source,
    )
