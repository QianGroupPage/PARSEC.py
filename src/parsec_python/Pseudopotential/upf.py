"""Norm-conserving pseudopotentials in UPF v2 format (ONCVPSP, PseudoDojo and SG15 files).

A UPF file holds a local potential and, for each angular momentum ``l``, one or more Kleinman-Bylander projectors
``beta_i(r)`` coupled by a matrix ``D_ij``,

    V_NL = sum_ij D_ij |beta_i Y_lm><beta_j Y_lm| ,

all in Rydberg and bohr on a radial mesh of the file (a linear one for PseudoDojo).  PARSEC.py's
:class:`~parsec_python.Pseudopotential.potre.ParsecPseudopotential` keeps one projector per ``l``, derived from
semilocal channel potentials and a pseudo-wavefunction, on the logarithmic grid ``r(i) = a*exp(b*(i-1)) - a``.  This
module reads the file (:func:`read_upf`), brings the projectors to the separable form PARSEC.py uses
(:func:`kleinman_bylander_channels`: ``D`` diagonalised within each ``l``, the radial function ``sqrt|d| beta/r`` and the
sign of ``d``, as ``radial_projector`` returns for a Martins file) and resamples the tables to a PARSEC grid
(:func:`to_parsec_pseudopotential`).

Conventions of the file (UPF v2): ``PP_BETA`` and ``PP_CHI`` hold ``r*beta(r)`` and ``r*R(r)``, ``PP_LOCAL`` the local
potential ``V(r)`` (with the Coulomb tail ``-2 Z/r``), ``PP_RHOATOM`` the valence charge ``4 pi r^2 rho(r)`` and
``PP_NLCC`` the model core charge ``rho_c(r)`` in electrons/bohr**3.

Only scalar-relativistic files are read here; the spin-orbit (``PP_SPIN_ORB``) files of the fully relativistic
PseudoDojo set are refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np
from scipy.interpolate import CubicSpline

from .potre import ParsecPseudopotential

#: Key of the local potential in ``channel_potentials`` of a UPF potential.  A UPF local potential belongs to no
#: angular momentum, so no ``l`` is used for it.
LOCAL_CHANNEL = -1

_ATTRIBUTE = re.compile(r'([A-Za-z_]+)\s*=\s*"([^"]*)"')


@dataclass(frozen=True)
class UpfProjector:
    """One ``PP_BETA``: ``values`` is ``r*beta(r)`` on the mesh of the file (Rydberg^(1/2)-free, bohr^(-1/2))."""

    index: int
    angular_momentum: int
    cutoff_radius: float
    values: np.ndarray


@dataclass(frozen=True)
class UpfWavefunction:
    """One ``PP_CHI``: the reduced pseudo-wavefunction ``u(r) = r*R(r)`` of a valence state."""

    label: str
    angular_momentum: int
    occupation: float
    values: np.ndarray


@dataclass(frozen=True)
class UpfPseudopotential:
    """What a UPF v2 file holds, unconverted (bohr, Rydberg)."""

    symbol: str
    z_valence: float
    functional: str
    relativistic: str
    mesh: np.ndarray
    local_potential: np.ndarray
    projectors: tuple[UpfProjector, ...]
    dij: np.ndarray
    wavefunctions: tuple[UpfWavefunction, ...]
    valence_charge: np.ndarray
    core_density: np.ndarray | None
    source: Path
    info: str = field(default="", compare=False)

    @property
    def maximum_angular_momentum(self) -> int:
        return max(projector.angular_momentum for projector in self.projectors)


@dataclass(frozen=True)
class KleinmanBylanderChannel:
    """A separable projector ``sign * |chi Y_lm><chi Y_lm|`` of the nonlocal operator.

    ``values`` is the radial function ``chi(r) = sqrt|d| * beta(r)`` of PARSEC's tables (the ``R``-like function that
    ``radial_projector`` returns: ``beta(r)`` of the file divided by ``r``) on ``radii``, and ``sign`` the sign of
    the coupling ``d``.  ``index`` counts the projectors of one ``l`` from zero.
    """

    angular_momentum: int
    index: int
    sign: float
    coupling: float
    cutoff_radius: float
    values: np.ndarray


def _attributes(text: str) -> dict[str, str]:
    return {key.lower(): value for key, value in _ATTRIBUTE.findall(text)}


def _floats(body: str) -> np.ndarray:
    return np.array(body.replace("D", "E").replace("d", "e").split(), dtype=float)


def _section(text: str, tag: str) -> tuple[dict[str, str], str] | None:
    match = re.search(rf"<{re.escape(tag)}\b([^>]*)>(.*?)</{re.escape(tag)}>", text, re.S)
    if match is None:
        return None
    return _attributes(match.group(1)), match.group(2)


def read_upf(path: str | Path) -> UpfPseudopotential:
    """Read a norm-conserving, scalar-relativistic UPF v2 file."""

    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    if "<UPF" not in text:
        raise ValueError(f"{source} is not a UPF file")
    header_match = re.search(r"<PP_HEADER\b(.*?)/?>", text, re.S)
    if header_match is None:
        raise ValueError(f"{source} has no PP_HEADER")
    header = _attributes(header_match.group(1))
    if header.get("pseudo_type", "NC").strip().upper() not in {"NC", "SL"}:
        raise ValueError(f"{source}: pseudo_type {header.get('pseudo_type')!r} is not norm-conserving")
    if "<PP_SPIN_ORB" in text or header.get("has_so", "F").strip().upper().startswith(("T", ".T")):
        raise ValueError(f"{source} has spin-orbit projectors; only scalar-relativistic UPF files are read")

    mesh_section = _section(text, "PP_R")
    local_section = _section(text, "PP_LOCAL")
    if mesh_section is None or local_section is None:
        raise ValueError(f"{source} lacks PP_R or PP_LOCAL")
    mesh = _floats(mesh_section[1])
    local_potential = _floats(local_section[1])

    projectors = []
    for match in re.finditer(r"<PP_BETA\.(\d+)\b([^>]*)>(.*?)</PP_BETA\.\1>", text, re.S):
        attributes = _attributes(match.group(2))
        values = _floats(match.group(3))
        size = int(attributes.get("size", values.size))
        projectors.append(
            UpfProjector(
                index=int(match.group(1)),
                angular_momentum=int(attributes["angular_momentum"]),
                cutoff_radius=float(attributes.get("cutoff_radius", 0.0)),
                values=values[:size] if size <= values.size else values,
            )
        )
    if not projectors:
        raise ValueError(f"{source} has no PP_BETA projectors")
    projectors.sort(key=lambda projector: projector.index)

    dij_section = _section(text, "PP_DIJ")
    count = len(projectors)
    dij = np.zeros((count, count))
    if dij_section is not None:
        flat = _floats(dij_section[1])
        if flat.size != count * count:
            raise ValueError(f"{source}: PP_DIJ has {flat.size} entries for {count} projectors")
        dij = flat.reshape(count, count)

    wavefunctions = []
    for match in re.finditer(r"<PP_CHI\.(\d+)\b([^>]*)>(.*?)</PP_CHI\.\1>", text, re.S):
        attributes = _attributes(match.group(2))
        wavefunctions.append(
            UpfWavefunction(
                label=attributes.get("label", "").strip(),
                angular_momentum=int(attributes["l"]),
                occupation=float(attributes.get("occupation", 0.0)),
                values=_floats(match.group(3)),
            )
        )

    rho_section = _section(text, "PP_RHOATOM")
    if rho_section is None:
        raise ValueError(f"{source} lacks PP_RHOATOM")
    core_section = _section(text, "PP_NLCC")
    info = _section(text, "PP_INFO")

    for name, array in (("PP_LOCAL", local_potential), ("PP_RHOATOM", _floats(rho_section[1]))):
        if array.size != mesh.size:
            raise ValueError(f"{source}: {name} has {array.size} points, the mesh {mesh.size}")
    return UpfPseudopotential(
        symbol=header.get("element", source.stem).strip(),
        z_valence=float(header["z_valence"]),
        functional=header.get("functional", "").strip(),
        relativistic=header.get("relativistic", "scalar").strip(),
        mesh=mesh,
        local_potential=local_potential,
        projectors=tuple(projectors),
        dij=dij,
        wavefunctions=tuple(wavefunctions),
        valence_charge=_floats(rho_section[1]),
        core_density=None if core_section is None else _floats(core_section[1]),
        source=source,
        info="" if info is None else info[1].strip(),
    )


def kleinman_bylander_channels(upf: UpfPseudopotential, *, tolerance: float = 1.0e-12) -> tuple[KleinmanBylanderChannel, ...]:
    """The separable projectors of a UPF potential, ``D`` diagonalised within each angular momentum.

    ``D`` couples the projectors of one ``l`` only.  Its block for ``l`` is diagonalised, ``D = U diag(d) U^T``, which
    turns ``sum_ij D_ij |beta_i><beta_j|`` into ``sum_k d_k |beta'_k><beta'_k|`` with ``beta'_k = sum_i U_ik beta_i``
    (the ONCVPSP files already come with a diagonal ``D``, then ``U`` is the identity up to the order of equal
    ``d``).  A coupling with ``|d| <= tolerance`` is dropped.
    """

    mesh = upf.mesh
    projectors = upf.projectors
    cross = 0.0
    for first, left in enumerate(projectors):
        for second, right in enumerate(projectors):
            if left.angular_momentum != right.angular_momentum:
                cross = max(cross, abs(upf.dij[first, second]))
    if cross > 1.0e-10:
        raise ValueError(f"{upf.source}: PP_DIJ couples projectors of different angular momenta ({cross:.3e})")
    # r*beta/r: the first mesh point of a UPF file is r = 0, where beta(r)/r has the limit of its slope
    safe = np.where(mesh > 0.0, mesh, 1.0)
    channels: list[KleinmanBylanderChannel] = []
    for angular_momentum in sorted({projector.angular_momentum for projector in projectors}):
        members = [i for i, projector in enumerate(projectors) if projector.angular_momentum == angular_momentum]
        block = upf.dij[np.ix_(members, members)]
        couplings, rotation = np.linalg.eigh(0.5 * (block + block.T))
        order = np.argsort(-np.abs(couplings))
        index = 0
        for position in order:
            coupling = float(couplings[position])
            if abs(coupling) <= tolerance:
                continue
            combined = sum(rotation[k, position] * projectors[member].values for k, member in enumerate(members))
            radial = np.asarray(combined, dtype=float) / safe
            if mesh[0] == 0.0 and radial.size > 2:
                radial[0] = 2.0 * radial[1] - radial[2]
            channels.append(
                KleinmanBylanderChannel(
                    angular_momentum=angular_momentum,
                    index=index,
                    sign=1.0 if coupling > 0.0 else -1.0,
                    coupling=coupling,
                    cutoff_radius=max(projectors[member].cutoff_radius for member in members),
                    values=np.sqrt(abs(coupling)) * radial,
                )
            )
            index += 1
    return tuple(channels)


def parsec_logarithmic_grid(
    maximum_radius: float, *, offset: float = 1.0e-3, step: float = 6.0e-3
) -> tuple[np.ndarray, float, float]:
    """``r(i) = a*exp(b*(i-1)) - a`` for ``i = 2 ... n`` with the last radius at or beyond ``maximum_radius``.

    ``a = offset``, ``b = step``.  The spacing at ``r`` is ``b*(r + a)``: 0.012 bohr at 2 bohr, as fine as the
    0.01 bohr mesh of a PseudoDojo file where the projectors live.  Returns ``(radii, a, b)``.
    """

    count = int(np.ceil(np.log1p(maximum_radius / offset) / step)) + 1
    radii = offset * np.exp(step * np.arange(1, count)) - offset
    return radii, float(offset), float(step)


def _resample(mesh: np.ndarray, values: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Cubic spline of a table of the file on ``radii`` (zero beyond the mesh)."""

    spline = CubicSpline(mesh, values)
    out = np.zeros_like(radii)
    inside = radii <= mesh[-1]
    out[inside] = spline(radii[inside])
    return out


@dataclass(frozen=True)
class OncvPseudopotential(ParsecPseudopotential):
    """A UPF potential in the form of the PARSEC tables, with all its projectors.

    ``channel_potentials`` holds the local potential only, under the key :data:`LOCAL_CHANNEL`; a UPF file has no
    semilocal channel potentials, so the Kleinman-Bylander projectors are stored directly in ``kb_channels``
    (radial functions on ``radii``, signs of the couplings), several for one ``l`` where the file has them.
    ``radial_wavefunctions`` holds the valence pseudo-wavefunction ``u_l(r)`` of each ``l`` that has one (the last one
    of the file where several are given), which fixes the set of angular momenta and serves the DFT+U projector.
    """

    kb_channels: tuple[KleinmanBylanderChannel, ...] = ()

    #: the projectors are tables of the file, not derived from semilocal potentials (``radial_derivatives``)
    projectors_are_tabulated = True

    def nonlocal_channels(self, local_l: int) -> list[tuple[int, int]]:
        """``(l, index)`` of every separable projector, in the order of ``kb_channels``."""

        return [(channel.angular_momentum, channel.index) for channel in self.kb_channels]

    def radial_projector(self, angular_momentum: int, local_l: int, index: int = 0) -> tuple[np.ndarray, float]:
        """Radial function and sign of the ``index``-th projector of ``angular_momentum`` (as the Martins potentials)."""

        self.validate_local_channel(local_l)
        for channel in self.kb_channels:
            if channel.angular_momentum == angular_momentum and channel.index == index:
                return np.array(channel.values, dtype=float), channel.sign
        raise ValueError(f"no projector index {index} for l={angular_momentum} in {self.source}")

    def projector_count(self, angular_momentum: int) -> int:
        return sum(1 for channel in self.kb_channels if channel.angular_momentum == angular_momentum)


def to_parsec_pseudopotential(
    upf: UpfPseudopotential, *, offset: float = 1.0e-3, step: float = 6.0e-3
) -> OncvPseudopotential:
    """Resample a UPF potential onto a PARSEC logarithmic grid (cubic splines of the tables of the file)."""

    radii, a, b = parsec_logarithmic_grid(float(upf.mesh[-1]), offset=offset, step=step)
    local = _resample(upf.mesh, upf.local_potential, radii)
    # beyond the last mesh point the file ends in the Coulomb tail
    outside = radii > upf.mesh[-1]
    local[outside] = -2.0 * upf.z_valence / radii[outside]

    channels = tuple(
        KleinmanBylanderChannel(
            angular_momentum=channel.angular_momentum,
            index=channel.index,
            sign=channel.sign,
            coupling=channel.coupling,
            cutoff_radius=channel.cutoff_radius,
            values=_resample(upf.mesh, channel.values, radii),
        )
        for channel in kleinman_bylander_channels(upf)
    )

    safe = np.where(upf.mesh > 0.0, upf.mesh, 1.0)
    density = upf.valence_charge / (4.0 * np.pi * safe**2)
    if upf.mesh[0] == 0.0 and density.size > 2:
        density[0] = 2.0 * density[1] - density[2]
    valence = _resample(upf.mesh, density, radii)
    core = np.zeros_like(radii) if upf.core_density is None else _resample(upf.mesh, upf.core_density, radii)

    waves: dict[int, np.ndarray] = {}
    occupations: dict[int, float] = {}
    for wavefunction in upf.wavefunctions:
        waves[wavefunction.angular_momentum] = _resample(upf.mesh, wavefunction.values, radii)
        occupations[wavefunction.angular_momentum] = wavefunction.occupation
    cutoffs = {
        channel.angular_momentum: channel.cutoff_radius for channel in channels if channel.cutoff_radius > 0.0
    }
    for angular_momentum in sorted({channel.angular_momentum for channel in channels} - set(waves)):
        # an l with a projector but no stored state: a zero table keeps the set of angular momenta complete
        waves[angular_momentum] = np.zeros_like(radii)
        occupations[angular_momentum] = 0.0

    return OncvPseudopotential(
        symbol=upf.symbol,
        correlation=upf.functional.lower() or "pbe",
        relativity="sr",
        core_marker="nc" if upf.core_density is None else "nlcc",
        comment=f"converted from {upf.source.name} (UPF)",
        configuration="",
        number_of_channels=len({channel.angular_momentum for channel in channels}),
        number_of_spin_orbit_channels=0,
        radial_parameter_a=a,
        radial_parameter_b=b,
        ionic_charge=float(upf.z_valence),
        radii=radii,
        channel_potentials={LOCAL_CHANNEL: local},
        valence_density=valence,
        core_density=core,
        radial_wavefunctions=waves,
        channel_occupations=occupations,
        channel_cutoffs=cutoffs,
        source=upf.source,
        file_format="upf",
        local_channel=LOCAL_CHANNEL,
        kb_channels=channels,
    )


def read_upf_pseudopotential(path: str | Path, *, offset: float = 1.0e-3, step: float = 6.0e-3) -> OncvPseudopotential:
    """Read a UPF v2 file as a PARSEC pseudopotential (:func:`read_upf` then :func:`to_parsec_pseudopotential`)."""

    return to_parsec_pseudopotential(read_upf(path), offset=offset, step=step)


__all__ = [
    "KleinmanBylanderChannel",
    "LOCAL_CHANNEL",
    "OncvPseudopotential",
    "UpfPseudopotential",
    "UpfProjector",
    "UpfWavefunction",
    "kleinman_bylander_channels",
    "parsec_logarithmic_grid",
    "read_upf",
    "read_upf_pseudopotential",
    "to_parsec_pseudopotential",
]
