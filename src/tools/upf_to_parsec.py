#!/usr/bin/env python3
"""Convert a semilocal norm-conserving UPF v2 file to PARSEC POTRE.DAT.

This converter targets PARSEC's ``martins_new`` pseudopotential reader.  It
currently supports the conservative subset used by the FHI H and Si UPFs in
the 0d_Si28H36 benchmark:

* UPF v2, norm-conserving semilocal (``pseudo_type="SL"``);
* one scalar potential and one PP_CHI reference function for every
  consecutive angular channel (the UPF local channel may lie above the
  header's nonlocal ``l_max``);
* no spin-orbit terms, PAW/ultrasoft data, or nonlinear core correction;
* a pure-exponential UPF radial mesh.

The conversion is not a text-only format change.  UPF potentials are sampled
on a pure-exponential mesh, while PARSEC's Martins-new reader requires

    r_k = a * (exp(b*k) - 1),  k = 1, ..., nr.

All radial fields are therefore resampled onto that exact shifted-log mesh.
Potential values remain in Rydbergs and are written as r*V, PP_CHI is already
the reduced radial function u=rR, and PP_RHOATOM is already 4*pi*r^2*rho.

The UPF local channel is retained.  The generated file must be paired with
``Local_Component`` set to that channel in parsec.in.
"""

from __future__ import annotations

import argparse
import bisect
import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TextIO


ANGULAR_LABELS = ("s", "p", "d", "f")
PARSEC_XC_CODES = ("ca", "pl", "pw", "pb", "xa", "wi", "hl")


class ConversionError(ValueError):
    """Raised when an input cannot be converted without guessing."""


@dataclass(frozen=True)
class Channel:
    """One UPF semilocal angular channel."""

    angular_momentum: int
    potential: tuple[float, ...]
    wavefunction: tuple[float, ...]
    occupation: float
    principal_n: int | None


@dataclass(frozen=True)
class UpfData:
    """Validated UPF data used by the conversion."""

    version: str
    element: str
    functional: str
    z_valence: float
    reference_electrons: float
    local_l: int
    radius: tuple[float, ...]
    mesh_dx: float
    channels: tuple[Channel, ...]
    radial_valence_charge: tuple[float, ...]
    projector_cutoff: float


@dataclass(frozen=True)
class ParsecData:
    """Arrays in PARSEC Martins-new storage conventions."""

    element: str
    xc_code: str
    z_valence: float
    reference_electrons: float
    local_l: int
    grid_a: float
    grid_b: float
    radius: tuple[float, ...]
    channels: tuple[Channel, ...]
    radial_core_charge: tuple[float, ...]
    radial_valence_charge: tuple[float, ...]
    projector_cutoff: float
    source_name: str
    source_version: str
    source_functional: str


def parse_float(token: str) -> float:
    """Parse Fortran-style E or D exponent notation."""

    return float(token.replace("D", "E").replace("d", "e"))


def parse_bool(value: str | None) -> bool:
    """Parse a UPF boolean attribute."""

    if value is None:
        return False
    normalized = value.strip().upper()
    if normalized in {"T", "TRUE", ".TRUE."}:
        return True
    if normalized in {"F", "FALSE", ".FALSE."}:
        return False
    raise ConversionError(f"unrecognized UPF boolean value {value!r}")


def parse_values(element: ET.Element | None, expected: int, label: str) -> tuple[float, ...]:
    """Read and validate one numerical UPF field."""

    if element is None or element.text is None:
        raise ConversionError(f"missing {label}")
    try:
        values = tuple(parse_float(token) for token in element.text.split())
    except ValueError as exc:
        raise ConversionError(f"{label} contains a non-numeric value") from exc
    if len(values) != expected:
        raise ConversionError(
            f"{label} contains {len(values)} values; expected {expected}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ConversionError(f"{label} contains a non-finite value")
    return values


def require_attribute(element: ET.Element, name: str, label: str) -> str:
    """Return a required XML attribute."""

    value = element.get(name)
    if value is None:
        raise ConversionError(f"{label} is missing its {name!r} attribute")
    return value


def validate_increasing_positive(values: Sequence[float], label: str) -> None:
    """Require a finite, strictly increasing, positive mesh."""

    if not values:
        raise ConversionError(f"{label} is empty")
    if values[0] <= 0.0:
        raise ConversionError(f"{label} must omit r=0 and start at positive radius")
    for left, right in zip(values, values[1:]):
        if not (math.isfinite(left) and right > left):
            raise ConversionError(f"{label} is not strictly increasing and finite")


def validate_pure_exponential_grid(radius: Sequence[float], dx: float) -> None:
    """Check the UPF mesh form supported by this converter."""

    if not math.isfinite(dx) or dx <= 0.0:
        raise ConversionError(f"invalid UPF mesh dx={dx!r}")
    r0 = radius[0]
    max_relative_error = max(
        abs(actual - r0 * math.exp(dx * index))
        / max(abs(actual), abs(r0 * math.exp(dx * index)), 1.0)
        for index, actual in enumerate(radius)
    )
    if max_relative_error > 5.0e-11:
        raise ConversionError(
            "only pure-exponential UPF meshes are currently supported; "
            f"mesh mismatch is {max_relative_error:.3e}"
        )


def parse_upf(path: Path, allow_ionized_reference: bool = False) -> UpfData:
    """Parse and validate the supported UPF subset."""

    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ConversionError(f"cannot read UPF file {path}: {exc}") from exc

    if root.tag != "UPF":
        raise ConversionError(f"{path} does not have a UPF root element")
    version = root.get("version", "")
    if not version.startswith("2."):
        raise ConversionError(f"only UPF v2 is supported, found version {version!r}")

    header = root.find("PP_HEADER")
    mesh = root.find("PP_MESH")
    if header is None or mesh is None:
        raise ConversionError("UPF is missing PP_HEADER or PP_MESH")

    pseudo_type = require_attribute(header, "pseudo_type", "PP_HEADER").strip().upper()
    if pseudo_type != "SL":
        raise ConversionError(
            "conversion requires a norm-conserving semilocal UPF "
            f"(pseudo_type='SL'), found {pseudo_type!r}"
        )
    unsupported_flags = {
        "is_ultrasoft": parse_bool(header.get("is_ultrasoft")),
        "is_paw": parse_bool(header.get("is_paw")),
        "is_coulomb": parse_bool(header.get("is_coulomb")),
        "has_so": parse_bool(header.get("has_so")),
        "core_correction": parse_bool(header.get("core_correction")),
    }
    relativistic = header.get("relativistic", "").strip().lower()
    if relativistic in {"full", "fully", "relativistic"}:
        unsupported_flags["fully_relativistic"] = True
    if root.find("PP_NLCC") is not None:
        unsupported_flags["PP_NLCC"] = True
    enabled = [name for name, value in unsupported_flags.items() if value]
    if enabled:
        raise ConversionError(
            "unsupported UPF feature(s): "
            + ", ".join(enabled)
            + "; this converter intentionally refuses lossy conversion"
        )

    element = require_attribute(header, "element", "PP_HEADER").strip()
    if not element or len(element) > 2:
        raise ConversionError(f"invalid element symbol {element!r}")
    functional = require_attribute(header, "functional", "PP_HEADER").strip()
    z_valence = parse_float(require_attribute(header, "z_valence", "PP_HEADER"))
    local_l = int(require_attribute(header, "l_local", "PP_HEADER"))
    l_max = int(require_attribute(header, "l_max", "PP_HEADER"))
    mesh_size = int(require_attribute(header, "mesh_size", "PP_HEADER"))
    if l_max not in range(-1, 4):
        raise ConversionError(
            "PARSEC Martins-new supports nonlocal channels only through f; "
            f"l_max={l_max}"
        )
    if local_l not in range(4):
        raise ConversionError(
            f"PARSEC Martins-new supports a local channel only through f; l={local_l}"
        )
    if z_valence <= 0.0 or not math.isfinite(z_valence):
        raise ConversionError(f"invalid z_valence={z_valence!r}")
    if mesh_size < 20:
        raise ConversionError(
            f"radial mesh has only {mesh_size} points; PARSEC needs tail guard points"
        )

    radius = parse_values(mesh.find("PP_R"), mesh_size, "PP_R")
    validate_increasing_positive(radius, "PP_R")
    mesh_dx = parse_float(require_attribute(mesh, "dx", "PP_MESH"))
    validate_pure_exponential_grid(radius, mesh_dx)

    local_potential = parse_values(root.find("PP_LOCAL"), mesh_size, "PP_LOCAL")
    semilocal = root.find("PP_SEMILOCAL")
    if semilocal is None:
        raise ConversionError("semilocal UPF is missing PP_SEMILOCAL")
    potentials: dict[int, tuple[float, ...]] = {local_l: local_potential}
    for child in semilocal:
        l_text = child.get("L") or child.get("l")
        if l_text is None:
            raise ConversionError(f"{child.tag} is missing angular momentum L")
        angular_momentum = int(l_text)
        if angular_momentum in potentials:
            raise ConversionError(f"duplicate semilocal potential for l={angular_momentum}")
        potentials[angular_momentum] = parse_values(
            child, mesh_size, f"{child.tag} (l={angular_momentum})"
        )

    # In UPF, l_max is the largest nonlocal/projector angular momentum.
    # A common FHI conversion chooses the highest generated channel as local,
    # so l_local can be one larger than l_max (for example, d-local with
    # l_max=1).  Martins-new needs the complete semilocal set.
    channel_max = max(l_max, local_l)
    expected_l = set(range(channel_max + 1))
    if set(potentials) != expected_l:
        raise ConversionError(
            "semilocal potentials must cover consecutive angular channels; "
            f"found {sorted(potentials)}, expected {sorted(expected_l)}"
        )

    pswfc = root.find("PP_PSWFC")
    if pswfc is None:
        raise ConversionError("UPF is missing PP_PSWFC reference functions")
    wavefunctions: dict[int, tuple[tuple[float, ...], float, int | None]] = {}
    for child in pswfc:
        l_text = child.get("l") or child.get("L")
        if l_text is None:
            raise ConversionError(f"{child.tag} is missing angular momentum l")
        angular_momentum = int(l_text)
        if angular_momentum in wavefunctions:
            raise ConversionError(
                f"multiple PP_CHI reference functions for l={angular_momentum}; "
                "PARSEC Martins-new accepts only one per channel"
            )
        occupation = parse_float(child.get("occupation", "0"))
        if occupation < 0.0 or not math.isfinite(occupation):
            raise ConversionError(
                f"{child.tag} has invalid occupation {occupation!r}"
            )
        n_text = child.get("n")
        principal_n = int(n_text) if n_text is not None else None
        wavefunctions[angular_momentum] = (
            parse_values(child, mesh_size, f"{child.tag} (l={angular_momentum})"),
            occupation,
            principal_n,
        )
    if set(wavefunctions) != expected_l:
        raise ConversionError(
            "PP_CHI fields must cover all consecutive semilocal channels; "
            f"found {sorted(wavefunctions)}, expected {sorted(expected_l)}"
        )

    channels = tuple(
        Channel(
            angular_momentum=angular_momentum,
            potential=potentials[angular_momentum],
            wavefunction=wavefunctions[angular_momentum][0],
            occupation=wavefunctions[angular_momentum][1],
            principal_n=wavefunctions[angular_momentum][2],
        )
        for angular_momentum in range(channel_max + 1)
    )

    radial_valence_charge = parse_values(
        root.find("PP_RHOATOM"), mesh_size, "PP_RHOATOM"
    )

    nonlocal_field = root.find("PP_NONLOCAL")
    if nonlocal_field is None:
        raise ConversionError("UPF is missing PP_NONLOCAL projectors")
    projector_count = int(
        require_attribute(header, "number_of_proj", "PP_HEADER")
    )
    expected_projector_l = expected_l - {local_l}
    if projector_count != len(expected_projector_l):
        raise ConversionError(
            "conversion requires exactly one projector for each nonlocal "
            f"channel; header reports {projector_count}, expected "
            f"{len(expected_projector_l)}"
        )

    cutoffs: list[float] = []
    projectors: dict[int, tuple[float, ...]] = {}
    for child in nonlocal_field:
        if not child.tag.startswith("PP_BETA"):
            continue
        angular_momentum = int(
            require_attribute(child, "angular_momentum", child.tag)
        )
        if angular_momentum in projectors:
            raise ConversionError(
                f"multiple PP_BETA projectors for l={angular_momentum} "
                "cannot be represented by Martins-new"
            )
        projectors[angular_momentum] = parse_values(
            child, mesh_size, f"{child.tag} projector"
        )
        cutoff_text = child.get("cutoff_radius")
        if cutoff_text is not None:
            cutoff = parse_float(cutoff_text)
            if cutoff > 0.0 and math.isfinite(cutoff):
                cutoffs.append(cutoff)
    if set(projectors) != expected_projector_l:
        raise ConversionError(
            "PP_BETA angular channels do not match the semilocal channels; "
            f"found {sorted(projectors)}, expected {sorted(expected_projector_l)}"
        )

    # A single-projector semilocal UPF must have beta_l proportional to
    # (V_l-V_local)*chi_l.  Channel-wise scaling is allowed because PP_DIJ
    # may carry the inverse scaling.
    local_potential_values = potentials[local_l]
    for angular_momentum, projector in projectors.items():
        potential_differences = tuple(
            potential - local_value
            for potential, local_value in zip(
                potentials[angular_momentum], local_potential_values
            )
        )
        raw_projector = tuple(
            difference * wavefunction
            for difference, wavefunction in zip(
                potential_differences,
                wavefunctions[angular_momentum][0],
            )
        )
        # QE's FHI reader deliberately truncates beta after the semilocal
        # potential difference falls below 1e-6 Ry.  Exclude only that
        # documented zero tail from the proportionality check.
        support = tuple(
            index
            for index, (beta, difference) in enumerate(
                zip(projector, potential_differences)
            )
            if beta != 0.0 or abs(difference) > 1.05e-6
        )
        raw_norm = sum(raw_projector[index] ** 2 for index in support)
        if raw_norm == 0.0:
            raise ConversionError(
                f"semilocal l={angular_momentum} projector is identically zero"
            )
        scale = sum(
            projector[index] * raw_projector[index]
            for index in support
        ) / raw_norm
        scale_norm = max(
            max(abs(projector[index]) for index in support),
            abs(scale) * max(abs(raw_projector[index]) for index in support),
            1.0e-30,
        )
        relative_residual = max(
            abs(projector[index] - scale * raw_projector[index])
            for index in support
        ) / scale_norm
        if relative_residual > 2.0e-8:
            raise ConversionError(
                f"PP_BETA for l={angular_momentum} is not proportional to "
                "(V_l-V_local)*PP_CHI; its nonlocal operator cannot be "
                "reproduced by this converter"
            )
    if not cutoffs:
        raise ConversionError(
            "UPF has no positive PP_BETA cutoff_radius; PARSEC needs a nonzero "
            "projector cutoff in each wavefunction record"
        )
    projector_cutoff = max(cutoffs)
    if projector_cutoff >= radius[-1]:
        raise ConversionError(
            f"projector cutoff {projector_cutoff} is outside the radial mesh"
        )

    occupation_sum = sum(channel.occupation for channel in channels)
    if abs(occupation_sum - z_valence) > 5.0e-6 * max(1.0, z_valence):
        if not allow_ionized_reference:
            raise ConversionError(
                f"PP_CHI occupations sum to {occupation_sum:.12g}, "
                f"but z_valence is {z_valence:.12g}; pass "
                "--allow-ionized-reference only when this charge state is intentional"
            )
        if occupation_sum <= 0.0:
            raise ConversionError("ionized reference has no occupied pseudo-wavefunctions")

    return UpfData(
        version=version,
        element=element,
        functional=functional,
        z_valence=z_valence,
        reference_electrons=occupation_sum,
        local_l=local_l,
        radius=radius,
        mesh_dx=mesh_dx,
        channels=channels,
        radial_valence_charge=radial_valence_charge,
        projector_cutoff=projector_cutoff,
    )


def infer_parsec_xc(functional: str) -> str:
    """Map unambiguous UPF functional labels to PARSEC's two-letter codes."""

    normalized = functional.upper().replace("_", "-").replace(" ", "")
    if normalized == "PBE":
        return "pb"
    if normalized in {"SLA-PW", "LDA-PW", "PW92", "PWLDA"}:
        return "pl"
    if normalized in {"SLA-PZ", "LDA-PZ", "PZ", "CA"}:
        return "ca"
    raise ConversionError(
        f"cannot map UPF functional {functional!r} unambiguously to PARSEC; "
        "pass --xc-code explicitly"
    )


def even_origin_extrapolation(
    radius: Sequence[float], values: Sequence[float], target: float
) -> float:
    """Extrapolate a finite central potential as V(0)+c*r^2."""

    fit_points = min(8, len(radius))
    x_values = [value * value for value in radius[:fit_points]]
    y_values = values[:fit_points]
    x_mean = sum(x_values) / fit_points
    y_mean = sum(y_values) / fit_points
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0.0:
        return values[0]
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator
    # Anchor at the first tabulated value to avoid an interpolation jump.
    return values[0] + slope * (target * target - radius[0] * radius[0])


def power_origin_extrapolation(
    radius: Sequence[float],
    values: Sequence[float],
    target: float,
    fallback_power: float,
) -> float:
    """Extrapolate a regular radial quantity with a small-r power law."""

    first = values[0]
    if first == 0.0:
        return 0.0
    power = fallback_power
    for index in range(1, min(8, len(values))):
        candidate = values[index]
        if candidate != 0.0 and candidate * first > 0.0:
            estimated = math.log(abs(candidate / first)) / math.log(
                radius[index] / radius[0]
            )
            if math.isfinite(estimated):
                power = min(12.0, max(0.0, estimated))
            break
    return first * (target / radius[0]) ** power


def log_linear_interpolate(
    radius: Sequence[float],
    log_radius: Sequence[float],
    values: Sequence[float],
    target: float,
) -> float:
    """Shape-preserving linear interpolation in log(r)."""

    index = bisect.bisect_right(radius, target)
    if index == 0:
        raise ConversionError("internal interpolation error below source mesh")
    if index >= len(radius):
        return values[-1]
    lower = index - 1
    fraction = (
        math.log(target) - log_radius[lower]
    ) / (log_radius[index] - log_radius[lower])
    return values[lower] + fraction * (values[index] - values[lower])


def find_coulomb_tail_start(
    radius: Sequence[float],
    potentials: Sequence[Sequence[float]],
    z_valence: float,
) -> float:
    """Find the earliest suffix in which every channel is -2Z/r."""

    tolerance = 2.0e-9 * max(1.0, abs(z_valence))
    first = len(radius)
    for index in range(len(radius) - 1, -1, -1):
        if all(
            abs(radius[index] * potential[index] + 2.0 * z_valence) <= tolerance
            for potential in potentials
        ):
            first = index
        else:
            break
    if first == len(radius) or len(radius) - first < 10:
        raise ConversionError(
            "UPF potentials do not provide a sufficiently long common "
            "Coulomb tail r*V=-2*z_valence"
        )
    return radius[first]


def convert_data(
    upf: UpfData,
    source_name: str,
    xc_code: str,
    grid_refinement: int = 2,
) -> ParsecData:
    """Resample supported UPF fields into PARSEC storage conventions."""

    if xc_code not in PARSEC_XC_CODES:
        raise ConversionError(
            f"invalid PARSEC XC code {xc_code!r}; choose from "
            + ", ".join(PARSEC_XC_CODES)
        )
    if grid_refinement not in {1, 2}:
        raise ConversionError("grid refinement must be 1 or 2")

    # The source mesh is r_i=a*exp((i-1)*b).  Reusing its a and b gives
    # PARSEC's required shifted mesh a*expm1(k*b).  One midpoint subdivision
    # substantially improves the trapezoidal KB integrals PARSEC constructs
    # for this UPF.  The last target point is one refined step beyond the
    # source mesh and is an analytic outer guard point.
    grid_a = upf.radius[0]
    grid_b = upf.mesh_dx / grid_refinement
    target_size = grid_refinement * (len(upf.radius) - 1) + 1
    target_radius = tuple(
        grid_a * math.expm1(grid_b * index)
        for index in range(1, target_size + 1)
    )
    validate_increasing_positive(target_radius, "generated PARSEC radial grid")

    log_source_radius = tuple(math.log(value) for value in upf.radius)
    tail_start = find_coulomb_tail_start(
        upf.radius,
        [channel.potential for channel in upf.channels],
        upf.z_valence,
    )

    converted_channels: list[Channel] = []
    for channel in upf.channels:
        potential: list[float] = []
        wavefunction: list[float] = []
        for target in target_radius:
            if target < upf.radius[0]:
                potential_value = even_origin_extrapolation(
                    upf.radius, channel.potential, target
                )
                wavefunction_value = power_origin_extrapolation(
                    upf.radius,
                    channel.wavefunction,
                    target,
                    channel.angular_momentum + 1.0,
                )
            else:
                if target >= tail_start:
                    potential_value = -2.0 * upf.z_valence / target
                else:
                    potential_value = log_linear_interpolate(
                        upf.radius,
                        log_source_radius,
                        channel.potential,
                        target,
                    )
                if target > upf.radius[-1]:
                    wavefunction_value = 0.0
                else:
                    wavefunction_value = log_linear_interpolate(
                        upf.radius,
                        log_source_radius,
                        channel.wavefunction,
                        target,
                    )
            potential.append(target * potential_value)
            wavefunction.append(wavefunction_value)

        converted_channels.append(
            Channel(
                angular_momentum=channel.angular_momentum,
                potential=tuple(potential),
                wavefunction=tuple(wavefunction),
                occupation=channel.occupation,
                principal_n=channel.principal_n,
            )
        )

    radial_valence_charge: list[float] = []
    for target in target_radius:
        if target < upf.radius[0]:
            value = power_origin_extrapolation(
                upf.radius,
                upf.radial_valence_charge,
                target,
                fallback_power=2.0,
            )
        elif target > upf.radius[-1]:
            value = 0.0
        else:
            value = log_linear_interpolate(
                upf.radius,
                log_source_radius,
                upf.radial_valence_charge,
                target,
            )
        # Linear interpolation preserves positivity; remove only negative
        # roundoff from a nominally zero tail.
        if value < 0.0 and abs(value) < 1.0e-14:
            value = 0.0
        if value < 0.0:
            raise ConversionError("interpolated PP_RHOATOM became negative")
        radial_valence_charge.append(value)

    return ParsecData(
        element=upf.element,
        xc_code=xc_code,
        z_valence=upf.z_valence,
        reference_electrons=upf.reference_electrons,
        local_l=upf.local_l,
        grid_a=grid_a,
        grid_b=grid_b,
        radius=target_radius,
        channels=tuple(converted_channels),
        radial_core_charge=(0.0,) * len(target_radius),
        radial_valence_charge=tuple(radial_valence_charge),
        projector_cutoff=upf.projector_cutoff,
        source_name=source_name,
        source_version=upf.version,
        source_functional=upf.functional,
    )


def write_values(handle: TextIO, values: Iterable[float]) -> None:
    """Write four values per line in Fortran-readable scientific notation."""

    row: list[float] = []
    for value in values:
        row.append(value)
        if len(row) == 4:
            handle.write("".join(f" {item: .12E}" for item in row) + "\n")
            row.clear()
    if row:
        handle.write("".join(f" {item: .12E}" for item in row) + "\n")


def configuration_description(channels: Sequence[Channel], cutoff: float) -> str:
    """Build the informational third Martins-new header record."""

    parts: list[str] = []
    for channel in channels:
        n_value = channel.principal_n
        prefix = str(n_value) if n_value is not None else ""
        parts.append(
            f"{prefix}{ANGULAR_LABELS[channel.angular_momentum]}"
            f"({channel.occupation:6.2f}) rc={cutoff:7.4f}"
        )
    return " ".join(parts)


def write_parsec(path: Path, data: ParsecData) -> None:
    """Write one PARSEC Martins-new POTRE.DAT file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f" {data.element:<2} {data.xc_code:2} nrl nc  \n")
        handle.write(
            f" converted from {data.source_name} "
            f"(UPF {data.source_version}, semilocal norm-conserving)\n"
        )
        handle.write(
            " "
            + configuration_description(data.channels, data.projector_cutoff)
            + "\n"
        )
        handle.write(
            f" {len(data.channels):3d}  0 {len(data.radius):4d}"
            f"  {data.grid_a:.12E}  {data.grid_b:.12E}"
            f"  {data.z_valence:.12E}"
            "   nl nls nr a b zion\n"
        )

        handle.write(" Radial grid follows\n")
        write_values(handle, data.radius)

        for channel in data.channels:
            handle.write(" Pseudopotential follows (l on next line)\n")
            handle.write(f" {channel.angular_momentum:d}\n")
            write_values(handle, channel.potential)

        handle.write(" Core charge follows\n")
        write_values(handle, data.radial_core_charge)
        handle.write(" Valence charge follows\n")
        write_values(handle, data.radial_valence_charge)

        for channel in data.channels:
            handle.write(" Pseudo-wave-function follows (l, zelect, rc)\n")
            handle.write(
                f" {channel.angular_momentum:d}"
                f"  {channel.occupation:.12E}"
                f"  {data.projector_cutoff:.12E}\n"
            )
            write_values(handle, channel.wavefunction)


def trapezoid(values: Sequence[float], radius: Sequence[float]) -> float:
    """Integrate values over the explicit radial mesh."""

    return sum(
        0.5 * (values[index] + values[index + 1])
        * (radius[index + 1] - radius[index])
        for index in range(len(radius) - 1)
    )


def validate_converted_data(data: ParsecData) -> None:
    """Apply converter-side invariants stricter than PARSEC's reader."""

    nr = len(data.radius)
    if nr < 20 or len(data.channels) not in range(1, 5):
        raise ConversionError("invalid generated mesh or channel count")
    expected_l = list(range(len(data.channels)))
    actual_l = [channel.angular_momentum for channel in data.channels]
    if actual_l != expected_l:
        raise ConversionError(
            f"generated channels are {actual_l}, expected consecutive {expected_l}"
        )
    if data.local_l not in expected_l:
        raise ConversionError(f"generated local l={data.local_l} is absent")

    for index, radius in enumerate(data.radius, start=1):
        expected = data.grid_a * math.expm1(data.grid_b * index)
        if abs(radius - expected) > 2.0e-12 * max(1.0, radius):
            raise ConversionError("generated radial mesh does not match header a,b")

    arrays: list[Sequence[float]] = [
        data.radial_core_charge,
        data.radial_valence_charge,
    ]
    arrays.extend(channel.potential for channel in data.channels)
    arrays.extend(channel.wavefunction for channel in data.channels)
    if any(len(values) != nr for values in arrays):
        raise ConversionError("generated radial fields do not all have nr values")
    if any(not math.isfinite(value) for values in arrays for value in values):
        raise ConversionError("generated data contains a non-finite value")

    charge = trapezoid(data.radial_valence_charge, data.radius)
    if abs(charge - data.reference_electrons) > 2.0e-3 * max(
        1.0, data.reference_electrons
    ):
        raise ConversionError(
            f"generated radial valence charge integrates to {charge:.12g}, "
            f"expected reference occupation {data.reference_electrons:.12g}"
        )

    local = data.channels[data.local_l]
    for channel in data.channels:
        if abs(channel.potential[-1] + 2.0 * data.z_valence) > 1.0e-10:
            raise ConversionError(
                f"l={channel.angular_momentum} does not have r*V=-2Z at the tail"
            )
        norm = trapezoid(
            [value * value for value in channel.wavefunction], data.radius
        )
        if channel.occupation > 1.0e-12 and abs(norm - 1.0) > 2.0e-3:
            raise ConversionError(
                f"occupied l={channel.angular_momentum} wavefunction norm is {norm:.12g}"
            )
        if channel.angular_momentum != data.local_l:
            denominator = trapezoid(
                [
                    wave * wave * (r_v - local_r_v) / radius
                    for wave, r_v, local_r_v, radius in zip(
                        channel.wavefunction,
                        channel.potential,
                        local.potential,
                        data.radius,
                    )
                ],
                data.radius,
            )
            if abs(denominator) < 1.0e-12:
                raise ConversionError(
                    f"l={channel.angular_momentum} Kleinman-Bylander "
                    "normalization integral is too small"
                )


def read_numeric_block(
    lines: Sequence[str], position: int, count: int, label: str
) -> tuple[tuple[float, ...], int]:
    """Read a known-size value block from the generated text file."""

    values: list[float] = []
    while len(values) < count:
        if position >= len(lines):
            raise ConversionError(f"generated file ends inside {label}")
        tokens = lines[position].split()
        try:
            row = [parse_float(token) for token in tokens]
        except ValueError as exc:
            raise ConversionError(f"non-numeric line inside generated {label}") from exc
        if not row or len(values) + len(row) > count:
            raise ConversionError(f"wrong value count in generated {label}")
        values.extend(row)
        position += 1
    return tuple(values), position


def validate_written_values(
    actual: Sequence[float], expected: Sequence[float], label: str
) -> None:
    """Check that text formatting preserved every generated value."""

    if len(actual) != len(expected):
        raise ConversionError(f"generated {label} has the wrong value count")
    for index, (actual_value, expected_value) in enumerate(
        zip(actual, expected), start=1
    ):
        if not math.isclose(
            actual_value,
            expected_value,
            rel_tol=6.0e-13,
            abs_tol=1.0e-300,
        ):
            raise ConversionError(
                f"generated {label} value {index} changed during text formatting"
            )


def validate_written_file(path: Path, expected: ParsecData) -> None:
    """Read the emitted section structure back and verify exact counts."""

    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise ConversionError(f"cannot reread generated file {path}: {exc}") from exc
    if len(lines) < 10:
        raise ConversionError("generated POTRE.DAT is unexpectedly short")
    if lines[0][1:3].strip() != expected.element:
        raise ConversionError("generated first-line element field is malformed")
    if lines[0][4:6].strip() != expected.xc_code:
        raise ConversionError("generated first-line XC field is malformed")

    header = lines[3].split()
    try:
        n_channels, n_minor, nr = map(int, header[:3])
        grid_a, grid_b, z_valence = map(parse_float, header[3:6])
    except (ValueError, IndexError) as exc:
        raise ConversionError("generated numeric header is malformed") from exc
    if (n_channels, n_minor, nr) != (
        len(expected.channels),
        0,
        len(expected.radius),
    ):
        raise ConversionError("generated numeric header counts are wrong")
    if not math.isclose(grid_a, expected.grid_a, rel_tol=6.0e-13):
        raise ConversionError("generated numeric header a is wrong")
    if not math.isclose(grid_b, expected.grid_b, rel_tol=6.0e-13):
        raise ConversionError("generated numeric header b is wrong")
    if not math.isclose(z_valence, expected.z_valence, rel_tol=6.0e-13):
        raise ConversionError("generated numeric header zion is wrong")

    position = 4
    if lines[position].strip() != "Radial grid follows":
        raise ConversionError("generated radial-grid label is missing")
    position += 1
    radius, position = read_numeric_block(lines, position, nr, "radial grid")
    validate_written_values(radius, expected.radius, "radial grid")

    for channel in expected.channels:
        if lines[position].strip() != "Pseudopotential follows (l on next line)":
            raise ConversionError("generated pseudopotential label is missing")
        position += 1
        if int(lines[position].strip()) != channel.angular_momentum:
            raise ConversionError("generated pseudopotential l value is wrong")
        position += 1
        potential, position = read_numeric_block(
            lines, position, nr, f"l={channel.angular_momentum} potential"
        )
        validate_written_values(
            potential,
            channel.potential,
            f"l={channel.angular_momentum} potential",
        )

    if lines[position].strip() != "Core charge follows":
        raise ConversionError("generated core-charge label is missing")
    position += 1
    core_charge, position = read_numeric_block(lines, position, nr, "core charge")
    validate_written_values(
        core_charge, expected.radial_core_charge, "core charge"
    )

    if lines[position].strip() != "Valence charge follows":
        raise ConversionError("generated valence-charge label is missing")
    position += 1
    valence_charge, position = read_numeric_block(
        lines, position, nr, "valence charge"
    )
    validate_written_values(
        valence_charge, expected.radial_valence_charge, "valence charge"
    )

    for channel in expected.channels:
        if (
            lines[position].strip()
            != "Pseudo-wave-function follows (l, zelect, rc)"
        ):
            raise ConversionError("generated wavefunction label is missing")
        position += 1
        metadata = lines[position].split()
        if len(metadata) != 3 or int(metadata[0]) != channel.angular_momentum:
            raise ConversionError("generated wavefunction metadata is malformed")
        occupation = parse_float(metadata[1])
        cutoff = parse_float(metadata[2])
        if not math.isclose(
            occupation, channel.occupation, rel_tol=6.0e-13, abs_tol=1.0e-300
        ):
            raise ConversionError("generated wavefunction occupation is wrong")
        if not math.isclose(
            cutoff,
            expected.projector_cutoff,
            rel_tol=6.0e-13,
            abs_tol=1.0e-300,
        ):
            raise ConversionError("generated wavefunction cutoff is wrong")
        position += 1
        wavefunction, position = read_numeric_block(
            lines, position, nr, f"l={channel.angular_momentum} wavefunction"
        )
        validate_written_values(
            wavefunction,
            channel.wavefunction,
            f"l={channel.angular_momentum} wavefunction",
        )

    if position != len(lines):
        raise ConversionError("generated file contains unexpected trailing records")


def write_validated_parsec(path: Path, data: ParsecData) -> None:
    """Validate a sibling temporary file before atomically installing it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        write_parsec(temporary_path, data)
        validate_written_file(temporary_path, data)
        temporary_path.chmod(mode)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def report(data: ParsecData, output: Path) -> None:
    """Print a compact numerical conversion report."""

    charge = trapezoid(data.radial_valence_charge, data.radius)
    print(f"wrote {output}")
    print(
        f"element={data.element}  UPF functional={data.source_functional}  "
        f"PARSEC XC code={data.xc_code}  zion={data.z_valence:g}"
    )
    if abs(data.reference_electrons - data.z_valence) > 5.0e-6:
        print(
            f"ionized atomic reference: Ne={data.reference_electrons:g}  "
            f"zion-Ne={data.z_valence-data.reference_electrons:+g}"
        )
    print(
        f"channels={','.join(ANGULAR_LABELS[c.angular_momentum] for c in data.channels)}  "
        f"local={ANGULAR_LABELS[data.local_l]}  "
        f"projector cutoff={data.projector_cutoff:.12g} bohr"
    )
    print(
        f"mesh={len(data.radius)}  a={data.grid_a:.12g} bohr  "
        f"b={data.grid_b:.12g}  "
        f"r=[{data.radius[0]:.12g}, {data.radius[-1]:.12g}] bohr"
    )
    print(f"radial valence charge integral={charge:.12g}")
    for channel in data.channels:
        norm = trapezoid(
            [value * value for value in channel.wavefunction], data.radius
        )
        print(
            f"  l={channel.angular_momentum}: occupation={channel.occupation:g}  "
            f"wavefunction norm={norm:.12g}  tail rV={channel.potential[-1]:.12g}"
        )
    print(
        "PARSEC input pairing: "
        f"Local_Component: {ANGULAR_LABELS[data.local_l]} ; "
        f"Correlation_Type: {data.xc_code}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert a semilocal norm-conserving UPF v2 pseudopotential "
            "to PARSEC Martins-new POTRE.DAT."
        )
    )
    parser.add_argument("input", type=Path, help="input UPF v2 file")
    parser.add_argument("output", type=Path, help="output *_POTRE.DAT file")
    parser.add_argument(
        "--xc-code",
        choices=PARSEC_XC_CODES,
        help=(
            "PARSEC two-letter XC provenance code; inferred for known UPF "
            "functional labels (PW92 LDA maps to 'pl', not 'pw')"
        ),
    )
    parser.add_argument(
        "--allow-ionized-reference",
        action="store_true",
        help=(
            "allow PP_CHI occupations and PP_RHOATOM charge to differ from "
            "z_valence (required for an intentional core-hole/ionic reference)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output file",
    )
    parser.add_argument(
        "--grid-refinement",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "subdivide the UPF logarithmic spacing before writing "
            "(default: 2; finer meshes improve PARSEC's KB quadrature)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    args = build_parser().parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        print("error: input and output paths must be different", file=sys.stderr)
        return 2
    if args.output.exists() and not args.force:
        print(
            f"error: output already exists: {args.output} (use --force to replace)",
            file=sys.stderr,
        )
        return 2

    try:
        upf = parse_upf(
            args.input,
            allow_ionized_reference=args.allow_ionized_reference,
        )
        xc_code = args.xc_code or infer_parsec_xc(upf.functional)
        converted = convert_data(
            upf,
            args.input.name,
            xc_code,
            grid_refinement=args.grid_refinement,
        )
        validate_converted_data(converted)
        write_validated_parsec(args.output, converted)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: cannot write {args.output}: {exc}", file=sys.stderr)
        return 1

    report(converted, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
