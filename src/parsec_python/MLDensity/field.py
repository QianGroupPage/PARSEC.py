"""Portable ML-density fields and exact mapping to a PARSEC cluster grid.

The DFT grid is always authoritative.  A one-dimensional field must carry the
same ordered physical coordinates as the active PARSEC grid.  A dense
three-dimensional field is either indexed directly when it has the exact
underlying Cartesian-grid shape, or sampled from an explicitly described
affine voxel lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.ndimage import map_coordinates

from ..Grid import RealSpaceGrid


BOHR_TO_ANGSTROM = 0.529177210903
_SCHEMA = "parsec_python.ml_density.v1"


@dataclass(frozen=True)
class DensityLoadResult:
    """Validated volume density on the active PARSEC grid."""

    density: np.ndarray
    source: Path
    units: str
    integral_before_clipping: float
    integral_after_clipping: float
    negative_values_clipped: int


def normalize_density_units(value: str) -> str:
    """Return one canonical unit label used by the interchange format."""

    normalized = str(value).strip().lower().replace(" ", "")
    normalized = normalized.replace("å", "angstrom").replace("^", "")
    normalized = normalized.replace("**", "")
    aliases = {
        "auto": "auto",
        "e/bohr3": "e_per_bohr3",
        "electron/bohr3": "e_per_bohr3",
        "electrons/bohr3": "e_per_bohr3",
        "eperbohr3": "e_per_bohr3",
        "e_per_bohr3": "e_per_bohr3",
        "1/bohr3": "e_per_bohr3",
        "e/angstrom3": "e_per_angstrom3",
        "electron/angstrom3": "e_per_angstrom3",
        "electrons/angstrom3": "e_per_angstrom3",
        "eperangstrom3": "e_per_angstrom3",
        "e_per_angstrom3": "e_per_angstrom3",
        "1/angstrom3": "e_per_angstrom3",
        "electronspervoxel": "electrons_per_voxel",
        "electron_per_voxel": "electrons_per_voxel",
        "electrons_per_voxel": "electrons_per_voxel",
        "e/voxel": "electrons_per_voxel",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported density units {value!r}") from error


def _npz_string(archive: Mapping[str, np.ndarray], key: str) -> str | None:
    if key not in archive:
        return None
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"ML density metadata {key!r} must be scalar")
    return str(value.reshape(()).item())


def _active_from_exact_dense(density: np.ndarray, grid: RealSpaceGrid) -> np.ndarray:
    if tuple(density.shape) != tuple(grid.lookup.shape):
        raise ValueError(
            "legacy dense ML density shape does not match the PARSEC Cartesian "
            f"grid: got {density.shape}, expected {grid.lookup.shape}; use a .npz "
            "file with origin_bohr and voxel_vectors_bohr to enable interpolation"
        )
    local = grid.integer_coordinates - grid.index_min[None, :]
    return np.asarray(density[local[:, 0], local[:, 1], local[:, 2]], dtype=float)


def _active_from_affine_dense(
    density: np.ndarray,
    grid: RealSpaceGrid,
    origin_bohr: np.ndarray,
    voxel_vectors_bohr: np.ndarray,
    interpolation: str,
) -> np.ndarray:
    origin = np.asarray(origin_bohr, dtype=float)
    vectors = np.asarray(voxel_vectors_bohr, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin_bohr must contain three finite values")
    if vectors.shape != (3, 3) or not np.all(np.isfinite(vectors)):
        raise ValueError("voxel_vectors_bohr must be a finite 3x3 matrix")
    determinant = float(np.linalg.det(vectors))
    if abs(determinant) < 1.0e-14:
        raise ValueError("voxel_vectors_bohr is singular")

    # Rows of ``vectors`` are the Cartesian displacement for increasing each
    # array index by one: r(i,j,k) = origin + [i,j,k] @ vectors.
    fractional_indices = (grid.coordinates - origin[None, :]) @ np.linalg.inv(vectors)
    order = 1 if interpolation == "linear" else 0
    return map_coordinates(
        np.asarray(density, dtype=float),
        fractional_indices.T,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _convert_to_bohr_density(
    values: np.ndarray,
    units: str,
    *,
    source_voxel_volume_bohr3: float,
) -> np.ndarray:
    if units == "e_per_bohr3":
        return np.asarray(values, dtype=float)
    if units == "e_per_angstrom3":
        return np.asarray(values, dtype=float) * BOHR_TO_ANGSTROM**3
    if units == "electrons_per_voxel":
        if not np.isfinite(source_voxel_volume_bohr3) or source_voxel_volume_bohr3 <= 0:
            raise ValueError("a positive source voxel volume is required for e/voxel")
        return np.asarray(values, dtype=float) / source_voxel_volume_bohr3
    raise ValueError(f"cannot convert density units {units!r}")


def load_density_for_grid(
    path: str | Path,
    grid: RealSpaceGrid,
    *,
    units: str = "auto",
    interpolation: str = "linear",
    negative_policy: str = "clip",
) -> DensityLoadResult:
    """Load, map, unit-convert, and validate an ML initial density.

    A bare legacy ``.npy`` array is interpreted as ``e/angstrom^3`` when
    ``units='auto'``, matching the old-architecture ML4Den path.  Portable
    ``.npz`` fields must carry a ``units`` scalar unless the caller explicitly
    supplies one.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ML density file does not exist: {source}")
    requested_units = normalize_density_units(units)
    interpolation = str(interpolation).strip().lower()
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("interpolation must be linear or nearest")
    negative_policy = str(negative_policy).strip().lower()
    if negative_policy not in {"clip", "error", "allow"}:
        raise ValueError("negative density policy must be clip, error, or allow")

    source_voxel_volume = grid.volume_element
    if source.suffix.lower() == ".npy":
        dense = np.load(source, allow_pickle=False)
        if dense.ndim != 3:
            raise ValueError("a legacy ML .npy density must be a 3D array")
        values = _active_from_exact_dense(dense, grid)
        resolved_units = (
            "e_per_angstrom3" if requested_units == "auto" else requested_units
        )
    elif source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            schema = _npz_string(archive, "schema")
            if schema is not None and schema != _SCHEMA:
                raise ValueError(f"unsupported ML density schema {schema!r}")
            key_units = {
                "density_e_per_bohr3": "e_per_bohr3",
                "density_e_per_angstrom3": "e_per_angstrom3",
                "density_electrons_per_voxel": "electrons_per_voxel",
            }
            density_key = next((key for key in key_units if key in archive), None)
            if density_key is None:
                density_key = "density" if "density" in archive else "values"
            if density_key not in archive:
                raise ValueError("ML density .npz does not contain a density array")
            density = np.asarray(archive[density_key], dtype=float)
            metadata_units = _npz_string(archive, "units")
            inferred_units = key_units.get(density_key)
            if requested_units != "auto":
                resolved_units = requested_units
                if metadata_units is not None:
                    declared = normalize_density_units(metadata_units)
                    if declared != resolved_units:
                        raise ValueError(
                            f"input requests {resolved_units}, but {source.name} declares {declared}"
                        )
            elif inferred_units is not None:
                resolved_units = inferred_units
            elif metadata_units is not None:
                resolved_units = normalize_density_units(metadata_units)
            else:
                raise ValueError(
                    "portable ML density .npz requires a units scalar or a unit-qualified key"
                )

            if density.ndim == 1:
                if density.size != grid.size:
                    raise ValueError(
                        f"point ML density has {density.size} values; PARSEC grid has {grid.size}"
                    )
                coordinate_key = (
                    "coordinates_bohr"
                    if "coordinates_bohr" in archive
                    else "probe_coordinates_bohr"
                )
                if coordinate_key not in archive:
                    raise ValueError(
                        "a point ML density must carry coordinates_bohr for grid validation"
                    )
                coordinates = np.asarray(archive[coordinate_key], dtype=float)
                tolerance = max(1.0e-10, 1.0e-8 * grid.spacing)
                if coordinates.shape != grid.coordinates.shape or not np.allclose(
                    coordinates, grid.coordinates, rtol=0.0, atol=tolerance
                ):
                    raise ValueError(
                        "point ML density coordinates do not match the ordered PARSEC grid"
                    )
                values = density
            elif density.ndim == 3:
                if "voxel_vectors_bohr" in archive or "origin_bohr" in archive:
                    if "voxel_vectors_bohr" not in archive or "origin_bohr" not in archive:
                        raise ValueError(
                            "affine dense density requires both origin_bohr and voxel_vectors_bohr"
                        )
                    vectors = np.asarray(archive["voxel_vectors_bohr"], dtype=float)
                    source_voxel_volume = abs(float(np.linalg.det(vectors)))
                    values = _active_from_affine_dense(
                        density,
                        grid,
                        np.asarray(archive["origin_bohr"], dtype=float),
                        vectors,
                        interpolation,
                    )
                else:
                    values = _active_from_exact_dense(density, grid)
            else:
                raise ValueError("ML density array must be one- or three-dimensional")
    else:
        raise ValueError("ML density input must use .npy or .npz")

    values = _convert_to_bohr_density(
        values,
        resolved_units,
        source_voxel_volume_bohr3=source_voxel_volume,
    )
    if values.shape != (grid.size,) or not np.all(np.isfinite(values)):
        raise ValueError("mapped ML density must be a finite vector on the PARSEC grid")
    integral_before = grid.integrate(values)
    negative_count = int(np.count_nonzero(values < 0.0))
    if negative_count and negative_policy == "error":
        raise ValueError(
            f"ML density contains {negative_count} negative values; use clip to correct them"
        )
    if negative_count and negative_policy == "clip":
        values = np.maximum(values, 0.0)
    integral_after = grid.integrate(values)
    if not np.isfinite(integral_after) or integral_after <= 0.0:
        raise ValueError("ML density has a nonpositive electron integral after validation")
    return DensityLoadResult(
        density=np.ascontiguousarray(values, dtype=float),
        source=source,
        units=resolved_units,
        integral_before_clipping=float(integral_before),
        integral_after_clipping=float(integral_after),
        negative_values_clipped=(negative_count if negative_policy == "clip" else 0),
    )


def save_point_density(
    path: str | Path,
    density: np.ndarray,
    coordinates_bohr: np.ndarray,
    *,
    units: str,
    provider: str,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Write the portable exact-point density format used by model adapters."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(density, dtype=float)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    if values.ndim != 1 or coordinates.shape != (values.size, 3):
        raise ValueError("point density and coordinates have inconsistent shapes")
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(_SCHEMA),
        "density": values,
        "coordinates_bohr": coordinates,
        "units": np.asarray(normalize_density_units(units)),
        "provider": np.asarray(str(provider)),
    }
    for key, value in (metadata or {}).items():
        payload[f"metadata_{key}"] = np.asarray(str(value))
    np.savez_compressed(output, **payload)
    return output


__all__ = [
    "BOHR_TO_ANGSTROM",
    "DensityLoadResult",
    "load_density_for_grid",
    "normalize_density_units",
    "save_point_density",
]
