"""``afm_scan.dat``: the table of an AFM scan of an FDET tip (:mod:`parsec_python.Embedding.afm`)."""

from __future__ import annotations

import os
from pathlib import Path

from ..Embedding.afm import (
    RYDBERG_PER_BOHR2_IN_N_PER_M,
    AFMScanSettings,
    force_gradients,
    scan_points,
    unconverged_gradients,
)

#: The scan table a run writes beside its input.
AFM_SCAN_FILE = "afm_scan.dat"

#: The file that gives a folder to one scan, or to one part of a scan (:func:`claim_scan_folder`).
SCAN_FOLDER_MARK = "afm_scan_part.lock"


def scan_table_lines(settings: AFMScanSettings, points) -> list[str]:
    """The text of ``afm_scan.dat``: one row per point, sorted by line and height."""

    gradients = force_gradients(settings, points)
    left_out = unconverged_gradients(settings, points)
    lines = [
        "# AFM scan of an FDET tip (PARSEC.py): E(x, y, z) and k_ts = d2E/dz2 by three-point finite differences",
        f"# heights {settings.heights}, dz = {settings.height_step:.6f} bohr; displacements of the tip from the input "
        "structure in bohr; no tilt correction",
        "# k_ts '-': end of a line or a point not run (yet); 'unconv': one of its three SCF loops did not converge",
        "# line  dx[bohr]  dy[bohr]  dz[bohr]  E[Ry]  k_ts[Ry/bohr^2]  k_ts[N/m]  converged",
    ]
    for point in sorted(points, key=lambda item: (item.line, item.height)):
        key = (point.line, point.height)
        gradient = gradients.get(key)
        dx, dy, dz = (float(value) for value in point.displacement)
        mark = "unconv" if key in left_out else "-"
        lines.append(
            f"{point.line + 1:5d} {dx:10.5f} {dy:10.5f} {dz:10.5f} {point.energy:20.10f} "
            + (
                f"{gradient:16.8e} {gradient * RYDBERG_PER_BOHR2_IN_N_PER_M:12.6f}"
                if gradient is not None
                else f"{mark:>16s} {mark:>12s}"
            )
            + f" {'yes' if point.converged else 'no'}"
        )
    return lines


def scan_table_name(settings: AFMScanSettings) -> str:
    """``afm_scan.dat``, or ``afm_scan_lines_a-b.dat`` for the part ``a..b`` of a scan."""

    if settings.line_range is None:
        return AFM_SCAN_FILE
    return f"afm_scan_lines_{settings.line_range[0]}-{settings.line_range[1]}.dat"


def write_scan_table(directory, settings: AFMScanSettings, points) -> Path:
    """Write ``afm_scan.dat`` (with the line range in its name for a part of a scan).

    The table goes to a temporary file of this process first and replaces the old one in one step, so that a
    run stopped while writing leaves the table of the points before.
    """

    path = Path(directory) / scan_table_name(settings)
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    partial.write_text("\n".join(scan_table_lines(settings, points)) + "\n", encoding="ascii")
    os.replace(partial, path)
    return path


def write_scan_progress(directory, settings: AFMScanSettings, records) -> Path:
    """The table of the structures finished so far, written after every one so that a later failure keeps them."""

    records = list(records)
    points = scan_points(settings, [record.energy for record in records], [record.converged for record in records])
    return write_scan_table(directory, settings, points)


def claim_scan_folder(translation, directory) -> str | None:
    """Give the folder ``directory`` to the scan of ``translation``; refuse a folder of another part of a scan.

    The parts of a scan (``AFM_Scan_Line_Range``) write the same fixed-name files beside their inputs
    (parsec.out, the archive, atom.cor, relax_restart.dat, parsec.dat, ...), so each part needs a folder of its
    own.  The first scan run in a folder writes ``afm_scan_part.lock`` with its line range ("all" for a whole
    scan), created exclusively so that two parts started at once cannot both take the folder; a run of another
    range there is refused, a rerun of the same range is let through.  Returns the line for parsec.out,
    ``None`` without a scan.
    """

    settings = getattr(translation, "afm_scan", None)
    if settings is None:
        return None
    label = "all" if settings.line_range is None else f"{settings.line_range[0]}-{settings.line_range[1]}"
    path = Path(directory) / SCAN_FOLDER_MARK
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        owner = path.read_text(encoding="ascii", errors="replace").strip()
        if owner != label:
            raise ValueError(
                f"{path} gives this folder to the AFM scan lines {owner}, this run is the lines {label}: every "
                "part of a scan (AFM_Scan_Line_Range) needs a folder of its own, the parts write the same "
                "parsec.out, archive, atom.cor and restart files (remove the file if no other part runs here)"
            ) from None
    else:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(label + "\n")
    return f" AFM scan lines {label}: folder marked by {SCAN_FOLDER_MARK}"


def scan_lines_after_run(translation, outcome, write, directory) -> Path | None:
    """Write the table of the scan of ``translation`` from the ionic ``outcome``; ``None`` without a scan."""

    settings = getattr(translation, "afm_scan", None)
    if settings is None or outcome is None:
        return None
    records = list(outcome.records)
    points = scan_points(settings, [record.energy for record in records], [record.converged for record in records])
    path = write_scan_table(directory, settings, points)
    gradients = force_gradients(settings, points)
    left_out = unconverged_gradients(settings, points)
    write(
        f" AFM scan: {len(points)} tip positions, {len(gradients)} force gradients k_ts"
        + (f" ({len(left_out)} left out: an SCF loop did not converge)" if left_out else "")
        + f": {path}"
    )
    for line in scan_table_lines(settings, points):
        write(f" {line}")
    return path


__all__ = [
    "AFM_SCAN_FILE",
    "SCAN_FOLDER_MARK",
    "claim_scan_folder",
    "scan_lines_after_run",
    "scan_table_lines",
    "scan_table_name",
    "write_scan_progress",
    "write_scan_table",
]
