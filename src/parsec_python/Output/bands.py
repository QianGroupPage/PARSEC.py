"""Band-structure files (Fortran ``bands.dat`` / ``bands_plot.dat``) and plots.

``bands.dat`` follows ``calc_bands.f90``::

    # of spins, # of states, # of segments, Ef (Ry)
    <spnum> <nstate> <nlines> <Ef>
    idx of segment, # of k-points
    <segment> <nkpt>              (one line per segment)
    spin, segment, kpt, kpt coord. (1/bohr), followed by eigs (Ry)
    <spin> <segment> <kpt>  kx ky kz        (format 3i6,3(3x,f12.8))
    <eigenvalue [Ry]>                       (one line per state)
    ...

``bands_plot.dat`` has one block per state: ``<k index>  <E [eV]>`` (one
column per spin), blocks separated by a blank line (gnuplot style).
Eigenvalues are *not* shifted by the Fermi level in either file.

Spinor bands (spin-orbit, non-collinear) also get ``bands_spin.dat``, for spin-texture plots: per state a
block of ``k index, path length [1/bohr], E [eV], <S_x>, <S_y>, <S_z>, degeneracy`` (gnuplot style), with
``<S_i>`` in units of hbar/2 (the expectation of the Pauli matrix, between -1 and 1) in the axes of the cell.
The spin of a state inside a degenerate level (Kramers pairs at time-reversal-invariant points, or any
level of an inversion-symmetric crystal) is not defined by the level: any mixture of the states is as good an
eigenvector.  The sum over the level is, so the file gives every state of a degenerate level the mean of the
level (written in the last column, the number of states in it).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from ..models import BandStructureResult

RYDBERG_TO_EV = 13.605693122994


def write_bands_files(directory: str | Path, result: BandStructureResult) -> tuple[Path, Path]:
    """Write ``bands.dat`` and ``bands_plot.dat`` into ``directory``."""
    directory = Path(directory)
    spin_count, k_count, state_count = result.eigenvalues.shape
    counts = [int(value) for value in result.segment_counts]
    lines = [
        " # of spins, # of states, # of segments, Ef (Ry)",
        f" {spin_count} {state_count} {len(counts)} {result.fermi_level:.16E}",
        " idx of segment, # of k-points",
    ]
    lines.extend(f" {index} {count}" for index, count in enumerate(counts, start=1))
    lines.append(" spin, segment, kpt, kpt coord. (1/bohr), followed by eigs (Ry)")
    offsets = np.concatenate([[0], np.cumsum(counts)])
    for spin in range(spin_count):
        for segment, count in enumerate(counts, start=1):
            for point in range(count):
                row = offsets[segment - 1] + point
                kx, ky, kz = result.k_points[row]
                lines.append(
                    f"{spin + 1:6d}{segment:6d}{point + 1:6d}"
                    f"   {kx:12.8f}   {ky:12.8f}   {kz:12.8f}"
                )
                lines.extend(
                    f" {value:24.16E}" for value in result.eigenvalues[spin, row]
                )
    bands_path = directory / "bands.dat"
    bands_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_lines = []
    for state in range(state_count):
        for row in range(k_count):
            columns = "".join(
                f"   {result.eigenvalues[spin, row, state] * RYDBERG_TO_EV:12.8f}"
                for spin in range(spin_count)
            )
            plot_lines.append(f"{row + 1:6d}   {columns}")
        plot_lines.append(" ")
    plot_path = directory / "bands_plot.dat"
    plot_path.write_text("\n".join(plot_lines) + "\n", encoding="utf-8")
    return bands_path, plot_path


DEGENERACY_TOLERANCE_RY = 1.0e-4


def degenerate_levels(energies: np.ndarray, tolerance: float = DEGENERACY_TOLERANCE_RY) -> list[np.ndarray]:
    """Groups of consecutive states of ``energies`` (ascending) that lie within ``tolerance`` of their neighbour."""
    energies = np.asarray(energies, dtype=float)
    order = np.argsort(energies, kind="stable")
    groups: list[list[int]] = [[int(order[0])]] if energies.size else []
    for previous, current in zip(order[:-1], order[1:]):
        if energies[current] - energies[previous] <= tolerance:
            groups[-1].append(int(current))
        else:
            groups.append([int(current)])
    return [np.asarray(group) for group in groups]


def level_averaged_spin(
    result: BandStructureResult, tolerance: float = DEGENERACY_TOLERANCE_RY
) -> tuple[np.ndarray, np.ndarray]:
    """``<sigma>`` of every state with each degenerate level replaced by its mean, and the size of the level.

    Returns arrays of shape ``(k, states, 3)`` and ``(k, states)``.  Only the sum of the spin over a degenerate
    level is determined by the Hamiltonian, so the mean is what a plot can show.
    """
    if result.spin_expectation is None:
        raise ValueError("this band structure has no spin expectation values (it is not a spinor calculation)")
    spin = np.asarray(result.spin_expectation)[0]
    energies = result.eigenvalues[0]
    averaged = spin.copy()
    sizes = np.ones(energies.shape, dtype=int)
    for k in range(energies.shape[0]):
        for level in degenerate_levels(energies[k], tolerance):
            if level.size > 1:
                averaged[k, level] = spin[k, level].mean(axis=0)
                sizes[k, level] = level.size
    return averaged, sizes


def write_spin_file(
    directory: str | Path, result: BandStructureResult, tolerance: float = DEGENERACY_TOLERANCE_RY
) -> Path | None:
    """Write ``bands_spin.dat`` (see the module docstring); ``None`` for a band structure without spin."""
    if result.spin_expectation is None:
        return None
    averaged, sizes = level_averaged_spin(result, tolerance)
    energies = result.eigenvalues[0] * RYDBERG_TO_EV
    distance = result.path_distance
    lines = [
        "# <S_x>, <S_y>, <S_z> of every band along the path, in units of hbar/2 (the Pauli expectation),",
        "# in the axes of the cell; E in eV, not shifted by the Fermi level; path length in 1/bohr.",
        f"# a degenerate level (states within {tolerance:g} Ry) shows the mean spin of the level;"
        " the last column is its size.",
        f"# Fermi level: {result.fermi_level * RYDBERG_TO_EV:.8f} eV",
        "#   k      path [1/bohr]       E [eV]         S_x          S_y          S_z      deg",
    ]
    states = energies.shape[1]
    for state in range(states):
        for row in range(energies.shape[0]):
            sx, sy, sz = averaged[row, state]
            lines.append(
                f"{row + 1:6d}  {distance[row]:14.8f}  {energies[row, state]:14.8f}"
                f"  {sx:11.6f}  {sy:11.6f}  {sz:11.6f}  {int(sizes[row, state]):4d}"
            )
        lines.append(" ")
    path = Path(directory) / "bands_spin.dat"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_bands_dat(path: str | Path) -> BandStructureResult:
    """Read a ``bands.dat`` (from this package or Fortran PARSEC)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    spin_count, state_count, segment_count = (int(value) for value in lines[1].split()[:3])
    fermi_level = float(lines[1].split()[3].lower().replace("d", "e"))
    counts = [int(lines[3 + index].split()[1]) for index in range(segment_count)]
    cursor = 3 + segment_count + 1
    k_total = sum(counts)
    k_points = np.zeros((k_total, 3))
    eigenvalues = np.zeros((spin_count, k_total, state_count))
    for spin in range(spin_count):
        row = 0
        for count in counts:
            for _ in range(count):
                fields = lines[cursor].split()
                k_points[row] = [float(value) for value in fields[3:6]]
                eigenvalues[spin, row] = [
                    float(lines[cursor + 1 + state].split()[0].lower().replace("d", "e"))
                    for state in range(state_count)
                ]
                cursor += 1 + state_count
                row += 1
    return BandStructureResult(
        mode="file",
        channels=("up", "down") if spin_count == 2 else ("all",),
        segment_names=("",) * segment_count,
        segment_counts=np.asarray(counts),
        k_points=k_points,
        eigenvalues=eigenvalues,
        residual_norms=np.zeros((spin_count, k_total)),
        fermi_level=fermi_level,
    )


def save_band_structure_archive(path: str | Path, result: BandStructureResult) -> Path:
    """Portable NumPy archive of a band structure (with segment names)."""
    path = Path(path)
    payload = {
        "mode": np.asarray(result.mode),
        "channels": np.asarray(result.channels),
        "segment_names": np.asarray(result.segment_names),
        "segment_counts": np.asarray(result.segment_counts),
        "k_points_cartesian_inv_bohr": result.k_points,
        "path_distance": result.path_distance,
        "eigenvalues_ry": result.eigenvalues,
        "residual_norms": result.residual_norms,
        "fermi_level_ry": np.asarray(result.fermi_level),
    }
    if result.spin_expectation is not None:
        payload["spin_expectation"] = result.spin_expectation
    if result.attempts is not None:
        payload["solver_attempts"] = result.attempts
    np.savez_compressed(path, **payload)
    return path


def load_band_structure_archive(path: str | Path) -> BandStructureResult:
    archive = np.load(path)
    return BandStructureResult(
        mode=str(archive["mode"]),
        channels=tuple(str(value) for value in archive["channels"]),
        segment_names=tuple(str(value) for value in archive["segment_names"]),
        segment_counts=archive["segment_counts"],
        k_points=archive["k_points_cartesian_inv_bohr"],
        eigenvalues=archive["eigenvalues_ry"],
        residual_norms=archive["residual_norms"],
        fermi_level=float(archive["fermi_level_ry"]),
        spin_expectation=archive["spin_expectation"] if "spin_expectation" in archive.files else None,
        attempts=archive["solver_attempts"] if "solver_attempts" in archive.files else None,
    )


def _tick_labels(result: BandStructureResult) -> tuple[list[float], list[str]]:
    """Plot ticks at the segment boundaries; ``A-B`` names give the vertex labels."""
    distance = result.path_distance
    counts = [int(value) for value in result.segment_counts]
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    positions, labels = [], []
    for index, (start, name) in enumerate(zip(starts, result.segment_names)):
        left, _, right = name.partition("-")
        positions.append(float(distance[start]))
        labels.append(left if _ else "")
        if index == len(counts) - 1:
            positions.append(float(distance[-1]))
            labels.append(right if _ else "")
    merged_positions, merged_labels = [], []
    for position, label in zip(positions, labels):
        if merged_positions and abs(position - merged_positions[-1]) < 1e-12:
            if label and label != merged_labels[-1]:
                merged_labels[-1] = (
                    f"{merged_labels[-1]}|{label}" if merged_labels[-1] else label
                )
            continue
        merged_positions.append(position)
        merged_labels.append(label)
    return merged_positions, merged_labels


def plot_band_comparison(
    entries: Sequence[tuple[str, BandStructureResult]],
    *,
    window_ev: tuple[float, float] | None = (-8.0, 6.0),
    output: str | Path | None = None,
    title: str | None = None,
):
    """Overlay band structures (energy relative to each run's Fermi level).

    ``entries`` is ``[(label, result), ...]``, e.g. without and with
    spin-orbit coupling.  Spin-polarized channels share the entry's color
    (the down channel dashed).  Returns the matplotlib figure; saves to
    ``output`` when given.
    """
    import matplotlib

    if output is not None:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    lss = ("-", "--", ":", "-.")
    reference = entries[0][1]
    for index, (label, result) in enumerate(entries):
        color = colors[index % len(colors)]
        ls = lss[index % len(lss)]
        distance = result.path_distance
        for channel in range(result.eigenvalues.shape[0]):
            energies = (result.eigenvalues[channel] - result.fermi_level) * RYDBERG_TO_EV
            for band in range(energies.shape[1]):
                axis.plot(
                    distance,
                    energies[:, band],
                    color=color,
                    lw=1.0,
                    ls=ls,
                    alpha=0.85,
                    label=label if (band == 0 and channel == 0) else None,
                )
    positions, labels = _tick_labels(reference)
    for position in positions:
        axis.axvline(position, color="0.75", lw=0.6)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.axhline(0.0, color="0.4", lw=0.6, ls=":")
    axis.set_xlim(reference.path_distance[0], reference.path_distance[-1])
    if window_ev is not None:
        axis.set_ylim(*window_ev)
    axis.set_ylabel("E - E$_F$ (eV)")
    if title:
        axis.set_title(title)
    axis.legend(loc="upper right", frameon=False)
    figure.tight_layout()
    if output is not None:
        figure.savefig(output, dpi=160)
    return figure


def plot_spin_texture(
    result: BandStructureResult,
    *,
    components: str = "xyz",
    window_ev: tuple[float, float] | None = (-2.0, 2.0),
    output: str | Path | None = None,
    tolerance: float = DEGENERACY_TOLERANCE_RY,
    title: str | None = None,
):
    """Bands coloured by ``<S_x>``, ``<S_y>`` and/or ``<S_z>`` (one panel per component in ``components``).

    Energy relative to the Fermi level; the colour of a segment of a band is the spin of its two end points,
    from -1 (blue) to +1 (red) in units of hbar/2.  Degenerate levels show their mean spin
    (:func:`level_averaged_spin`).  Returns the matplotlib figure; saves it to ``output`` when given.
    """
    import matplotlib

    if output is not None:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    averaged, _ = level_averaged_spin(result, tolerance)
    components = components.lower()
    if not components or any(c not in "xyz" for c in components):
        raise ValueError("components is a string of x, y and z")
    energies = (result.eigenvalues[0] - result.fermi_level) * RYDBERG_TO_EV
    distance = result.path_distance
    figure, axes = plt.subplots(
        1, len(components), figsize=(4.6 * len(components) + 0.8, 5.0), sharey=True, squeeze=False
    )
    mappable = None
    positions, labels = _tick_labels(result)
    for axis, component in zip(axes[0], components):
        values = averaged[:, :, "xyz".index(component)]
        for band in range(energies.shape[1]):
            points = np.column_stack((distance, energies[:, band]))
            segments = np.stack((points[:-1], points[1:]), axis=1)
            colours = 0.5 * (values[:-1, band] + values[1:, band])
            collection = LineCollection(segments, cmap="coolwarm", norm=plt.Normalize(-1.0, 1.0), linewidths=2.0)
            collection.set_array(colours)
            axis.add_collection(collection)
            mappable = collection
        for position in positions:
            axis.axvline(position, color="0.75", lw=0.6)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.axhline(0.0, color="0.4", lw=0.6, ls=":")
        axis.set_xlim(distance[0], distance[-1])
        if window_ev is not None:
            axis.set_ylim(*window_ev)
        else:
            axis.set_ylim(energies.min() - 0.2, energies.max() + 0.2)
        axis.set_title(f"$\\langle S_{component}\\rangle$")
    axes[0][0].set_ylabel("E - E$_F$ (eV)")
    colourbar = figure.colorbar(mappable, ax=axes[0].tolist(), pad=0.02)
    colourbar.set_label(r"spin ($\hbar/2$)")
    if title:
        figure.suptitle(title)
    if output is not None:
        figure.savefig(output, dpi=160, bbox_inches="tight")
    return figure


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay band structures from bands.dat files (e.g. without and with SOC), or with --spin plot the "
            "spin texture of the bands of a spinor calculation from its parsec_python_bands.npz."
        )
    )
    parser.add_argument("bands", nargs="*", type=Path, help="bands.dat files")
    parser.add_argument("--spin", type=Path, metavar="BANDS.NPZ",
                        help="plot <S_x>, <S_y>, <S_z> along the path from this band-structure archive")
    parser.add_argument("--components", default="xyz", help="which of x, y, z to plot with --spin (default xyz)")
    parser.add_argument("--tolerance", type=float, default=DEGENERACY_TOLERANCE_RY,
                        help="Ry within which states form a degenerate level whose mean spin is shown (--spin)")
    parser.add_argument("--labels", nargs="+", help="legend label per file")
    parser.add_argument("--names", help="segment names for the ticks, e.g. 'G-X,X-M,M-G'")
    parser.add_argument("--window", nargs=2, type=float, default=None, metavar=("EMIN", "EMAX"),
                        help="energy window relative to E_F in eV (default -8 6; -2 2 with --spin)")
    parser.add_argument("-o", "--output", type=Path, default=Path("bands.png"))
    arguments = parser.parse_args(argv)
    if arguments.spin is not None:
        result = load_band_structure_archive(arguments.spin)
        if arguments.names:
            names = tuple(arguments.names.split(","))
            if len(names) != len(result.segment_counts):
                parser.error("--names needs one name per path segment")
            result.segment_names = names
        output = arguments.output if arguments.output != Path("bands.png") else Path("bands_spin.png")
        plot_spin_texture(result, components=arguments.components,
                          window_ev=tuple(arguments.window) if arguments.window else (-2.0, 2.0),
                          output=output, tolerance=arguments.tolerance)
        print(f"wrote {output}")
        return 0
    if not arguments.bands:
        parser.error("give bands.dat files, or --spin BANDS.NPZ")
    labels = arguments.labels or [path.parent.name or path.name for path in arguments.bands]
    if len(labels) != len(arguments.bands):
        parser.error("give one --labels entry per bands file")
    entries = []
    for label, path in zip(labels, arguments.bands):
        result = read_bands_dat(path)
        if arguments.names:
            names = tuple(arguments.names.split(","))
            if len(names) != len(result.segment_counts):
                parser.error("--names needs one name per path segment")
            result.segment_names = names
        entries.append((label, result))
    plot_band_comparison(
        entries, window_ev=tuple(arguments.window) if arguments.window else (-8.0, 6.0), output=arguments.output
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEGENERACY_TOLERANCE_RY",
    "degenerate_levels",
    "level_averaged_spin",
    "load_band_structure_archive",
    "plot_spin_texture",
    "write_spin_file",
    "plot_band_comparison",
    "read_bands_dat",
    "save_band_structure_archive",
    "write_bands_files",
]
