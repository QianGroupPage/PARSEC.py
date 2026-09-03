#!/usr/bin/env python3
"""Create a Figure-5-style comparison of CHEBFF and CHEBDAV.

The script reads the preserved cold-run suite summaries rather than parsing the
individual output files again:

* ``results_summary_chebff.csv`` -- CHEBFF first diagonalization
* ``results_summary.csv`` -- CHEBDAV first diagonalization

Each solver was followed by the same subspace/CheFSI SCF path.  The wall-time
panels use ``solver_wall_seconds``: elapsed time inside the calculation driver,
including setup and SCF but excluding Python interpreter startup in the suite
launcher.  This is the most directly comparable timing recorded by both runs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent

MOLECULE_ORDER = (
    "C10H8",
    "C2H2",
    "C2H4",
    "C2H6",
    "C3H8O",
    "C6H6",
    "CH3CH2CH2OH",
    "CH3CH2OCH3",
    "CH3CH2OH",
    "CH3CHOHCH3",
    "CH3CN",
    "CH4",
    "CO2",
    "H2O",
)

# Order and colors follow Figure 5: SAD (blue), learned model (green), SCDP
# (orange).  The repository uses the full name ChargE3Net rather than CE3N.
METHODS = (
    ("sad", "SAD", "#4C78A8"),
    ("charge3net", "ChargE3Net", "#54A24B"),
    ("scdp", "SCDP", "#F58518"),
)

MOLECULE_LABELS = {
    "C10H8": r"C$_{10}$H$_8$",
    "C2H2": r"C$_2$H$_2$",
    "C2H4": r"C$_2$H$_4$",
    "C2H6": r"C$_2$H$_6$",
    "C3H8O": r"C$_3$H$_8$O",
    "C6H6": r"C$_6$H$_6$",
    "CH3CH2CH2OH": r"CH$_3$CH$_2$CH$_2$OH",
    "CH3CH2OCH3": r"CH$_3$CH$_2$OCH$_3$",
    "CH3CH2OH": r"CH$_3$CH$_2$OH",
    "CH3CHOHCH3": r"CH$_3$CHOHCH$_3$",
    "CH3CN": r"CH$_3$CN",
    "CH4": r"CH$_4$",
    "CO2": r"CO$_2$",
    "H2O": r"H$_2$O",
}


@dataclass(frozen=True)
class Result:
    molecule: str
    method: str
    iterations: int
    total_energy_ry: float
    scf_seconds: float
    solver_wall_seconds: float


def read_summary(path: Path) -> dict[tuple[str, str], Result]:
    """Read and validate one complete converged 14-molecule/3-guess suite."""

    records: dict[tuple[str, str], Result] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["status"].strip().lower() != "converged":
                raise ValueError(
                    f"{path.name}: {row['molecule']}/{row['method']} did not converge"
                )
            result = Result(
                molecule=row["molecule"],
                method=row["method"].lower(),
                iterations=int(row["iterations"]),
                total_energy_ry=float(row["total_energy_ry"]),
                scf_seconds=float(row["scf_seconds"]),
                solver_wall_seconds=float(row["solver_wall_seconds"]),
            )
            key = (result.molecule, result.method)
            if key in records:
                raise ValueError(f"{path.name}: duplicate result for {key}")
            records[key] = result

    expected = {
        (molecule, method)
        for molecule in MOLECULE_ORDER
        for method, _label, _color in METHODS
    }
    missing = expected - records.keys()
    extra = records.keys() - expected
    if missing or extra:
        raise ValueError(
            f"{path.name}: suite mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return records


def metric_values(
    records: dict[tuple[str, str], Result], method: str, metric: str
) -> np.ndarray:
    """Return one metric in the fixed molecule order used on the x axis."""

    return np.asarray(
        [getattr(records[(molecule, method)], metric) for molecule in MOLECULE_ORDER],
        dtype=float,
    )


def add_grouped_bars(
    ax: plt.Axes,
    records: dict[tuple[str, str], Result],
    metric: str,
    ylabel: str,
    title: str,
    panel: str,
    common_ylim: tuple[float, float],
) -> None:
    """Draw one Figure-5-style panel with values and dashed method means."""

    x = np.arange(len(MOLECULE_ORDER), dtype=float)
    bar_width = 0.245
    offsets = (-bar_width, 0.0, bar_width)
    is_iteration = metric == "iterations"

    for offset, (method, label, color) in zip(offsets, METHODS, strict=True):
        values = metric_values(records, method, metric)
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        mean = float(values.mean())
        ax.axhline(mean, color=color, linestyle=(0, (5, 3)), linewidth=1.3, zorder=2)
        mean_text = f"{mean:.1f}"
        ax.text(
            len(MOLECULE_ORDER) - 0.05,
            mean,
            mean_text,
            color=color,
            fontsize=7.5,
            fontweight="semibold",
            va="center",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.8},
            clip_on=False,
        )

        labels = [f"{int(value)}" if is_iteration else f"{value:.1f}" for value in values]
        ax.bar_label(
            bars,
            labels=labels,
            padding=1.5,
            fontsize=6.2,
            rotation=90,
            color="#333333",
        )

    ax.set_ylim(*common_ylim)
    ax.set_xlim(-0.65, len(MOLECULE_ORDER) + 0.62)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.text(
        0.006,
        0.965,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xticks(x, [MOLECULE_LABELS[name] for name in MOLECULE_ORDER])
    ax.tick_params(axis="x", labelrotation=42, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def write_combined_data(
    path: Path,
    chebff: dict[tuple[str, str], Result],
    chebdav: dict[tuple[str, str], Result],
) -> None:
    """Write the paired values behind the figure for easy auditing/reuse."""

    fields = (
        "molecule",
        "method",
        "chebff_iterations",
        "chebdav_iterations",
        "chebff_energy_ry",
        "chebdav_energy_ry",
        "energy_difference_ry",
        "chebff_scf_seconds",
        "chebdav_scf_seconds",
        "chebff_solver_wall_seconds",
        "chebdav_solver_wall_seconds",
        "solver_wall_speedup",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for molecule in MOLECULE_ORDER:
            for method, _label, _color in METHODS:
                old = chebff[(molecule, method)]
                new = chebdav[(molecule, method)]
                writer.writerow(
                    {
                        "molecule": molecule,
                        "method": method,
                        "chebff_iterations": old.iterations,
                        "chebdav_iterations": new.iterations,
                        "chebff_energy_ry": f"{old.total_energy_ry:.8f}",
                        "chebdav_energy_ry": f"{new.total_energy_ry:.8f}",
                        "energy_difference_ry": f"{new.total_energy_ry - old.total_energy_ry:.8e}",
                        "chebff_scf_seconds": f"{old.scf_seconds:.6f}",
                        "chebdav_scf_seconds": f"{new.scf_seconds:.6f}",
                        "chebff_solver_wall_seconds": f"{old.solver_wall_seconds:.6f}",
                        "chebdav_solver_wall_seconds": f"{new.solver_wall_seconds:.6f}",
                        "solver_wall_speedup": f"{old.solver_wall_seconds / new.solver_wall_seconds:.6f}",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chebff",
        type=Path,
        default=ROOT / "results_summary_chebff.csv",
        help="CHEBFF suite summary CSV",
    )
    parser.add_argument(
        "--chebdav",
        type=Path,
        default=ROOT / "results_summary.csv",
        help="CHEBDAV suite summary CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures",
        help="directory for PNG, PDF, and paired figure data",
    )
    args = parser.parse_args()

    chebff = read_summary(args.chebff.resolve())
    chebdav = read_summary(args.chebdav.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_iterations = np.concatenate(
        [
            metric_values(records, method, "iterations")
            for records in (chebff, chebdav)
            for method, _label, _color in METHODS
        ]
    )
    all_wall = np.concatenate(
        [
            metric_values(records, method, "solver_wall_seconds")
            for records in (chebff, chebdav)
            for method, _label, _color in METHODS
        ]
    )
    iteration_ylim = (0.0, float(np.ceil((all_iterations.max() + 3.0) / 5.0) * 5.0))
    wall_ylim = (0.0, float(np.ceil((all_wall.max() + 1.5) / 5.0) * 5.0))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(24, 11.5), constrained_layout=False)

    add_grouped_bars(
        axes[0, 0],
        chebff,
        "iterations",
        "SCF iterations",
        "CHEBFF 1st + SUBSPACE",
        "(a)",
        iteration_ylim,
    )
    add_grouped_bars(
        axes[0, 1],
        chebff,
        "solver_wall_seconds",
        "Solver wall time (s)",
        "CHEBFF 1st + SUBSPACE",
        "(b)",
        wall_ylim,
    )
    add_grouped_bars(
        axes[1, 0],
        chebdav,
        "iterations",
        "SCF iterations",
        "CHEBDAV 1st + SUBSPACE",
        "(c)",
        iteration_ylim,
    )
    add_grouped_bars(
        axes[1, 1],
        chebdav,
        "solver_wall_seconds",
        "Solver wall time (s)",
        "CHEBDAV 1st + SUBSPACE",
        "(d)",
        wall_ylim,
    )

    # Keep molecule labels only on the lower row, as in Figure 5.
    for ax in axes[0, :]:
        ax.tick_params(axis="x", labelbottom=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.994),
        fontsize=11,
    )
    fig.suptitle(
        "CHEBFF and CHEBDAV convergence and timing with different initial densities",
        fontsize=15,
        y=0.955,
    )
    fig.text(
        0.5,
        0.012,
        "Dashed lines and colored numbers show the per-density-guess averages over 14 molecules.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.055, right=0.982, top=0.905, bottom=0.105, hspace=0.22, wspace=0.11)

    stem = args.output_dir / "chebff_chebdav_iterations_timing"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    write_combined_data(args.output_dir / "chebff_chebdav_figure_data.csv", chebff, chebdav)

    max_energy_difference = max(
        abs(chebdav[key].total_energy_ry - chebff[key].total_energy_ry) for key in chebff
    )
    mean_speedup = np.mean(
        [chebff[key].solver_wall_seconds / chebdav[key].solver_wall_seconds for key in chebff]
    )
    print(f"Wrote {stem.with_suffix('.png')}")
    print(f"Wrote {stem.with_suffix('.pdf')}")
    print(f"Wrote {args.output_dir / 'chebff_chebdav_figure_data.csv'}")
    print(f"Maximum |CHEBDAV - CHEBFF| final energy: {max_energy_difference:.3e} Ry")
    print(f"Mean paired CHEBDAV wall-time speedup: {mean_speedup:.3f}x")


if __name__ == "__main__":
    main()
