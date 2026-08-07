from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


SCF_PATTERN = re.compile(r"SCF iter #\s+(\d+)")
DIAG_PATTERN = re.compile(r"Diagonalization time \[sec\] :\s+([0-9.eE+-]+)")
HARTREE_PATTERN = re.compile(r"Hartree potential time \[sec\]:\s+([0-9.eE+-]+)")
TOTAL_PATTERN = re.compile(r"Total runtime \[sec\]\s*=\s*([0-9.eE+-]+)")
OUTFILE_PATTERN = re.compile(r"diagmeth(?P<method>\d+)[\\/](?P<backend>cpu|gpu)[\\/].+\.out$", re.IGNORECASE)
METHOD_LABELS = {
    0: "Lanczos",
    2: "Chsubsp",
    3: "FirstFilt",
}


def parse_output_file(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    diag_times = [float(match.group(1)) for match in DIAG_PATTERN.finditer(text)]
    hartree_times = [float(match.group(1)) for match in HARTREE_PATTERN.finditer(text)]
    total_match = TOTAL_PATTERN.search(text)

    if not diag_times:
        raise ValueError(f"No diagonalization times found in {path}")
    if not hartree_times:
        raise ValueError(f"No Hartree times found in {path}")

    post_first_diag_times = diag_times[1:]
    avg_post_first_diag = float(np.mean(post_first_diag_times)) if post_first_diag_times else 0.0

    return {
        "first_diag_time": diag_times[0],
        "avg_post_first_diag_time": avg_post_first_diag,
        "avg_hartree_time": float(np.mean(hartree_times)),
        "total_runtime": float(total_match.group(1)) if total_match is not None else float("nan"),
        "n_scf_iterations": float(len(diag_times)),
    }


def collect_metrics(root: Path) -> dict[int, dict[str, dict[str, float]]]:
    results: dict[int, dict[str, dict[str, float]]] = {}
    for out_file in sorted(root.rglob("*.out")):
        match = OUTFILE_PATTERN.search(str(out_file))
        if match is None:
            continue

        method = int(match.group("method"))
        backend = match.group("backend").lower()
        results.setdefault(method, {})[backend] = parse_output_file(out_file)

    if not results:
        raise ValueError(f"No h8c10 timing outputs found under {root}")

    return results


def add_value_labels(ax: plt.Axes, bars, values: list[float], value_fontsize: float) -> None:
    finite_values = [value for value in values if np.isfinite(value)]
    baseline = max(finite_values) * 0.02 if finite_values else 0.1

    for bar, value in zip(bars, values):
        x_center = bar.get_x() + bar.get_width() / 2
        if np.isfinite(value):
            ax.annotate(
                f"{value:.2f}",
                xy=(x_center, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
                fontweight="bold",
                rotation=0,
            )
        else:
            ax.annotate(
                "N/A",
                xy=(x_center, baseline),
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
                fontweight="bold",
                rotation=0,
                color="#666666",
            )


def plot_metrics(
    results: dict[int, dict[str, dict[str, float]]],
    output_path: Path,
    bar_width: float,
    group_gap: float,
    metric_spacing: float,
    title_fontsize: float,
    axis_label_fontsize: float,
    tick_fontsize: float,
    legend_fontsize: float,
    value_fontsize: float,
) -> None:
    methods = sorted(results)
    method_labels = [METHOD_LABELS.get(method, f"diagmeth{method}") for method in methods]
    metric_specs = [
        ("first_diag_time", "1st SCF diag time", "#2b6cb0"),
        ("avg_post_first_diag_time", "Avg diag time after 1st SCF", "#38a169"),
        ("avg_hartree_time", "Avg Hartree potential time", "#ed8936"),
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    backend_offsets = {"cpu": -bar_width / 2, "gpu": bar_width / 2}
    backend_hatches = {"cpu": "", "gpu": "//"}
    backend_labels = {"cpu": "CPU", "gpu": "GPU"}

    group_centers = []
    all_values: list[float] = []

    for group_index, method in enumerate(methods):
        group_origin = group_index * group_gap
        metric_centers = [group_origin + metric_spacing * metric_index for metric_index in range(len(metric_specs))]
        group_centers.append(float(np.mean(metric_centers)))

        for metric_center, (metric_key, _, color) in zip(metric_centers, metric_specs):
            for backend in ("cpu", "gpu"):
                value = results[method][backend][metric_key]
                position = metric_center + backend_offsets[backend]
                bars = ax.bar(
                    [position],
                    [value],
                    bar_width,
                    color=color,
                    edgecolor="black",
                    linewidth=0.8,
                    hatch=backend_hatches[backend],
                )
                add_value_labels(ax, bars, [value], value_fontsize)
                if np.isfinite(value):
                    all_values.append(value)

    for separator_index in range(1, len(methods)):
        left_center = group_centers[separator_index - 1]
        right_center = group_centers[separator_index]
        separator_x = (left_center + right_center) / 2
        ax.axvline(separator_x, color="#cccccc", linestyle="--", linewidth=1.0, alpha=0.8)

    ymax = max(all_values) * 1.1 if all_values else 1.0
    ax.set_ylim(0, ymax)
    ax.set_xticks(group_centers, method_labels)
    ax.set_xlabel("Diagonalization Method", fontsize=axis_label_fontsize, fontweight="bold")
    ax.set_ylabel("Time [sec]", fontsize=axis_label_fontsize, fontweight="bold")
    ax.set_title("H8C10 CPU/GPU timing comparison", fontsize=title_fontsize, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    metric_handles = [Patch(facecolor=color, edgecolor="black", label=label) for _, label, color in metric_specs]
    backend_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=backend_hatches[backend], label=backend_labels[backend])
        for backend in ("cpu", "gpu")
    ]
    legend_metrics = ax.legend(
        handles=metric_handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.99),
        ncols=1,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
        prop={"size": legend_fontsize, "weight": "bold"},
    )
    ax.add_artist(legend_metrics)
    ax.legend(
        handles=backend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.99),
        ncols=1,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
        prop={"size": legend_fontsize, "weight": "bold"},
    )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(results: dict[int, dict[str, dict[str, float]]]) -> None:
    header = (
        "method".ljust(10)
        + "backend".ljust(8)
        + "first_diag".rjust(14)
        + "avg_post_first".rjust(16)
        + "avg_hartree".rjust(14)
    )
    print(header)
    print("-" * len(header))
    for method in sorted(results):
        for backend in ("cpu", "gpu"):
            values = results[method][backend]
            print(
                f"diagmeth{method}".ljust(10)
                + backend.ljust(8)
                + f"{values['first_diag_time']:.3f}".rjust(14)
                + f"{values['avg_post_first_diag_time']:.3f}".rjust(16)
                + f"{values['avg_hartree_time']:.3f}".rjust(14)
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot H8C10 CPU/GPU timing comparisons.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("samples/h8c10"),
        help="Root folder containing diagmeth*/cpu|gpu/*.out files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("samples/h8c10/h8c10_timing_comparison.png"),
        help="Where to save the figure.",
    )
    parser.add_argument(
        "--bar-width",
        type=float,
        default=0.24,
        help="Width of each CPU/GPU bar within a timing pair.",
    )
    parser.add_argument(
        "--group-gap",
        type=float,
        default=3.0,
        help="Horizontal spacing between neighboring diagmeth groups.",
    )
    parser.add_argument(
        "--metric-spacing",
        type=float,
        default=0.78,
        help="Horizontal spacing between timing categories inside one diagmeth group.",
    )
    parser.add_argument(
        "--title-fontsize",
        type=float,
        default=18,
        help="Font size for the plot title.",
    )
    parser.add_argument(
        "--axis-label-fontsize",
        type=float,
        default=15,
        help="Font size for the x/y axis labels.",
    )
    parser.add_argument(
        "--tick-fontsize",
        type=float,
        default=14,
        help="Font size for the x/y tick labels.",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=13,
        help="Font size for legend text.",
    )
    parser.add_argument(
        "--value-fontsize",
        type=float,
        default=10,
        help="Font size for the numeric labels above the bars.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    results = collect_metrics(args.root.resolve())
    print_summary(results)
    plot_metrics(
        results,
        args.output.resolve(),
        bar_width=args.bar_width,
        group_gap=args.group_gap,
        metric_spacing=args.metric_spacing,
        title_fontsize=args.title_fontsize,
        axis_label_fontsize=args.axis_label_fontsize,
        tick_fontsize=args.tick_fontsize,
        legend_fontsize=args.legend_fontsize,
        value_fontsize=args.value_fontsize,
    )
    print(f"\nSaved figure to {args.output.resolve()}")


if __name__ == "__main__":
    main()
