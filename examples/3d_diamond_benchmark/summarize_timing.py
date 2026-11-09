"""Summarize the timing.json of full-SCF runner runs: energy, iterations, times and their split, memory per rank.

Usage: python summarize_timing.py <run dir or timing.json> ... [--json OUT.json]

Prints, for every run: total energy and iterations; SCF seconds per iteration; the first eigensolve; the split of
the SCF into diagonalization (with, for a basis the ranks share, the stages of the Ritz steps of the root:
filter, H applied, the exchanges to rows and back, Gram products, their sum over the ranks, the small solve on the
root, the rotation), density, Hartree and XC; the preparation of every rank; the GPU memory peak of every rank
(nvidia-smi samples) and its host high water. With --json the same numbers go to a file.
"""

import argparse
import json
from pathlib import Path


def summary(path: Path) -> dict:
    if path.is_dir():
        path = path / "timing.json"
    data = json.loads(path.read_text())
    result = data["result"]
    ranks = data["per_rank"]
    timings = result["scf_timings"]
    iterations = int(result["iterations"])
    record = dict(
        path=str(path), ranks=data["ranks"], nodes=data["nodes"], gpus=data["gpu_count"],
        converged=result["converged"], iterations=iterations, energy_ry=result["energies_ry"]["total"],
        scf_seconds=result["scf_seconds"], first_diagonalization_seconds=result["first_iteration_diagonalization_seconds"],
        pre_scf_seconds=result["pre_scf_setup_reporting_seconds"], reported_seconds=result["reported_program_seconds"],
        preparation_seconds=[round(row["preparation_seconds"], 2) for row in ranks],
        scf_split_seconds={key: round(value, 3) for key, value in timings.items()
                           if key.endswith("seconds") and isinstance(value, (int, float)) and value},
        gpu_peak_mib=[max((value or 0) for value in (row["gpu_memory"].get("peak_used_mib_by_selected_device")
                                                    or {}).values()) if row["gpu_memory"].get("available") else None
                      for row in ranks],
        host_high_water_gib=[round(max((item.get("process_rss_high_water_bytes") or 0)
                                       for item in row["memory_observations"] if item) / 2**30, 2) for row in ranks],
    )
    if any("device_live_peak" in row for row in ranks):
        # PARSEC_DEVICE_LIVE_PEAK=1: the most bytes the arrays of a rank held at once (GiB).
        record["live_peak_gib"] = [round(max((item["live_peak_bytes"] for item in row.get("device_live_peak", {}).values()),
                                             default=0) / 2**30, 2) for row in ranks]
    later = [item for item in result["iteration_history"][1:]]
    if later:
        record["later_iteration_seconds"] = round(sum(item["total_seconds"] for item in later) / len(later), 3)
        record["later_iteration_split_seconds"] = {
            key: round(sum(item[key] for item in later) / len(later), 4)
            for key in ("diagonalization_seconds", "occupations_density_seconds", "hartree_seconds", "xc_seconds",
                        "mixing_energy_seconds")
            if key in later[0]}
    shared = [state for state in ranks[0]["distributed_state"] if "ranks" in state]
    if shared:
        state = shared[0]
        passes = max(int(state["ritz_passes"]), 1)
        record["shared_root_ritz_passes"] = int(state["ritz_passes"])
        if "ritz_route" in state:
            record["shared_ritz_route"] = state["ritz_route"]
        record["shared_root_seconds"] = {key: round(value, 3) for key, value in state["seconds"].items()}
        record["shared_root_seconds_per_pass"] = {key: round(value / passes, 4) for key, value in state["seconds"].items()}
        record["shared_exchanged_gb_per_rank"] = round(state["exchanged_bytes"] / 1e9, 2)
    stages = ranks[0].get("preparation_stages") or []
    record["root_stages_seconds"] = {item["stage"]: round(item["end_seconds"] - item["start_seconds"], 2)
                                     for item in stages if item["end_seconds"] - item["start_seconds"] > 0.2}
    if len(ranks) > 1:
        stages = ranks[1].get("preparation_stages") or []
        record["rank1_stages_seconds"] = {item["stage"]: round(item["end_seconds"] - item["start_seconds"], 2)
                                          for item in stages if item["end_seconds"] - item["start_seconds"] > 0.2}
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    records = []
    for run in args.runs:
        try:
            records.append(summary(run))
        except (OSError, KeyError, ValueError) as error:
            print(f"{run}: {type(error).__name__}: {error}")
            continue
        for key, value in records[-1].items():
            print(f"  {key}: {value}")
        print()
    if args.json is not None:
        args.json.write_text(json.dumps(records, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
