"""Acceptance of ``PARSEC_CUPY_STENCIL_KERNEL=pipelined`` on the operators of real sectors, bit for bit.

    python -m parsec_python.acceleration.benchmarks.stencil_pipelined_check \\
        --input <folder>/parsec.in@16,0 --output check.json

For every input (with its pseudopotential files beside it) the stencil and the projectors of one symmetry sector
(``--sector``, 0 by default) are built by the host builders of a run, and for every tile size after ``@`` (16 by
default; 0 is the slot-major layout) operators are built with the switch at ``former`` and at ``pipelined``.  Then,
with the comparisons of ``tests/test_stencil_pipeline.py``, outputs prefilled with a NaN pattern and compared whole
as 64-bit words, NaN payloads included:

1. the recurrence kernel of the two operators on every launch path (widths 1 to 6, every flag, seven operand
   layouts, slot counts below the compiled one) for every kind of input: finite, wide exponents, signed zeros,
   subnormals and infinities, scattered NaNs, pairs of NaNs of different payloads that meet in one operation, and
   every value a NaN of its own payload;
2. the methods of the two operators (``chebyshev_recurrence`` and ``apply``) on blocks of 1 to 14 columns in four
   layouts, counting which kernel took which launch;
3. the production routes of ``CuPyHamiltonian``: ``H X`` (``apply_into``), the filter graphs
   (``BlockFilterGraphs``) on plans of 6, 14 (6 + 6 + 2) and 48 columns, and the filter without graphs
   (``chebff_filter``) on 14 columns, with finite vectors, one NaN, two NaNs of different payloads in neighbouring
   rows of the sixth column, and every value a NaN.

It reports the registers and local memory of the kernels and writes everything to ``--output``.  Exit status 0 when
every case passes; 2 when a word differs, a launch goes to the wrong kernel or a case raises; 1 when CuPy cannot be
used or an input cannot be prepared.  On a GPU other than the A100 the payload of a NaN where two NaNs meet may
differ (``README.md``); the counts "not as two NaNs" leave such words out.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def log(*items):
    print("PIPELINED", *items, flush=True)


def real_sector(path, sector):
    """The stencil metadata and the nonlocal operator of ``sector`` from the host builders."""
    from unittest.mock import patch

    from parsec_python.Input import parse_parsec_input
    from parsec_python.acceleration import driver
    from parsec_python.acceleration.Symmetry import (
        load_or_build_reflection_decomposition, load_or_detect_reflection_reduction)
    from parsec_python.acceleration.Symmetry.operator_cache import load_or_build_reduced_operators
    from parsec_python.acceleration.backends.selection import BackendSelection

    with patch.dict(os.environ, {"PARSEC_IONIC_BACKEND": "native", "PARSEC_ACCELERATED_RESIDENT": "0"}):
        translation = parse_parsec_input(Path(path))
        selection = BackendSelection(requested="auto", selected="cupy", finite_difference_builder="native",
                                     hartree_backend="native")
        reference = driver._prepare_reference_physics(
            translation.problem, selection, orbital_operators_only=True, defer_native_laplacian=True,
            deferred_laplacian_cache_directory=None)
        reduction, _ = load_or_detect_reflection_reduction(reference.grid, reference.atoms, cache_directory=None)
        decomposition, _ = load_or_build_reflection_decomposition(reference.grid, reduction, reduction_key=None,
                                                                  cache_directory=None)
        bundle = load_or_build_reduced_operators(decomposition, reference.negative_laplacian,
                                                 reference.nonlocal_operator, cache_directory=None,
                                                 representations=(sector,))
    return bundle.stencil_metadata[sector], bundle.nonlocal_operators[sector]


def hamiltonians(metadata, nonlocal_operator, tile, potential):
    """``CuPyHamiltonian`` of the sector with the switch at ``former`` and at ``pipelined``."""
    from parsec_python.acceleration.backends.cupy import CuPyHamiltonian
    from parsec_python.acceleration.tests.test_stencil_pipeline import _switch

    built = {}
    for value in ("former", "pipelined"):
        with _switch(value):
            built[value] = CuPyHamiltonian(None, potential, nonlocal_operator, retain_generic_laplacian=False,
                                           finite_difference_metadata=metadata, implicit_tile=tile)
    return built


def route_vectors(rows, width, kind, row):
    """Vectors of one kind for the routes; ``row`` has projector entries, so that two NaNs there meet in the scatter
    as well as in the stencil."""
    from parsec_python.acceleration.tests.test_stencil_pipeline import nan_pattern

    host = np.random.default_rng(5).uniform(-1.0, 1.0, (rows, width))
    words = host.view(np.uint64)
    if kind in ("one_nan", "two_nans"):
        words[row, 5] = np.uint64(0x7FF800000000BEEF)
    if kind == "two_nans":
        words[row + 1, 5] = np.uint64(0xFFF800000000F00D)
        words[row, 2] = np.uint64(0x7FF4000000000AB0)
        words[row + 1, 2] = np.uint64(0x7FF8000000000AB1)
    if kind == "all_nan":
        words[...] = nan_pattern(host.shape, 3)
    return host


def compare_routes(cp, built, bound, row, degree):
    """H X, the filter graphs and the filter without graphs of both operators; differing words per route."""
    from parsec_python.acceleration.Eigensolvers.chebyshev import FilterBlock, chebff_filter
    from parsec_python.acceleration.Eigensolvers.filter_graph import BlockFilterGraphs
    from parsec_python.acceleration.tests.test_stencil_pipeline import _switch

    rows = int(built["former"].shape[0])
    plans = {6: (FilterBlock(0, 6, degree),),
             14: (FilterBlock(0, 6, degree), FilterBlock(6, 12, degree), FilterBlock(12, 14, degree)),
             48: tuple(FilterBlock(start, start + 6, degree) for start in range(0, 48, 6))}
    found = {}
    for kind in ("finite", "one_nan", "two_nans", "all_nan"):
        host = route_vectors(rows, 48, kind, row)
        results = {}
        for value, operator in built.items():
            vectors = cp.asarray(host[:, :14], order="F")
            outputs = {"H X, 14 columns": operator.apply_into(vectors, cp.full_like(vectors, cp.nan))}
            for width, blocks in plans.items():
                outputs[f"filter graphs, {width} columns"] = BlockFilterGraphs(operator).apply(
                    cp.asarray(host[:, :width], order="F"), blocks, -bound, bound, -1.5 * bound, False)
            with _switch(value, PARSEC_CUPY_FILTER_GRAPHS="0"):
                outputs["filter without graphs, 14 columns"] = chebff_filter(
                    operator, cp.asarray(host[:, :14], order="F"), degree, -bound, bound, -1.5 * bound,
                    block_size=6, batch_blocks=False)
            cp.cuda.Device().synchronize()
            results[value] = {route: cp.asnumpy(array) for route, array in outputs.items()}
            # The graph workspaces and their buffers go before the next operator records its own.
            del outputs
            gc.collect()
        entry = {}
        for route, reference in results["former"].items():
            other = results["pipelined"][route]
            a, b = reference.view(np.uint64), other.view(np.uint64)
            both = np.isnan(reference) & np.isnan(other)
            entry[route] = dict(words=int(a.size), differing=int((a != b).sum()),
                                differing_not_as_two_nans=int(((a != b) & ~both).sum()), nan_words=int(both.sum()))
        found[kind] = entry
    return found


def run_input(cp, spec, sector, kinds, method_kinds, degree, report, save):
    from parsec_python.acceleration.tests.test_stencil_pipeline import (
        Operands, build_pair, compare_operator_methods, compare_recurrence_kernels, count_launches,
        takes_fast_path)

    path, _, tile_text = spec.partition("@")
    tiles = [int(item) for item in tile_text.split(",") if item.strip()] or [16]
    started = time.perf_counter()
    metadata, nonlocal_operator = real_sector(path, sector)
    slots, rows = int(metadata.neighbors.shape[0]), int(metadata.shape[0])
    name = "/".join(Path(path).resolve().parts[-3:-1])
    potential = np.random.default_rng(3).uniform(-0.5, 0.5, rows)
    log(f"input {name} sector {sector}: rows {rows}, slots {slots}, prepared in {time.perf_counter() - started:.0f} s")
    for tile in tiles:
        entry = dict(input=name, sector=sector, rows=rows, slots=slots, tile=tile)
        report["cases"].append(entry)
        try:
            began = time.perf_counter()
            built = hamiltonians(metadata, nonlocal_operator, tile, potential)
            reference = built["former"]
            projectors, projector_count = reference.projector_csr_data, int(reference.projector_count)
            offsets = cp.asnumpy(projectors[0])
            with_entries = np.flatnonzero(np.diff(offsets) > 0)
            row = int(with_entries[len(with_entries) // 2]) if with_entries.size else rows // 2
            row = min(row, rows - 2)
            stencils = built["former"].compact_finite_difference, built["pipelined"].compact_finite_difference
            entry["storage"] = stencils[0].storage_mode
            if tile:
                entry["regular_rows"] = int(stencils[0].implicit_statistics["regular_rows"])
            entry["registers"] = dict(former=stencils[0].stencil_kernel_attributes,
                                      pipelined=stencils[1].stencil_kernel_attributes)
            entry["same_apply_kernel"] = stencils[1].apply_kernel is stencils[0].apply_kernel
            entry["same_former_recurrence_kernel"] = (stencils[1].former_recurrence_kernel
                                                      is stencils[0].recurrence_kernel)
            # 1. the recurrence kernels on every launch path.
            former, pipelined = build_pair(cp, metadata, tile)
            entry["kernels"] = {}
            for index, kind in enumerate(kinds):
                operands = Operands(cp, rows, kind, 7919 * index + 31 * tile + slots, projectors, projector_count,
                                    columns=6)
                entry["kernels"][kind] = compare_recurrence_kernels(cp, former, pipelined, operands)
                del operands
                cp.get_default_memory_pool().free_all_blocks()
                found = entry["kernels"][kind]
                log(f"  {name} tile {tile:3d} kernel paths {kind:8s}: {found['launches']} launch pairs "
                    f"({found['fast_launches']} fast path), differing words {found['differing_words']} "
                    f"(not as two NaNs {found['differing_not_as_two_nans']}), NaN words compared {found['nan_words']}")
                for path_entry in found["differing"][:3]:
                    log("    DIFFERS", path_entry)
            # 2. the operators' methods, and which kernel took which launch.
            former_fast, former_rest = count_launches(former)
            fast, rest = count_launches(pipelined)
            entry["methods"] = {}
            for index, kind in enumerate(method_kinds):
                operands = Operands(cp, rows, kind, 104729 * index + tile + slots, projectors, projector_count)
                entry["methods"][kind] = compare_operator_methods(cp, former, pipelined, operands)
                del operands
                cp.get_default_memory_pool().free_all_blocks()
                found = entry["methods"][kind]
                log(f"  {name} tile {tile:3d} methods {kind:8s}: {found['calls']} call pairs, differing words "
                    f"{found['differing_words']} (not as two NaNs {found['differing_not_as_two_nans']}), "
                    f"NaN words compared {found['nan_words']}")
            entry["dispatch"] = dict(
                fast_path_launches=len(fast.launches), former_kernel_launches=len(rest.launches),
                fast_launches_take_the_fast_path=all(takes_fast_path(s, w, r, slots) for w, r, s in fast.launches),
                former_launches_do_not=not any(takes_fast_path(s, w, r, slots) for w, r, s in rest.launches),
                same_launches_as_former=sorted(fast.launches + rest.launches) == sorted(former_fast.launches),
                former_one_kernel=former_fast is former_rest)
            log(f"  {name} tile {tile:3d} dispatch: {entry['dispatch']}")
            del former, pipelined
            # 3. the production routes.
            bound = 1.01 * (slots * float(np.abs(metadata.coefficient_palette).max()) + 0.6 + 50.0)
            entry["routes"] = compare_routes(cp, built, bound, row, degree)
            for kind, routes in entry["routes"].items():
                for route, counts in routes.items():
                    log(f"  {name} tile {tile:3d} route {kind:8s} {route:36s} words {counts['words']:10d} NaN "
                        f"{counts['nan_words']:10d} differing {counts['differing']}")
            del built
            cp.get_default_memory_pool().free_all_blocks()
            entry["seconds"] = time.perf_counter() - began
        except Exception:
            entry["error"] = traceback.format_exc()
            log(f"  {name} tile {tile} ERROR {entry['error'].splitlines()[-1]}")
        save()


def differing(entry):
    words = sum(item["differing_words"] for item in entry.get("kernels", {}).values())
    words += sum(item["differing_words"] for item in entry.get("methods", {}).values())
    words += sum(counts["differing"] for routes in entry.get("routes", {}).values() for counts in routes.values())
    return words


def main():
    from parsec_python.acceleration.tests.test_stencil_pipeline import KINDS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, help="PATH[@TILES], e.g. parsec.in@16,0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sector", type=int, default=0)
    parser.add_argument("--kinds", default=",".join(KINDS), help="kinds of input of the kernel paths")
    parser.add_argument("--method-kinds", default="finite,twonan,allnan", help="kinds of input of the methods")
    parser.add_argument("--degree", type=int, default=7, help="steps of a filter block on the routes")
    args = parser.parse_args()
    report = dict(script="stencil_pipelined_check", started=time.strftime("%Y-%m-%d %H:%M:%S"),
                  arguments=vars(args), cases=[])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(report, indent=1, default=str) + "\n")

    status = 0
    try:
        import cupy as cp

        report.update(cupy=cp.__version__, device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
        try:
            report["nvrtc"] = list(cp.cuda.nvrtc.getVersion())
        except Exception as error:
            report["nvrtc"] = repr(error)
        log(f"CuPy {report['cupy']}, {report['device']}, NVRTC {report['nvrtc']}")
        kinds = tuple(item for item in args.kinds.split(",") if item)
        method_kinds = tuple(item for item in args.method_kinds.split(",") if item)
        for spec in args.input:
            run_input(cp, spec, args.sector, kinds, method_kinds, args.degree, report, save)
    except Exception:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
        status = 1
    bad = [entry for entry in report["cases"] if "error" in entry or differing(entry)
           or not all(entry.get("dispatch", {}).values())]
    report.update(failing_cases=len(bad), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    save()
    log(f"SUMMARY cases {len(report['cases'])}, differing words {sum(differing(entry) for entry in report['cases'])}, "
        f"failing cases {len(bad)}, status {status}")
    return status or (2 if bad else 0)


if __name__ == "__main__":
    raise SystemExit(main())
