"""SCF runs of the meta-GGAs on the molecules of the QE + libxc references, and the cost of the orbital term.

Writes a ``parsec.in`` of one molecule of the meta-GGA references
(geometry, FHI98PP pseudopotentials of ``src/pp_generation/NCPPs/PBE``,
sphere radius, states, as in ``examples/0d_meta_gga``) with the given
functional and grid, runs it on the accelerated path (or the reference solver)
and prints the iterations and a JSON summary: energy, HOMO/LUMO (eV),
iterations, wall times, the meta-GGA's statistics and, with
``--term-timing``, the time of one application of the device Hamiltonians to
a block of vectors with and without the orbital term ``-div(vtau grad)``.

Usage: ``python -m parsec_python.acceleration.benchmarks.meta_gga_check --molecule H2O --functional scan
--spacing 0.16 [--backend cupy|reference] [--symmetry auto|off] [--out DIR]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

RYDBERG_EV = 13.605693122994

#: element -> (Potential_Num, Local_Component, Core_Cutoff_Radius, Electron_Per_Orbital)
ELEMENTS = {
    "H": (1, "s", 1.3, "1"),
    "C": (3, "d", 1.6, "2 2 0"),
    "N": (3, "d", 1.6, "2 3 0"),
    "O": (3, "d", 1.6, "2 4 0"),
}
#: molecule -> (atoms in bohr, sphere radius of the references, states, spin channels (up, down) or None)
MOLECULES = {
    "H2O": (
        [("O", 0.0, 0.0, -0.553578522), ("H", 1.4304288085, 0.0, 0.553578522), ("H", -1.4304288085, 0.0, 0.553578522)],
        11.53,
        8,
        None,
    ),
    "CH4": (
        [("C", 0.0, 0.0, 0.0)]
        + [("H", 1.1859538349 * a, 1.1859538349 * b, 1.1859538349 * a * b) for a, b in ((1, 1), (-1, -1), (-1, 1), (1, -1))],
        11.5,
        8,
        None,
    ),
    "N2": ([("N", 0.0, 0.0, 1.0371761835), ("N", 0.0, 0.0, -1.0371761835)], 11.0, 10, None),
    "CO": ([("C", 0.0, 0.0, -1.0660889932), ("O", 0.0, 0.0, 1.0660889932)], 11.0, 10, None),
    "NH3": (
        [
            ("N", -0.442918532, 0.0, 0.3600192354),
            ("H", 1.328755596, 0.0, -0.3600192354),
            ("H", -1.328755596, 1.534314802, -0.3600192354),
            ("H", -1.328755596, -1.534314802, -0.3600192354),
        ],
        11.5,
        8,
        None,
    ),
    "O2": ([("O", 0.0, 0.0, 1.1409221477), ("O", 0.0, 0.0, -1.1409221477)], 11.14, 11, (7, 5)),
}


def functional_lines(functional: str) -> list[str]:
    """The input lines of a functional: ``Correlation_Type``, or for ``libxc:a+b[;fhc=on]`` ``LibXC_Functional``."""

    if not functional.lower().startswith("libxc:"):
        return [f"Correlation_Type: {functional}"]
    names, *options = functional[len("libxc:"):].split(";")
    lines = ["Correlation_Type: pbe", "LibXC_Functional: " + " ".join(names.split("+"))]
    for option in options:
        key, _, value = option.partition("=")
        if key.strip().lower() == "fhc":
            lines.append(f"LibXC_Enforce_FHC: {'.true.' if value.strip().lower() == 'on' else '.false.'}")
        elif key.strip().lower() == "threshold":
            lines.append(f"LibXC_Density_Threshold: {value.strip()}")
        else:
            raise ValueError(f"unknown libxc option {option!r}")
    return lines


def write_input(directory: Path, molecule: str, functional: str, spacing: float, radius: float | None,
                criterion: float, max_iterations: int, extra: list[str], diag_tolerance: float = 1e-6) -> Path:
    atoms, reference_radius, states, spin = MOLECULES[molecule]
    radius = reference_radius if radius is None else radius
    # The FHI98PP files of the references: examples/0d_meta_gga (copies of the library's NCPPs/PBE/*/FHI),
    # else the library itself.
    root = Path(__file__).resolve().parents[4]
    example = root / "examples" / "0d_meta_gga"
    library = root / "src" / "pp_generation" / "NCPPs" / "PBE"
    species = sorted({atom[0] for atom in atoms}, key=lambda symbol: [a[0] for a in atoms].index(symbol))
    lines = [
        f"Boundary_Sphere_Radius: {radius}",
        f"Grid_Spacing: {spacing}",
        "Coordinate_Unit: Cartesian_Bohr",
        "Expansion_Order: 12",
        *functional_lines(functional),
        f"States_Num: {states}",
        f"Convergence_Criterion: {criterion} Ry",
        f"Diag_Tolerance: {diag_tolerance}",
        f"Max_Iter: {max_iterations}",
        f"Atom_Types_Num: {len(species)}",
    ]
    if spin is not None:
        up, down = spin
        lines += ["Spin_Polarization: true", "Fermi_Temp: -1 K"]
        # occup.in of the reference (the triplet of O2): filled states, then no partial ones, per channel.
        (directory / "occup.in").write_text(
            f"spin up: {up} electrons\n{up}\n0\nspin down: {down} electrons\n{down}\n0\n"
        )
    for symbol in species:
        number, local, core, electrons = ELEMENTS[symbol]
        source = example / f"{symbol}_FHIPP.DAT"
        if not source.exists():
            source = library / symbol / "FHI" / f"{symbol}_FHIPP.DAT"
        shutil.copy(source, directory / f"{symbol}_FHIPP.DAT")
        lines += [
            "",
            f"Atom_Type: {symbol}",
            "Pseudopotential_Format: fhipp",
            f"Potential_Num: {number}",
            f"Local_Component: {local}",
            f"Core_Cutoff_Radius: {core}",
            "begin Electron_Per_Orbital",
            electrons,
            "end Electron_Per_Orbital",
            "Cubic_Spline: .true.",
        ]
        if spin is not None:
            lines.append("Initial_Spin_Polarization: 0.3")
        lines.append("begin Atom_Coord")
        lines += [f"  {x:.10f} {y:.10f} {z:.10f}" for name, x, y, z in atoms if name == symbol]
        lines.append("end Atom_Coord")
    lines += extra
    path = directory / "parsec.in"
    path.write_text("\n".join(lines) + "\n")
    return path


def _levels(result):
    occupations = np.asarray(result.occupations)
    eigenvalues = np.asarray(result.eigenvalues)
    filled = occupations > 1e-6
    homo = float(np.max(eigenvalues[filled])) * RYDBERG_EV
    empty = eigenvalues[~filled]
    lumo = float(np.min(empty)) * RYDBERG_EV if empty.size else float("nan")
    return homo, lumo


def term_timing(system, repeats: int = 5) -> dict:
    """Seconds of one application of the device operators to their saved subspaces, with and without the term."""

    import cupy as cp

    device = getattr(system, "meta_gga_device", None)
    if device is None or not device.terms:
        return {}
    timings = {}
    for representation, operator in device.operators().items():
        term = device.terms[representation]
        if not term.bound:
            continue
        rows = operator.shape[0]
        with cp.cuda.Device(term.device_id):
            block = cp.random.default_rng(1).standard_normal((rows, 16))
            seconds = {}
            for label, bound in (("with_term", term), ("without_term", None)):
                operator.bind_kinetic_term(bound)
                operator.apply(block)
                cp.cuda.Device().synchronize()
                start = cp.cuda.Event()
                stop = cp.cuda.Event()
                start.record()
                for _ in range(repeats):
                    operator.apply(block)
                stop.record()
                stop.synchronize()
                seconds[label] = cp.cuda.get_elapsed_time(start, stop) / 1000.0 / repeats
            operator.bind_kinetic_term(term)
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
            start.record()
            for _ in range(repeats):
                term.apply(block)
            stop.record()
            stop.synchronize()
            seconds["term_alone"] = cp.cuda.get_elapsed_time(start, stop) / 1000.0 / repeats
            if any(flux is not None for flux in term.doubled_fluxes):
                # The orbital form: the term without its first-order part of W.
                kept = list(term.doubled_fluxes)
                term.doubled_fluxes = [None] * len(kept)
                start = cp.cuda.Event()
                stop = cp.cuda.Event()
                start.record()
                for _ in range(repeats):
                    term.apply(block)
                stop.record()
                stop.synchronize()
                seconds["term_without_w"] = cp.cuda.get_elapsed_time(start, stop) / 1000.0 / repeats
                term.doubled_fluxes = kept
        timings[str(representation)] = dict(rows=int(rows), vectors=16, **seconds)
    return timings


def _trace_meta_gga() -> None:
    """Print the range of ``vtau_out`` (and where it peaks) at every mixing step (diagnostics)."""

    from parsec_python.SCF import meta_gga as loop_module

    original = loop_module.MetaGGALoop.mix

    def mix(self, mixer, inputs, outputs, kinetic, **keywords):
        for channel, (field, potential) in enumerate(zip(kinetic, outputs)):
            values = np.asarray(field.values if hasattr(field, "values") else field)
            local = np.asarray(potential.values if hasattr(potential, "values") else potential)
            densities = keywords.get("densities")
            density = None if densities is None else densities[channel]
            density = None if density is None else np.asarray(density.values if hasattr(density, "values") else density)
            peak = int(np.argmax(np.abs(values)))
            where = "" if density is None else f" at n {density[peak]:.2e}"
            given = inputs[channel]
            given = np.asarray(given.values if hasattr(given, "values") else given)
            residual = ""
            if density is not None and given.shape == local.shape:
                # Where the weighted residual sits: shares of sum n (V_out - V_in)^2 by density decade.
                weighted = np.maximum(density, 0.0) * (local - given) ** 2
                total = float(np.sum(weighted)) or 1.0
                shares = [
                    float(np.sum(weighted[(density >= low) & (density < high)])) / total
                    for low, high in ((1e-1, np.inf), (1e-2, 1e-1), (1e-3, 1e-2), (1e-5, 1e-3), (0.0, 1e-5))
                ]
                worst = int(np.argmax(weighted))
                residual = (
                    f"  residual shares n>1e-1,1e-2,1e-3,1e-5,below {' '.join(f'{share:.2f}' for share in shares)}"
                    f" worst at n {density[worst]:.2e} dV {local[worst] - given[worst]:.2e}"
                )
            print(
                f"   trace ch{channel} vtau_out [{values.min():.4g}, {values.max():.4g}]{where}  "
                f"V_out [{local.min():.4g}, {local.max():.4g}]{residual}",
                flush=True,
            )
        return original(self, mixer, inputs, outputs, kinetic, **keywords)

    loop_module.MetaGGALoop.mix = mix


def _long_lanczos_top(operator, steps: int = 60) -> float:
    """The largest Ritz value of a fully reorthogonalized Lanczos run of ``steps`` (diagnostics)."""

    import cupy as cp

    size = int(operator.shape[0])
    steps = min(steps, size)
    basis = cp.empty((size, steps), dtype=cp.float64)
    vector = cp.asarray(np.random.default_rng(5).standard_normal(size))
    vector /= cp.linalg.norm(vector)
    alphas, betas = [], []
    for step in range(steps):
        basis[:, step] = vector
        product = operator @ vector
        alpha = float(cp.dot(vector, product))
        product = product - basis[:, : step + 1] @ (basis[:, : step + 1].T @ product)
        product = product - basis[:, : step + 1] @ (basis[:, : step + 1].T @ product)
        beta = float(cp.linalg.norm(product))
        alphas.append(alpha)
        if beta < 1e-12 or step == steps - 1:
            break
        betas.append(beta)
        vector = product / beta
    count = len(alphas)
    matrix = np.diag(alphas) + np.diag(betas[: count - 1], 1) + np.diag(betas[: count - 1], -1)
    return float(np.linalg.eigvalsh(matrix)[-1])


def _trace_bounds() -> None:
    """Print the filter bounds of every device subspace pass and a long Lanczos estimate of the top (diagnostics)."""

    from parsec_python.acceleration.Eigensolvers import chebyshev as module

    for name in ("_exchange_block_filter",):
        original = getattr(module, name)

        def traced(operator, matrix, blocks, lower_bound, upper_bound, *args, _original=original, **keywords):
            top = _long_lanczos_top(operator)
            print(
                f"   bounds lower {float(lower_bound):.6g} upper {float(upper_bound):.6g} "
                f"long-Lanczos top {top:.6g} rows {operator.shape[0]}",
                flush=True,
            )
            return _original(operator, matrix, blocks, lower_bound, upper_bound, *args, **keywords)

        setattr(module, name, traced)


_INTERVALS: list = []


def _interval_statistics() -> None:
    """Record the filter interval of every device pass of an operator with an orbital term (the summary's ``filter``)."""

    from parsec_python.acceleration.Eigensolvers import chebyshev as module

    original = module._exchange_block_filter

    def recorded(operator, matrix, blocks, lower_bound, upper_bound, *args, **keywords):
        margin = float(getattr(operator, "spectral_margin", 0.0) or 0.0)
        degree = max((block.degree for block in blocks), default=0)
        _INTERVALS.append((float(lower_bound), float(upper_bound), margin, int(degree), int(matrix.shape[1])))
        return original(operator, matrix, blocks, lower_bound, upper_bound, *args, **keywords)

    module._exchange_block_filter = recorded


def _interval_summary() -> dict:
    if not _INTERVALS:
        return {}
    values = np.asarray(_INTERVALS, dtype=np.float64)
    return dict(
        passes=int(values.shape[0]),
        mean_upper=float(values[:, 1].mean()),
        max_upper=float(values[:, 1].max()),
        mean_span=float((values[:, 1] - values[:, 0]).mean()),
        mean_margin=float(values[:, 2].mean()),
        max_margin=float(values[:, 2].max()),
        mean_degree=float(values[:, 3].mean()),
        columns=int(values[:, 4].sum()),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--molecule", default="H2O", choices=sorted(MOLECULES))
    parser.add_argument("--functional", default="scan")
    parser.add_argument("--spacing", type=float, default=0.16)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--criterion", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--diag-tolerance", type=float, default=1e-6)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--symmetry", default="auto")
    parser.add_argument("--out", default=None)
    parser.add_argument("--term-timing", action="store_true")
    parser.add_argument("--trace", action="store_true", help="print the range of vtau_out and V_out of every iteration")
    parser.add_argument("--extra", action="append", default=[], help="an input line added to parsec.in")
    parser.add_argument(
        "--bound-trace", action="store_true",
        help="print the filter bounds of every device pass of an operator with an orbital term and a long Lanczos top",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="an existing parsec.in (its folder's .DAT files with it) instead of a molecule: Correlation_Type, "
        "Max_Iter and Convergence_Criterion are replaced",
    )
    arguments = parser.parse_args(argv)

    from parsec_python.Input import parse_parsec_input

    directory = Path(tempfile.mkdtemp(prefix="meta_gga_check_"))
    if arguments.input is None:
        path = write_input(
            directory, arguments.molecule, arguments.functional, arguments.spacing, arguments.radius,
            arguments.criterion, arguments.max_iterations, arguments.extra, arguments.diag_tolerance,
        )
    else:
        import re

        source = Path(arguments.input)
        # The pseudopotentials beside the input or up to three folders above it.
        for folder in (source.parent, *list(source.parent.parents)[:3]):
            for item in folder.glob("*.DAT"):
                if not (directory / item.name).exists():
                    shutil.copy(item, directory / item.name)
        text = source.read_text()
        text = re.sub(r"(?im)^(Correlation_Type|LibXC_Functional)\s*[: ].*$\n?", "", text)
        text += "".join(f"{line}\n" for line in functional_lines(arguments.functional))
        for key, value in (
            ("Max_Iter", str(arguments.max_iterations)),
            ("Convergence_Criterion", f"{arguments.criterion} Ry"),
        ):
            text, count = re.subn(rf"(?im)^{key}\s*[: ].*$", f"{key}: {value}", text)
            if not count:
                text += f"{key}: {value}\n"
        text += "".join(f"{line}\n" for line in arguments.extra)
        path = directory / "parsec.in"
        path.write_text(text)
        arguments.molecule = source.parent.name
    problem = parse_parsec_input(path).problem
    if arguments.trace:
        _trace_meta_gga()
    if arguments.bound_trace:
        _trace_bounds()
    _interval_statistics()
    started = time.perf_counter()

    def report(record):
        if not hasattr(record, "iteration"):
            print(f"record {type(record).__name__}", flush=True)
            return
        print(
            f"iter {record.iteration:3d} sre {record.weighted_residual:.3e} E {record.energies.total:.10f} "
            f"vtau.tau {getattr(record.energies, 'integral_vtau_tau', 0.0):.6f} t {record.total_seconds:.2f}",
            flush=True,
        )

    if arguments.backend == "reference":
        from parsec_python.SCF import prepare_single_point, run_scf

        system = prepare_single_point(problem)
        prepared = time.perf_counter()
        result = run_scf(system, callback=report)
        details = ()
    else:
        from parsec_python.acceleration.driver import prepare_single_point, run_scf

        system = prepare_single_point(problem, backend=arguments.backend, symmetry=arguments.symmetry)
        prepared = time.perf_counter()
        accelerated = run_scf(system, callback=report)
        result = accelerated.result
        details = system.backend_info.details
    finished = time.perf_counter()
    homo, lumo = _levels(result)
    record = getattr(result, "meta_gga", None)
    device = getattr(system, "meta_gga_device", None)
    summary = dict(
        molecule=arguments.molecule,
        functional=arguments.functional,
        spacing=arguments.spacing,
        grid_points=int(system.grid.size),
        backend=arguments.backend,
        symmetry=arguments.symmetry,
        converged=bool(result.converged),
        iterations=int(result.iterations),
        total_energy=float(result.energies.total),
        homo_ev=homo,
        lumo_ev=lumo,
        integral_vtau_tau=float(getattr(result.energies, "integral_vtau_tau", 0.0)),
        fermi_hole_clamped=None if record is None else int(record.fermi_hole_clamped),
        kinetic_residual=None if record is None else float(record.kinetic_residual),
        gradient_form=None if record is None else getattr(record, "gradient", None),
        preparation_seconds=prepared - started,
        scf_seconds=finished - prepared,
        seconds_per_iteration=(finished - prepared) / max(1, int(result.iterations)),
        diagonalization_seconds=float(result.timings.diagonalization_seconds),
        xc_seconds=float(result.timings.xc_seconds),
        hartree_seconds=float(result.timings.hartree_seconds),
        mixing_energy_seconds=float(result.timings.mixing_energy_seconds),
        binding_seconds=float(result.timings.hamiltonian_binding_seconds),
        density_seconds=float(result.timings.occupations_density_seconds),
        meta_gga_statistics=None if device is None else dict(device.statistics),
        filter=_interval_summary(),
        environment={
            key: value for key, value in __import__("os").environ.items() if key.startswith("PARSEC_META_GGA")
        },
        meta_gga_device_bytes=None if device is None else {str(k): int(v) for k, v in device.device_bytes.items()},
        details=[item for item in details if "meta" in item[0] or item[0].startswith("orbital_sector_dim")],
    )
    if arguments.term_timing and arguments.backend != "reference":
        summary["term_timing"] = term_timing(system)
    text = json.dumps(summary, indent=1, default=str)
    print(text)
    if arguments.out:
        Path(arguments.out).mkdir(parents=True, exist_ok=True)
        (Path(arguments.out) / f"{arguments.molecule}_{arguments.functional}_{arguments.symmetry}.json").write_text(text)
    shutil.rmtree(directory, ignore_errors=True)
    return 0 if result.converged else 3


if __name__ == "__main__":
    raise SystemExit(main())
