"""Accuracy of a semilocal or meta-GGA SCF: the run, then the derivatives of its orbitals by other operators.

Runs the SCF of one molecule of the meta-GGA references (inputs of
:mod:`parsec_python.acceleration.benchmarks.meta_gga_check`) and evaluates the
exchange-correlation functional once more on the final orbitals with other
first-derivative operators in ``sigma = |grad n|^2`` and
``tau = 1/2 sum f |grad psi|^2``:

* ``code``: the operator of the run (:func:`parsec_python.V_xc.meta_gga.grid_gradient`);
* ``fd<2M>``: centered stencils of order ``2M`` on the same grid, zero outside
  the domain (``fd12`` with ``Expansion_Order: 12`` is the operator of the run);
* ``fft``: spectral derivatives on the bounding box of the domain (padded);
* ``fft2x`` (``--refine 2``): the orbitals interpolated spectrally to a twice finer box and the
  functional summed there (the quadrature error of the energy density on the grid of the run).

Each operator is used with ``grad n`` of ``sigma`` from the density (``density``: the operator on
``n``) and from the orbitals (``orbital``: ``sum d f 2 psi D psi``).  For each: ``E_xc`` (Ry) and its parts by density shells, the points with
``tau < tau_W``, and the exchange-correlation part of the HOMO and LUMO,
``<psi|v|psi> + int vtau |grad psi|^2`` (Ry; ``v = dE/dn`` written as
``vrho psi^2 + 2 vsigma grad n . grad(psi^2)``).  The differences to ``code``
are first-order estimates of what the total energy and the levels would do if
the functional were evaluated with exact derivatives on the same orbitals
(the orbitals are stationary for the code's energy, so the energy change is
second order in the orbital change).

Usage: ``python -m parsec_python.acceleration.benchmarks.meta_gga_accuracy --molecule H2O --functional scan
--spacing 0.16 [--libxc "mgga_x_r2scan mgga_c_r2scan"] [--orders 12,20,32] [--refine 2] [--out DIR] [--label NAME]``,
or ``--input path/parsec.in`` for an input of one's own (its functional and grid): the ``fft2x`` rows then
estimate the quadrature error of its meta-GGA energy and levels.
"""

from __future__ import annotations

import argparse
import json
from math import factorial
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .meta_gga_check import MOLECULES, RYDBERG_EV, write_input

#: The distorted water molecule of the force references (refs/mgga: one O-H bond 1.10 A).
MOLECULES.setdefault(
    "H2O_dist",
    (
        [("O", -0.1066993491, 0.0, -0.6361642021), ("H", 1.5371281575, 0.0, 0.6361642021),
         ("H", -1.5371281575, 0.0, 0.4709928420)],
        11.66,
        8,
        None,
    ),
)

#: Density shells (e/bohr^3, upper edges first) of the energy decomposition.
SHELL_EDGES = (np.inf, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 0.0)
def fd_coefficients(order: int) -> np.ndarray:
    """Centered first-derivative weights ``d_j``, ``j = -M..M``, of order ``2M`` (as ``Laplacian``)."""

    width = order // 2
    weights = np.zeros(2 * width + 1)
    for shell in range(1, width + 1):
        value = (-1.0) ** (shell + 1) * factorial(width) ** 2 / (shell * factorial(width - shell) * factorial(width + shell))
        weights[width + shell] = value
        weights[width - shell] = -value
    return weights


class BoxDerivatives:
    """Derivatives of grid fields on the (padded) bounding box of a cluster domain."""

    def __init__(self, grid, method: str, pad: int = 24) -> None:
        self.grid = grid
        self.method = method
        self.h = float(grid.spacing)
        self.pad = int(pad)
        shape = np.asarray(grid.lookup.shape) + 2 * self.pad
        if method == "fft":
            shape = np.array([_fft_size(int(value)) for value in shape])
        self.shape = tuple(int(value) for value in shape)
        self.local = (grid.integer_coordinates - grid.index_min + self.pad).T
        if method == "fft":
            # rfftn: full frequencies on the first two axes, half on the last; the Nyquist
            # mode of an even size gets no derivative (it has no odd part).
            self.wavenumbers = []
            for axis, size in enumerate(self.shape):
                if axis < 2:
                    k = 2.0 * np.pi * np.fft.fftfreq(size, d=self.h)
                else:
                    k = 2.0 * np.pi * np.fft.rfftfreq(size, d=self.h)
                if size % 2 == 0:
                    k[size // 2] = 0.0
                self.wavenumbers.append(k)
        else:
            self.coefficients = fd_coefficients(int(method[2:])) / self.h

    def embed(self, values) -> np.ndarray:
        box = np.zeros(self.shape)
        box[self.local[0], self.local[1], self.local[2]] = values
        return box

    def extract(self, box) -> np.ndarray:
        return box[self.local[0], self.local[1], self.local[2]]

    def gradient(self, values) -> np.ndarray:
        """``(3, n)``: the gradient of a grid field at the points of the domain."""

        box = self.embed(values)
        if self.method == "fft":
            spectrum = np.fft.rfftn(box)
            result = []
            for axis in range(3):
                k = self.wavenumbers[axis]
                shape = [1, 1, 1]
                shape[axis] = k.size
                derivative = np.fft.irfftn(spectrum * (1j * k.reshape(shape)), s=self.shape, axes=(0, 1, 2))
                result.append(self.extract(derivative))
            return np.stack(result)
        width = (self.coefficients.size - 1) // 2
        result = []
        for axis in range(3):
            derivative = np.zeros(self.shape)
            for shell in range(1, width + 1):
                weight = self.coefficients[width + shell]
                derivative += weight * (np.roll(box, -shell, axis=axis) - np.roll(box, shell, axis=axis))
            result.append(self.extract(derivative))
        return np.stack(result)


def _fft_size(size: int) -> int:
    """The smallest 2^a 3^b 5^c at least ``size``."""

    best = None
    for a in range(0, 14):
        for b in range(0, 9):
            for c in range(0, 6):
                value = 2**a * 3**b * 5**c
                if value >= size and (best is None or value < best):
                    best = value
    return int(best)


class RefinedSpectral:
    """Spectral derivatives on a ``factor`` times finer box: the trigonometric interpolation of the grid fields.

    :meth:`interpolate` takes a field on the points of the domain to the
    points of the fine box (flattened), :meth:`gradient` differentiates such
    a fine field.  ``E_xc`` summed over the fine box measures the quadrature
    error of the energy density on the grid of the run (a plane-wave code
    evaluates the functional on its dense grid, twice the resolution of the
    orbitals).
    """

    def __init__(self, grid, factor: int = 2, pad: int = 8) -> None:
        self.coarse = BoxDerivatives(grid, "fft", pad=pad)
        self.factor = int(factor)
        self.method = f"fft{self.factor}x"
        self.shape = tuple(self.factor * size for size in self.coarse.shape)
        self.h = float(grid.spacing) / self.factor
        self.wavenumbers = []
        for axis, size in enumerate(self.shape):
            if axis < 2:
                k = 2.0 * np.pi * np.fft.fftfreq(size, d=self.h)
            else:
                k = 2.0 * np.pi * np.fft.rfftfreq(size, d=self.h)
            if size % 2 == 0:
                k[size // 2] = 0.0
            self.wavenumbers.append(k)

    @property
    def volume(self) -> float:
        return self.h**3

    def interpolate(self, values) -> np.ndarray:
        """The trigonometric interpolation of a domain field on the fine box (flattened); Nyquist modes dropped."""

        spectrum = np.fft.rfftn(self.coarse.embed(values))
        coarse = self.coarse.shape
        fine = np.zeros((self.shape[0], self.shape[1], self.shape[2] // 2 + 1), dtype=np.complex128)
        halves = [(size - 1) // 2 for size in coarse]
        rows = list(range(0, halves[0] + 1)) + list(range(coarse[0] - halves[0], coarse[0]))
        rows_fine = list(range(0, halves[0] + 1)) + list(range(self.shape[0] - halves[0], self.shape[0]))
        columns = list(range(0, halves[1] + 1)) + list(range(coarse[1] - halves[1], coarse[1]))
        columns_fine = list(range(0, halves[1] + 1)) + list(range(self.shape[1] - halves[1], self.shape[1]))
        depth = halves[2] + 1
        fine[np.ix_(rows_fine, columns_fine, range(depth))] = spectrum[np.ix_(rows, columns, range(depth))]
        values = np.fft.irfftn(fine, s=self.shape, axes=(0, 1, 2)) * float(self.factor**3)
        return values.reshape(-1)

    def gradient(self, values) -> np.ndarray:
        box = np.asarray(values).reshape(self.shape)
        spectrum = np.fft.rfftn(box)
        result = []
        for axis in range(3):
            k = self.wavenumbers[axis]
            shape = [1, 1, 1]
            shape[axis] = k.size
            derivative = np.fft.irfftn(spectrum * (1j * k.reshape(shape)), s=self.shape, axes=(0, 1, 2))
            result.append(derivative.reshape(-1))
        return np.stack(result)


def refined_channels(refined: RefinedSpectral, channels, levels, volume):
    """The channels with their occupied orbitals and chosen levels interpolated to the fine box (Euclidean-normalized there)."""

    fine_channels, fine_levels = [], {}
    scale = np.sqrt(refined.volume / volume)
    for index, (orbitals, occupations, eigenvalues, degeneracy) in enumerate(channels):
        keep = [column for column in range(orbitals.shape[1]) if occupations[column] != 0.0 or column in levels.get(index, ())]
        block = np.empty((int(np.prod(refined.shape)), len(keep)))
        for position, column in enumerate(keep):
            block[:, position] = refined.interpolate(orbitals[:, column]) * scale
        fine_channels.append((block, occupations[keep], eigenvalues[keep], degeneracy))
        fine_levels[index] = tuple(keep.index(column) for column in levels.get(index, ()))
    return fine_channels, fine_levels


class CodeDerivatives:
    """The operator of the run."""

    method = "code"

    def __init__(self, grid) -> None:
        from parsec_python.V_xc.meta_gga import grid_gradient

        self.gradient_operator = grid_gradient(grid)

    def gradient(self, values) -> np.ndarray:
        return self.gradient_operator.apply(np.asarray(values, dtype=np.float64))


def _host(array) -> np.ndarray:
    export = getattr(array, "to_full_host", None)
    if callable(export):
        return np.asarray(export(), dtype=np.float64)
    download = getattr(array, "get", None)
    if callable(download):
        return np.asarray(download(), dtype=np.float64)
    return np.asarray(array, dtype=np.float64)


def _channels(result):
    """``[(orbitals Q (Euclidean-normalized), occupations f, eigenvalues (Ry), degeneracy)]``."""

    if getattr(result, "spin", None) is not None:
        return [
            (_host(channel.wavefunctions), np.asarray(channel.occupations, dtype=np.float64),
             np.asarray(channel.eigenvalues, dtype=np.float64), 1.0)
            for channel in result.spin
        ]
    return [(
        _host(result.wavefunctions), np.asarray(result.occupations, dtype=np.float64),
        np.asarray(result.eigenvalues, dtype=np.float64), 2.0,
    )]


def _shells(values, density) -> list[float]:
    parts = []
    for upper, lower in zip(SHELL_EDGES[:-1], SHELL_EDGES[1:]):
        mask = (density <= upper) & (density > lower) if np.isfinite(upper) else density > lower
        parts.append(float(np.sum(values[mask])))
    parts.append(float(np.sum(values[density <= 0.0])))
    return parts


def _counts(mask, density) -> list[int]:
    counts = []
    for upper, lower in zip(SHELL_EDGES[:-1], SHELL_EDGES[1:]):
        shell = (density <= upper) & (density > lower) if np.isfinite(upper) else density > lower
        counts.append(int(np.count_nonzero(mask & shell)))
    return counts


def evaluate(evaluator, derivatives, channels, core, volume, meta: bool, levels: dict, form: str = "density") -> dict:
    """``E_xc``, its shells, ``tau < tau_W`` counts and the xc parts of the chosen levels for one operator.

    ``form`` is how ``grad n`` of ``sigma`` is made: ``density`` = the operator applied to ``n``
    (the discretization of ddf66501), ``orbital`` = ``sum_m d f_m 2 psi_m D psi_m`` (+ ``D n_core``),
    with which ``tau >= |grad n|^2/(8n)`` holds exactly.
    """

    from parsec_python.V_xc.meta_gga import weizsaecker_chain, weizsaecker_floor

    densities, taus, orbital_flux, level_orbitals = [], [], [], []
    for index, (orbitals, occupations, _levels, degeneracy) in enumerate(channels):
        weights = degeneracy * occupations / volume
        density = (orbitals * orbitals) @ weights
        densities.append(density)
        tau = np.zeros_like(density)
        product_gradient = np.zeros((3, density.size))
        chosen = {}
        for column in range(orbitals.shape[1]):
            if occupations[column] == 0.0 and column not in levels.get(index, ()):
                continue
            psi = orbitals[:, column] / np.sqrt(volume)
            gradient = derivatives.gradient(orbitals[:, column]) / np.sqrt(volume)
            if occupations[column] != 0.0:
                tau += 0.5 * degeneracy * occupations[column] * np.einsum("ij,ij->j", gradient, gradient)
                product_gradient += (2.0 * degeneracy * occupations[column]) * psi[None, :] * gradient
            if column in levels.get(index, ()):
                chosen[column] = gradient
        taus.append(tau)
        orbital_flux.append(product_gradient)
        level_orbitals.append(chosen)

    def gradient_of(channel, density, core_share):
        if form == "orbital":
            value = orbital_flux[channel]
            return value if core_share is None else value + derivatives.gradient(core_share)
        return derivatives.gradient(density)

    floored = meta and getattr(evaluator, "tau_floor", None) == "weizsaecker"
    if len(channels) == 1:
        n = densities[0] + (0.0 if core is None else core)
        grad_n = gradient_of(0, n, core)
        sigma = np.einsum("ij,ij->j", grad_n, grad_n)
        below_w = (n > 1e-30) & (taus[0] < sigma / (8.0 * np.maximum(n, 1e-300)))
        if meta:
            tau_in, below = weizsaecker_floor(np, n, sigma, taus[0]) if floored else (taus[0], None)
            energy, vrho, vsigma, vtau, _ = evaluator.unpolarized(n, sigma, tau_in, None)
            if floored:
                vrho, vsigma, vtau = weizsaecker_chain(np, n, sigma, below, vrho, vsigma, vtau)
        else:
            energy, vrho, vsigma = evaluator.unpolarized(n, sigma)
            vtau = None
        fields = [(vrho, 2.0 * vsigma[None, :] * grad_n, vtau)]
        total_density = n
        below_counts = _counts(below_w, n)
    else:
        half = None if core is None else 0.5 * core
        up = densities[0] + (0.0 if core is None else half)
        down = densities[1] + (0.0 if core is None else half)
        g_up, g_down = gradient_of(0, up, half), gradient_of(1, down, half)
        s_uu = np.einsum("ij,ij->j", g_up, g_up)
        s_ud = np.einsum("ij,ij->j", g_up, g_down)
        s_dd = np.einsum("ij,ij->j", g_down, g_down)
        below_counts = [0] * (len(SHELL_EDGES) - 1)
        for density, sigma_ss, tau in ((up, s_uu, taus[0]), (down, s_dd, taus[1])):
            mask = (density > 1e-30) & (tau < sigma_ss / (8.0 * np.maximum(density, 1e-300)))
            below_counts = [a + b for a, b in zip(below_counts, _counts(mask, up + down))]
        if meta:
            tau_up, tau_down = taus
            if floored:
                (tau_up, below_up), (tau_down, below_down) = (
                    weizsaecker_floor(np, up, s_uu, taus[0]), weizsaecker_floor(np, down, s_dd, taus[1]),
                )
            (energy, v_up, v_down, vs_uu, vs_ud, vs_dd, vt_up, vt_down, _, _) = evaluator.polarized(
                up, down, s_uu, s_ud, s_dd, tau_up, tau_down, None, None
            )
            if floored:
                v_up, vs_uu, vt_up = weizsaecker_chain(np, up, s_uu, below_up, v_up, vs_uu, vt_up)
                v_down, vs_dd, vt_down = weizsaecker_chain(np, down, s_dd, below_down, v_down, vs_dd, vt_down)
        else:
            energy, v_up, v_down, vs_uu, vs_ud, vs_dd = evaluator.polarized(up, down, s_uu, s_ud, s_dd)
            vt_up = vt_down = None
        fields = [
            (v_up, 2.0 * vs_uu[None, :] * g_up + vs_ud[None, :] * g_down, vt_up),
            (v_down, 2.0 * vs_dd[None, :] * g_down + vs_ud[None, :] * g_up, vt_down),
        ]
        total_density = up + down
    energy_ry = 2.0 * volume * np.asarray(energy, dtype=np.float64)
    level_parts = {}
    for channel, (orbitals, _occupations, _levels, _degeneracy) in enumerate(channels):
        vrho_s, flux, vtau_s = fields[channel]
        for column, gradient in level_orbitals[channel].items():
            q = orbitals[:, column]
            if form == "orbital":
                # <q| 2 sum_c [W_c D_c + D_c^T W_c] |q>/2 = 2 sum W . (q D q), W = flux (Hartree)
                gradient_term = 2.0 * float(np.sum(flux * (q[None, :] * gradient))) * np.sqrt(volume)
            else:
                gradient_term = float(np.sum(flux * derivatives.gradient(q * q)))
            local = float(np.dot(vrho_s, q * q)) + gradient_term
            kinetic = 0.0 if vtau_s is None else float(volume * np.dot(vtau_s, np.einsum("ij,ij->j", gradient, gradient)))
            level_parts[f"{channel}:{column}"] = dict(local=2.0 * local, tau=kinetic, total=2.0 * local + kinetic)
    return dict(
        exchange_correlation=float(np.sum(energy_ry)),
        shells=_shells(energy_ry, total_density),
        below_weizsaecker=below_counts,
        integral_tau=[float(volume * np.sum(tau)) for tau in taus],
        levels=level_parts,
    )


def _residual_levels(result, channels, levels, derivatives, volume) -> dict:
    """What the last residuals (output minus input of ``V`` and ``vtau``) do to the chosen levels (eV), and their norms by shell.

    ``<psi|V_out - V_in|psi>`` and ``int (vtau_out - vtau_in) |grad psi|^2``: first-order shifts of the
    levels if the last output potentials were used; the norms ``sqrt(h^3 sum d^2)`` per density shell.
    """

    record = getattr(result, "meta_gga", None)
    spin = getattr(result, "spin", None)
    if spin is not None:
        inputs = [np.asarray(channel.input_effective_potential) for channel in spin]
        outputs = [np.asarray(channel.output_effective_potential) for channel in spin]
        density = np.asarray(spin[0].density) + np.asarray(spin[1].density)
    else:
        inputs = [np.asarray(result.input_effective_potential)]
        outputs = [np.asarray(result.output_effective_potential)]
        density = np.asarray(result.density)
    report = {}
    try:
        for channel, columns in levels.items():
            orbitals = channels[channel][0]
            potential = outputs[channel] - inputs[channel]
            kinetic = None
            if record is not None:
                kinetic = np.asarray(record.output_kinetic_potentials[channel]) - np.asarray(record.kinetic_potentials[channel])
            norms = dict(
                potential=[float(np.sqrt(volume * value)) for value in _shells(potential * potential, density)],
                kinetic=None if kinetic is None else [float(np.sqrt(volume * value)) for value in _shells(kinetic * kinetic, density)],
            )
            report[f"norms_{channel}"] = norms
            for column in columns:
                q = orbitals[:, column]
                shift = float(np.dot(potential, q * q))
                if kinetic is not None:
                    gradient = derivatives.gradient(q)
                    shift_tau = float(np.dot(kinetic, np.einsum("ij,ij->j", gradient, gradient)))
                else:
                    shift_tau = 0.0
                report[f"{channel}:{column}"] = dict(potential_ev=shift * RYDBERG_EV, tau_ev=shift_tau * RYDBERG_EV)
    except Exception as error:  # diagnostics only
        report["error"] = repr(error)
    return report


def _levels_of(channels) -> dict:
    """Columns of the HOMO and LUMO of each channel."""

    chosen = {}
    for index, (_orbitals, occupations, eigenvalues, _degeneracy) in enumerate(channels):
        filled = np.flatnonzero(occupations > 1e-6)
        empty = np.flatnonzero(occupations <= 1e-6)
        picks = []
        if filled.size:
            picks.append(int(filled[np.argmax(eigenvalues[filled])]))
        if empty.size:
            picks.append(int(empty[np.argmin(eigenvalues[empty])]))
        chosen[index] = tuple(picks)
    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--molecule", default="H2O")
    parser.add_argument("--functional", default="scan", help="Correlation_Type (scan, pbe, ...)")
    parser.add_argument("--libxc", default=None, help="LibXC_Functional of the run (then --functional is ignored)")
    parser.add_argument("--spacing", type=float, default=0.16)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--criterion", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=None, help="Max_Iter (default 60; an --input keeps its own)")
    parser.add_argument(
        "--input", default=None,
        help="an existing parsec.in instead of a molecule (its functional and grid; the .DAT files beside it or up to "
        "three folders above it)",
    )
    parser.add_argument("--backend", default="cupy")
    parser.add_argument("--symmetry", default="auto")
    parser.add_argument("--orders", default="12,20,32")
    parser.add_argument("--no-fft", action="store_true")
    parser.add_argument("--forms", default="density,orbital", help="how grad n of sigma is made (see evaluate)")
    parser.add_argument(
        "--run-form", default="auto", help="the form of the code that ran (the reference of the deltas); auto: the run's"
    )
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--out", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--forces", action="store_true", help="also the forces of the run (Ry/bohr)")
    parser.add_argument("--move", default=None, help="ATOM,AXIS,DELTA: move atom ATOM (0-based, input order) by DELTA bohr")
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--shift", default=None, help="DX,DY,DZ: move every atom (bohr; the egg-box test)")
    parser.add_argument("--refine", type=int, default=0, help="also evaluate on a box this many times finer (spectral interpolation)")
    arguments = parser.parse_args(argv)
    if arguments.move:
        index, axis, delta = arguments.move.split(",")
        atoms, radius, states, spin = MOLECULES[arguments.molecule]
        atoms = [list(atom) for atom in atoms]
        atoms[int(index)][1 + int(axis)] += float(delta)
        moved = f"{arguments.molecule}@{arguments.move}"
        MOLECULES[moved] = ([tuple(atom) for atom in atoms], radius, states, spin)
        arguments.molecule = moved
    if arguments.shift:
        delta = [float(value) for value in arguments.shift.split(",")]
        atoms, radius, states, spin = MOLECULES[arguments.molecule]
        shifted = f"{arguments.molecule}+{arguments.shift}"
        MOLECULES[shifted] = (
            [(name, x + delta[0], y + delta[1], z + delta[2]) for name, x, y, z in atoms], radius, states, spin,
        )
        arguments.molecule = shifted

    from parsec_python.Input import parse_parsec_input
    from parsec_python.V_xc.meta_gga import is_meta_gga, meta_gga_evaluator

    directory = Path(tempfile.mkdtemp(prefix="meta_gga_accuracy_"))
    extra = list(arguments.extra)
    if arguments.libxc:
        extra.append(f"LibXC_Functional: {arguments.libxc}")
    if arguments.input is None:
        write_input(
            directory, arguments.molecule, "pbe" if arguments.libxc else arguments.functional, arguments.spacing,
            arguments.radius, arguments.criterion,
            60 if arguments.max_iterations is None else arguments.max_iterations, extra,
        )
    else:
        import re

        source = Path(arguments.input)
        for folder in (source.parent, *list(source.parent.parents)[:3]):
            for item in folder.glob("*.DAT"):
                if not (directory / item.name).exists():
                    shutil.copy(item, directory / item.name)
        for item in source.parent.glob("occup.in"):
            shutil.copy(item, directory / item.name)
        text = source.read_text()
        if arguments.max_iterations is not None:
            text, count = re.subn(r"(?im)^Max_Iter\s*[: ].*$", f"Max_Iter: {arguments.max_iterations}", text)
            if not count:
                text += f"Max_Iter: {arguments.max_iterations}\n"
        text += "".join(f"{line}\n" for line in extra)
        (directory / "parsec.in").write_text(text)
        arguments.molecule = source.parent.name
    problem = parse_parsec_input(directory / "parsec.in").problem
    history = []

    def report(record):
        if not hasattr(record, "iteration"):
            return
        history.append((int(record.iteration), float(record.weighted_residual), float(record.energies.total)))
        print(f"iter {record.iteration:3d} sre {record.weighted_residual:.3e} E {record.energies.total:.10f} "
              f"t {record.total_seconds:.2f}", flush=True)

    started = time.perf_counter()
    if arguments.backend == "reference":
        from parsec_python.SCF import prepare_single_point, run_scf

        system = prepare_single_point(problem)
        result = run_scf(system, callback=report)
    else:
        from parsec_python.acceleration.driver import prepare_single_point, run_scf

        system = prepare_single_point(problem, backend=arguments.backend, symmetry=arguments.symmetry)
        result = run_scf(system, callback=report).result
    scf_seconds = time.perf_counter() - started
    scf = system.input.scf
    meta = bool(is_meta_gga(scf))
    gradient_form = ("density", "not a meta-GGA")
    if meta:
        try:
            from parsec_python.V_xc.meta_gga import meta_gga_gradient_form

            gradient_form = tuple(meta_gga_gradient_form(scf))
        except ImportError:
            pass
    if arguments.run_form == "auto":
        arguments.run_form = gradient_form[0]
    if meta:
        evaluator = meta_gga_evaluator(scf)
    elif arguments.libxc:
        from parsec_python.V_xc.libxc import libxc_semilocal

        evaluator = libxc_semilocal(scf.xc_functional)
    else:
        from parsec_python.V_xc.semilocal_components import ComponentSemilocal, SemilocalComponent

        evaluator = ComponentSemilocal((SemilocalComponent("pbe_x", 1.0), SemilocalComponent("pbe_c", 1.0)))
    grid = system.grid
    volume = float(grid.volume_element)
    channels = _channels(result)
    levels = _levels_of(channels)
    core = np.asarray(result.core_density, dtype=np.float64) if np.any(result.core_density) else None
    density_check = sum((q * q) @ (d * f / volume) for q, f, _e, d in channels)
    eigen = {
        f"{index}:{column}": float(channels[index][2][column] * RYDBERG_EV)
        for index, columns in levels.items() for column in columns
    }
    forces = None
    if arguments.forces:
        if arguments.backend == "reference":
            from parsec_python.Forces.total import compute_forces
        else:
            from parsec_python.acceleration.driver import compute_forces
        computed = compute_forces(system, result, symmetrize=False)
        forces = dict(
            raw=np.asarray(computed.unsymmetrized).tolist(),
            net_removed=np.asarray(computed.total).tolist(),
            net=np.asarray(computed.net_force).tolist(),
            positions=np.asarray(computed.positions).tolist(),
        )
    analysis = {}
    operators = [] if arguments.no_analysis else [CodeDerivatives(grid)] + [BoxDerivatives(grid, f"fd{int(order)}") for order in arguments.orders.split(",") if order]
    if not arguments.no_fft and not arguments.no_analysis:
        operators.append(BoxDerivatives(grid, "fft"))
    forms = [form for form in arguments.forms.split(",") if form]
    for operator in operators:
        for form in forms:
            begun = time.perf_counter()
            key = f"{operator.method}/{form}"
            analysis[key] = evaluate(evaluator, operator, channels, core, volume, meta, levels, form)
            analysis[key]["seconds"] = time.perf_counter() - begun
    if arguments.refine > 1 and not arguments.no_analysis:
        begun = time.perf_counter()
        refined = RefinedSpectral(grid, arguments.refine)
        fine_channels, fine_levels = refined_channels(refined, channels, levels, volume)
        fine_core = None if core is None else refined.interpolate(core)
        names = {
            f"{index}:{position}": f"{index}:{column}"
            for index, columns in levels.items()
            for position, column in zip(fine_levels[index], columns)
        }
        prepared = time.perf_counter() - begun
        for form in forms:
            begun = time.perf_counter()
            item = evaluate(evaluator, refined, fine_channels, fine_core, refined.volume, meta, fine_levels, form)
            item["levels"] = {names[key]: value for key, value in item["levels"].items()}
            item["seconds"] = time.perf_counter() - begun + prepared
            item["fine_points"] = int(np.prod(refined.shape))
            analysis[f"{refined.method}/{form}"] = item
        del fine_channels
    code = analysis.get(f"code/{arguments.run_form}")
    for name, item in (analysis.items() if code is not None else ()):
        item["delta_exchange_correlation"] = item["exchange_correlation"] - code["exchange_correlation"]
        item["delta_shells"] = [a - b for a, b in zip(item["shells"], code["shells"])]
        item["delta_levels_ev"] = {
            key: (value["total"] - code["levels"][key]["total"]) * RYDBERG_EV for key, value in item["levels"].items()
        }
    residual_levels = _residual_levels(result, channels, levels, CodeDerivatives(grid), volume)
    energies = result.energies
    summary = dict(
        molecule=arguments.molecule,
        functional=arguments.libxc or arguments.functional,
        xc_functional=str(getattr(scf, "xc_functional", "")),
        meta_gga=meta,
        gradient_form=list(gradient_form),
        spacing=arguments.spacing,
        radius=arguments.radius,
        grid_points=int(grid.size),
        backend=arguments.backend,
        converged=bool(result.converged),
        iterations=int(result.iterations),
        total_energy=float(energies.total),
        exchange_correlation=float(energies.exchange_correlation),
        integral_vtau_tau=float(getattr(energies, "integral_vtau_tau", 0.0)),
        levels_ev=eigen,
        eigenvalues_ev=[(np.asarray(c[2]) * RYDBERG_EV).tolist() for c in channels],
        occupations=[np.asarray(c[1]).tolist() for c in channels],
        last_iterations=history[-8:],
        density_mismatch=float(np.max(np.abs(density_check - np.asarray(result.density)))),
        scf_seconds=scf_seconds,
        residual_levels=residual_levels,
        forces=forces,
        shell_edges=[str(edge) for edge in SHELL_EDGES],
        analysis=analysis,
    )
    text = json.dumps(summary, indent=1, default=str)
    print(text)
    if arguments.out:
        label = arguments.label or f"{arguments.molecule}_{(arguments.libxc or arguments.functional).replace(' ', '+')}_{arguments.spacing}"
        Path(arguments.out).mkdir(parents=True, exist_ok=True)
        (Path(arguments.out) / f"{label}.json").write_text(text)
    shutil.rmtree(directory, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
