"""Relax one structure with a machine-learned force field (run as a script).

``python worker.py request.json`` reads the request that
:mod:`parsec_python.MLFF.run` wrote, relaxes the structure with an ASE
calculator and writes the result beside it.  The script imports nothing of
``parsec_python``: it runs in whatever environment holds the model package
(``MLFF_Python``), which needs only NumPy and ASE besides that package, and
its process ends after the relaxation, so that the device memory of the
model is returned before the DFT calculation starts.

``python worker.py --check MODEL [--device D]`` builds the calculator and
evaluates a water molecule, to test an installation.

Models (``MLFF_Model``):

- ``uma-s-1p2``, ``uma-s-1p1``, ``uma-m-1p1`` and the other names of
  ``fairchem.core.pretrained_mlip.available_models`` (UMA, eSEN; package
  ``fairchem-core``), or a checkpoint ``*.pt``; ``MLFF_Task`` chooses the
  UMA task, ``omol`` for molecules and clusters.
- ``mace-omol``, ``mace-off-small|medium|large``, ``mace-mp-<name>`` with a
  name of ``mace.calculators.foundations_models.mace_mp_names`` (for example
  ``mace-mp-medium-mpa-0``), or a model file ``*.model`` (package
  ``mace-torch``).
- ``python:module:function`` or ``python:path/file.py:function``: a function
  that returns any ASE calculator; it is called with the keywords
  ``device``, ``task``, ``charge`` and ``spin``.

Units are ASE's: angstrom and eV.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

#: Where a missing package comes from, by backend.
INSTALL = {
    "fairchem": "pip install fairchem-core   (UMA weights are gated: accept the licence at "
    "https://huggingface.co/facebook/UMA and log in with `hf auth login`)",
    "mace": "pip install mace-torch",
}


class ModelError(RuntimeError):
    """The model cannot be built in this environment."""


def backend_of(model: str) -> str:
    """``fairchem``, ``mace`` or ``python`` for a model name."""

    name = model.strip()
    lower = name.lower()
    if lower.startswith("python:"):
        return "python"
    if lower.startswith("mace") or lower.endswith(".model"):
        return "mace"
    if lower.startswith(("uma", "esen", "allscaip", "fairchem:")) or lower.endswith(".pt"):
        return "fairchem"
    raise ModelError(
        f"unknown MLFF_Model {model!r}: use a UMA or eSEN name (uma-s-1p2), a MACE name "
        "(mace-omol, mace-off-medium, mace-mp-medium-mpa-0), a model file (*.model, *.pt) "
        "or python:module:function"
    )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _import(module: str, backend: str):
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ModelError(
            f"{module} cannot be imported by {sys.executable} ({error}); install it with "
            f"`{INSTALL[backend]}`, or name an interpreter that has it with MLFF_Python"
        ) from error


def _python_factory(spec: str):
    target = spec.split(":", 1)[1]
    location, separator, function = target.rpartition(":")
    if not separator or not location or not function:
        raise ModelError("python: models are named python:module:function or python:file.py:function")
    if location.endswith(".py"):
        path = Path(location).expanduser()
        if not path.is_file():
            raise ModelError(f"{location} does not exist")
        loaded = importlib.util.spec_from_file_location(f"parsec_mlff_{path.stem}", path)
        module = importlib.util.module_from_spec(loaded)
        loaded.loader.exec_module(module)
    else:
        module = importlib.import_module(location)
    factory = getattr(module, function, None)
    if not callable(factory):
        raise ModelError(f"{location} has no function {function}")
    return factory


def _fairchem_inference():
    """fairchem's default inference settings, compiled only on request.

    The default of fairchem compiles the model with ``torch.compile``, which
    costs more than a relaxation of some tens of steps saves (C795H300 on an
    A100: 22 steps in 90 s compiled, 9 s not).  ``PARSEC_MLFF_COMPILE=on``
    compiles where the C++ compiler of the platform is found (``cl`` on
    Windows, ``g++`` elsewhere); ``off``, the default, does not.
    """

    import dataclasses
    import shutil

    try:
        from fairchem.core.units.mlip_unit.api.inference import inference_settings_default
    except ImportError:
        return "default"
    settings = inference_settings_default()
    if (
        hasattr(settings, "execution_mode")
        and settings.execution_mode is None
        and importlib.util.find_spec("triton") is None
    ):
        # On a GPU fairchem picks kernels written in Triton, which the
        # PyTorch of Windows does not bring: the PyTorch path instead.
        settings = dataclasses.replace(settings, execution_mode="umas_fast_pytorch")
    if not getattr(settings, "compile", False):
        return settings
    choice = os.environ.get("PARSEC_MLFF_COMPILE", "off").strip().lower() or "off"
    if choice not in {"on", "1", "true", "yes"}:
        return dataclasses.replace(settings, compile=False)
    compiler = "cl" if sys.platform == "win32" else "g++"
    if shutil.which(compiler) is None:
        print(f"MLFF: no {compiler} on the PATH, the fairchem model runs without torch.compile", flush=True)
        return dataclasses.replace(settings, compile=False)
    return settings


def _download_advice(error: Exception) -> str:
    """What to do about a download of fairchem weights that failed."""

    text = f"{type(error).__name__} {error}"
    if "Gated" in text or "401" in text or "403" in text:
        return (
            "UMA weights are gated: accept the licence at https://huggingface.co/facebook/UMA, log in "
            "with `hf auth login` (or set HF_TOKEN), or give a local checkpoint *.pt"
        )
    if getattr(error, "errno", None) == 524 or "lock" in text.lower():
        cache = os.environ.get("FAIRCHEM_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache/fairchem"))
        return (
            f"the cache {cache} does not take file locks here (the compute nodes of some clusters, "
            "Perlmutter among them, mount the home directory without them): download the weights once "
            "where it does, for example `mlff.py check MODEL` on a login node, or set FAIRCHEM_CACHE_DIR "
            "to a scratch directory"
        )
    return (
        "check the network and the Hugging Face login (`hf auth whoami`), or give a local checkpoint *.pt"
    )


def build_calculator(model: str, *, device: str, task: str, charge: int, spin: int):
    """Return ``(calculator, description)`` for a model name."""

    backend = backend_of(model)
    name = model.strip()
    lower = name.lower()
    if backend == "python":
        calculator = _python_factory(name)(device=device, task=task, charge=charge, spin=spin)
        return calculator, f"{name} (user function)"
    if backend == "fairchem":
        imported = time.perf_counter()
        fairchem = _import("fairchem.core", "fairchem")
        print(f"MLFF: fairchem imported in {time.perf_counter() - imported:.1f} s", flush=True)
        # fairchem.core exports the module; it moved between releases.
        pretrained = getattr(fairchem, "pretrained_mlip", None) or importlib.import_module(
            "fairchem.core.calculate.pretrained_mlip"
        )
        inference = _fairchem_inference()
        if lower.endswith(".pt"):
            unit = pretrained.load_predict_unit(name, inference_settings=inference, device=device)
            label = Path(name).name
        else:
            label = lower.split(":", 1)[1] if lower.startswith("fairchem:") else lower
            available = tuple(getattr(pretrained, "available_models", ()))
            if available and label not in available:
                raise ModelError(
                    f"fairchem {getattr(fairchem, '__version__', '')} has no model {label!r}; "
                    "it has " + ", ".join(available)
                )
            try:
                unit = pretrained.get_predict_unit(label, inference_settings=inference, device=device)
            except Exception as error:  # gated weights, network, cache
                raise ModelError(
                    f"the weights of {label} could not be loaded ({type(error).__name__}: {error}); "
                    + _download_advice(error)
                ) from error
        calculator = fairchem.FAIRChemCalculator(unit, task_name=task)
        return calculator, f"{label} (fairchem {getattr(fairchem, '__version__', '?')}, task {task})"
    mace = _import("mace.calculators", "mace")
    version = getattr(importlib.import_module("mace"), "__version__", "?")
    if lower.endswith(".model"):
        calculator = mace.MACECalculator(model_paths=name, device=device, default_dtype="float64")
        return calculator, f"{Path(name).name} (mace-torch {version})"
    if lower in {"mace-omol", "mace-omol-0"}:
        calculator = mace.mace_omol(model="extra_large", device=device, default_dtype="float64")
    elif lower.startswith("mace-off-") or lower.startswith("mace-off:"):
        calculator = mace.mace_off(model=lower[9:], device=device, default_dtype="float64")
    elif lower.startswith("mace-mp-") or lower.startswith("mace-mp:"):
        calculator = mace.mace_mp(model=lower[8:], device=device, default_dtype="float64")
    else:
        raise ModelError(
            f"unknown MACE model {model!r}: use mace-omol, mace-off-small|medium|large, "
            "mace-mp-<name> (for example mace-mp-medium-mpa-0) or a file *.model"
        )
    return calculator, f"{name} (mace-torch {version})"


class PointGroup:
    """ASE constraint that keeps the point group of the structure.

    ``matrices[g] @ r[a]`` is the position of atom ``images[g][a]``.  Forces
    and new positions are averaged over the group, so that a structure the
    DFT calculation solves in symmetry sectors stays in its group to
    round-off.
    """

    def __init__(self, matrices, images):
        self.matrices = np.asarray(matrices, dtype=np.float64)
        self.images = np.asarray(images, dtype=np.int64)

    def _average(self, vectors):
        total = np.zeros_like(vectors)
        for matrix, image in zip(self.matrices, self.images):
            total[image] += vectors @ matrix.T
        return total / float(len(self.matrices))

    def adjust_positions(self, atoms, new):
        new[:] = self._average(np.asarray(new, dtype=np.float64))

    def adjust_forces(self, atoms, forces):
        forces[:] = self._average(np.asarray(forces, dtype=np.float64))

    def adjust_momenta(self, atoms, momenta):
        momenta[:] = self._average(np.asarray(momenta, dtype=np.float64))

    def get_removed_dof(self, atoms):
        return 0

    def index_shuffle(self, atoms, ind):
        raise NotImplementedError("the point-group constraint does not follow a reordering")

    def todict(self):
        return {
            "name": "PointGroup",
            "kwargs": {"matrices": self.matrices.tolist(), "images": self.images.tolist()},
        }


def _frame(handle, symbols, positions, forces, energy, step, fmax):
    handle.write(f"{len(symbols)}\n")
    handle.write(
        'Properties=species:S:1:pos:R:3:forces:R:3 pbc="F F F" '
        f"energy={energy:.10f} step={step} fmax={fmax:.6f}\n"
    )
    for symbol, position, force in zip(symbols, positions, forces):
        handle.write(
            f"{symbol:<3s} {position[0]:16.10f} {position[1]:16.10f} {position[2]:16.10f} "
            f"{force[0]:14.8f} {force[1]:14.8f} {force[2]:14.8f}\n"
        )
    handle.flush()


def relax(request: dict) -> dict:
    """Relax the structure of ``request``; return the result."""

    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS, FIRE, LBFGS

    started = time.perf_counter()
    device = resolve_device(request.get("device", "auto"))
    symbols = list(request["symbols"])
    try:
        atoms = Atoms(symbols=symbols, positions=np.asarray(request["positions"], dtype=np.float64))
    except (KeyError, ValueError) as error:
        raise ModelError(f"the atoms need chemical element symbols ({error})") from error
    charge = int(request.get("charge") or 0)
    spin = request.get("spin")
    if spin is None:
        # The smallest multiplicity of the electron count.
        spin = 1 if (int(atoms.get_atomic_numbers().sum()) - charge) % 2 == 0 else 2
    spin = int(spin)
    calculator, description = build_calculator(
        request["model"], device=device, task=request.get("task", "omol"), charge=charge, spin=spin
    )
    loaded = time.perf_counter()
    atoms.info["charge"], atoms.info["spin"] = charge, spin
    constraints = []
    if request.get("fixed"):
        constraints.append(FixAtoms(indices=list(request["fixed"])))
    group = request.get("symmetry")
    if group and len(group["matrices"]) > 1:
        constraints.append(PointGroup(group["matrices"], group["images"]))
    atoms.calc = calculator
    # Constraints after the calculator: the start is symmetrized once.
    atoms.set_constraint(constraints)
    if constraints:
        atoms.set_positions(atoms.get_positions(), apply_constraint=True)

    optimizer = {"lbfgs": LBFGS, "bfgs": BFGS, "fire": FIRE}[request.get("optimizer", "lbfgs")]
    log_path = request.get("log")
    log_handle = open(log_path, "w", encoding="utf-8") if log_path else None
    trajectory = open(request["trajectory"], "w", encoding="utf-8") if request.get("trajectory") else None
    energies, maxima = [], []

    def record():
        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        largest = float(np.sqrt((forces**2).sum(axis=1)).max()) if len(forces) else 0.0
        energies.append(energy)
        maxima.append(largest)
        if trajectory is not None:
            _frame(trajectory, symbols, atoms.get_positions(), forces, energy, len(energies) - 1, largest)
        every = max(1, int(request.get("report_every", 10)))
        if (len(energies) - 1) % every == 0:
            print(f"MLFF step {len(energies) - 1:5d}  E = {energy:18.8f} eV  fmax = {largest:10.5f} eV/ang",
                  flush=True)

    try:
        steps = int(request.get("steps", 500))
        fmax = float(request.get("fmax", 0.05))
        if steps == 0:
            record()
        else:
            run = optimizer(atoms, logfile=log_handle if log_handle is not None else os.devnull)
            run.attach(record, interval=1)
            run.run(fmax=fmax, steps=steps)
        if not energies:
            record()
    finally:
        if log_handle is not None:
            log_handle.close()
        if trajectory is not None:
            trajectory.close()
    finished = time.perf_counter()
    return {
        "positions": atoms.get_positions().tolist(),
        "energies": energies,
        "fmax": maxima,
        "steps": len(energies) - 1,
        "converged": bool(maxima and maxima[-1] <= float(request.get("fmax", 0.05))),
        "model": description,
        "device": device,
        "charge": charge,
        "spin": spin,
        "seconds": {"load": loaded - started, "relax": finished - loaded},
        "python": sys.executable,
    }


def check(model: str, device: str) -> int:
    """Build ``model`` and evaluate a water molecule; print what happened."""

    from ase import Atoms

    device = resolve_device(device)
    started = time.perf_counter()
    calculator, description = build_calculator(model, device=device, task="omol", charge=0, spin=1)
    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.0, -0.763, -0.477]])
    atoms.info["charge"], atoms.info["spin"] = 0, 1
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(f"{description} on {device} with {sys.executable}")
    print(f"water: E = {energy:.6f} eV, largest force {np.sqrt((forces**2).sum(axis=1)).max():.4f} eV/ang, "
          f"{time.perf_counter() - started:.1f} s")
    return 0 if math.isfinite(energy) else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # A download of weights that stalls ends with an error instead of
    # holding the DFT run (PARSEC_MLFF_NETWORK_TIMEOUT seconds per read).
    import socket

    socket.setdefaulttimeout(float(os.environ.get("PARSEC_MLFF_NETWORK_TIMEOUT", "300")))
    try:
        if argv and argv[0] == "--check":
            device = argv[argv.index("--device") + 1] if "--device" in argv else "auto"
            return check(argv[1], device)
        if len(argv) != 1:
            print("usage: worker.py request.json | worker.py --check MODEL [--device D]", file=sys.stderr)
            return 2
        path = Path(argv[0])
        request = json.loads(path.read_text(encoding="utf-8"))
        result = relax(request)
        Path(request["result"]).write_text(json.dumps(result), encoding="utf-8")
        return 0
    except ModelError as error:
        print(f"MLFF error: {error}", file=sys.stderr, flush=True)
        if len(argv) == 1 and not argv[0].startswith("--"):
            request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
            Path(request["result"]).write_text(json.dumps({"error": str(error)}), encoding="utf-8")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
