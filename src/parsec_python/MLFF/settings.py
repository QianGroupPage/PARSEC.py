"""Settings of a relaxation with a machine-learned force field (``Initial_Structure: mlff``).

The structure of the input is first relaxed with an interatomic potential
such as UMA or MACE; the DFT calculation, of any kind, then starts from the
relaxed coordinates.  This module holds only plain data, so that the parser
can build it without importing a model package.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

#: Model of ``Initial_Structure: mlff`` without ``MLFF_Model``.
DEFAULT_MODEL = "uma-s-1p2"
#: UMA task of a molecule or cluster.
DEFAULT_TASK = "omol"
#: Largest force on an atom, in eV/angstrom, at which the relaxation stops.
DEFAULT_FMAX = 0.05
DEFAULT_STEPS = 500
OPTIMIZERS = ("lbfgs", "bfgs", "fire")
DEVICES = ("auto", "cuda", "cpu")
#: Variable naming the interpreter of the model when the input names none.
PYTHON_VARIABLE = "PARSEC_MLFF_PYTHON"


@dataclass(frozen=True)
class MLFFSettings:
    """What the relaxation with the force field uses.

    ``model`` names the potential (:mod:`parsec_python.MLFF.catalog`),
    ``task`` the UMA task, ``fmax`` the force criterion in eV/angstrom and
    ``steps`` the largest number of optimizer steps.  ``python`` is the
    interpreter that has the model package, ``None`` for the variable
    ``PARSEC_MLFF_PYTHON`` or else the running one.  ``charge`` is the
    charge of the system in elementary charges and ``spin`` its spin
    multiplicity, both used by the models that take them (UMA ``omol``);
    ``None`` lets the run derive them from the input.  ``reuse`` takes the
    structure an earlier run relaxed from the same input with the same
    settings.
    """

    model: str = DEFAULT_MODEL
    task: str = DEFAULT_TASK
    fmax: float = DEFAULT_FMAX
    steps: int = DEFAULT_STEPS
    optimizer: str = "lbfgs"
    device: str = "auto"
    python: Path | None = None
    charge: int | None = None
    spin: int | None = None
    reuse: bool = True

    def __post_init__(self) -> None:
        if not str(self.model).strip():
            raise ValueError("MLFF_Model cannot be empty")
        if not (self.fmax > 0.0):
            raise ValueError("MLFF_Fmax must be positive")
        if int(self.steps) < 0:
            raise ValueError("MLFF_Steps must be nonnegative")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError("MLFF_Optimizer must be one of " + ", ".join(OPTIMIZERS))
        if self.device not in DEVICES:
            raise ValueError("MLFF_Device must be one of " + ", ".join(DEVICES))
        if self.spin is not None and int(self.spin) < 1:
            raise ValueError("MLFF_Spin is a multiplicity 2S+1 and at least 1")


def interpreter(settings: MLFFSettings) -> str:
    """The Python interpreter that runs the model.

    The path is made absolute but not resolved: the interpreter of a
    virtual environment is a link whose target lies outside it, and only
    the link finds the packages of the environment.
    """

    if settings.python is not None:
        return os.path.abspath(os.path.expanduser(str(settings.python)))
    named = os.environ.get(PYTHON_VARIABLE, "").strip()
    if named:
        return os.path.abspath(os.path.expanduser(named))
    return sys.executable


__all__ = [
    "DEFAULT_FMAX",
    "DEFAULT_MODEL",
    "DEFAULT_STEPS",
    "DEFAULT_TASK",
    "DEVICES",
    "MLFFSettings",
    "OPTIMIZERS",
    "PYTHON_VARIABLE",
    "interpreter",
]
