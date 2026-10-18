"""The density models PARSEC.py can launch, and how to add another one.

Each model runs in its own Python environment.  PARSEC.py talks to it through
one small script, its *bridge*, run by that environment's interpreter: the
bridge reads a request ``.npz`` (atoms and the points where the density is
wanted) and writes a density ``.npz``.  ``bridges/template.py`` documents
the protocol and is a working bridge to copy.

A model is added permanently with a bridge in ``bridges/`` and one
:func:`register_provider` call below.  A model that is not built in needs no
change to PARSEC.py at all: ``Initial_Density: custom`` together with
``ML_Density_Bridge: path/to/bridge.py`` (and ``ML_Density_Python``) runs any
script that follows the protocol.

This module imports nothing of PARSEC.py, so the settings dataclasses can
validate against it without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping


BRIDGES = Path(__file__).resolve().with_name("bridges")
CUSTOM = "custom"
# ``Initial_Density`` values that are not models.
NON_MODEL_METHODS = ("sad", "file")


@dataclass(frozen=True)
class DensityProvider:
    """One density model PARSEC.py can run through a bridge.

    ``models`` lists the accepted ``ML_Density_Model`` values; an empty tuple
    accepts any name.  ``find_checkpoint(repository, model)`` returns the
    checkpoint used when ``ML_Density_Checkpoint`` is not given, or ``None``
    when the model has no conventional location.  ``environment`` is added to
    the bridge's environment unless the caller already sets a variable.
    """

    name: str
    title: str
    bridge: Path | None
    models: tuple[str, ...]
    default_model: str
    repository_variable: str | None = None
    python_variable: str | None = None
    needs_repository: bool = True
    needs_checkpoint: bool = True
    find_checkpoint: Callable[[Path, str], Path | None] | None = None
    checkpoint_hint: str = ""
    environment: Mapping[str, str] = field(default_factory=dict)
    source: str = ""
    summary: str = ""

    def check_model(self, model: str) -> str:
        """Return the normalized model name, or raise for an unknown one."""

        normalized = str(model).strip().lower()
        if not normalized:
            raise ValueError("ML density model name cannot be empty")
        if self.models and normalized not in self.models:
            raise ValueError(
                f"{self.title} model must be one of {', '.join(self.models)}; "
                f"got {model!r}"
            )
        return normalized


_PROVIDERS: dict[str, DensityProvider] = {}


def register_provider(provider: DensityProvider, *, replace: bool = False) -> None:
    """Make ``provider.name`` a valid ``Initial_Density`` value."""

    name = provider.name.strip().lower()
    if name != provider.name:
        raise ValueError("a provider name must be lower case without spaces")
    if name in NON_MODEL_METHODS:
        raise ValueError(f"{name!r} is reserved for a non-model initial density")
    if name in _PROVIDERS and not replace:
        raise ValueError(f"density provider {name!r} is already registered")
    _PROVIDERS[name] = provider


def provider_names() -> tuple[str, ...]:
    """Every ``Initial_Density`` value that runs a model."""

    return tuple(_PROVIDERS)


def get_provider(name: str) -> DensityProvider:
    try:
        return _PROVIDERS[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown density model {name!r}; available: "
            + ", ".join(provider_names())
        ) from None


def initial_density_methods() -> tuple[str, ...]:
    """Every accepted ``Initial_Density`` value."""

    return (*NON_MODEL_METHODS, *provider_names())


def _charge3net_checkpoint(repository: Path, model: str) -> Path | None:
    candidate = repository / "models" / f"charge3net_{model}.pt"
    return candidate if candidate.is_file() else None


def _scdp_checkpoint(repository: Path, model: str) -> Path | None:
    # SCDP publishes its weights on Zenodo, apart from its Git repository.
    # Accept the archive unpacked anywhere below ``models/`` of the checkout:
    # one checkpoint is unambiguous, and of several the directory name tells
    # the fast model (no virtual nodes) from the accurate one (bond nodes).
    candidates = sorted((repository / "models").glob("**/*.ckpt"))
    if len(candidates) == 1:
        return candidates[0]
    token = "_bond_" if model in {"accurate", "qm9-accurate"} else "_none_"
    matching = [path for path in candidates if token in path.as_posix()]
    return matching[0] if len(matching) == 1 else None


register_provider(
    DensityProvider(
        name="charge3net",
        title="ChargE3Net",
        bridge=BRIDGES / "charge3net.py",
        models=("qm9", "mp", "nmc"),
        default_model="qm9",
        repository_variable="CHARGE3NET_REPO",
        python_variable="CHARGE3NET_PYTHON",
        find_checkpoint=_charge3net_checkpoint,
        checkpoint_hint=(
            "the official repository ships models/charge3net_<model>.pt; "
            "set ML_Density_Checkpoint for another file"
        ),
        source="https://github.com/AIforGreatGood/charge3net",
        summary="E(3)-equivariant network evaluated directly at every grid point",
    )
)
register_provider(
    DensityProvider(
        name="scdp",
        title="SCDP",
        bridge=BRIDGES / "scdp.py",
        models=("fast", "accurate", "qm9", "qm9-fast", "qm9-accurate"),
        default_model="fast",
        repository_variable="SCDP_REPO",
        python_variable="SCDP_PYTHON",
        find_checkpoint=_scdp_checkpoint,
        checkpoint_hint=(
            "SCDP weights are distributed separately "
            "(https://zenodo.org/records/13146215); set ML_Density_Checkpoint "
            "to the .ckpt file, or unpack the archive below models/ of the "
            "SCDP checkout"
        ),
        # Newer PyTorch loads checkpoints weights-only by default; the
        # published SCDP files and e3nn 0.4 constants need the full loader.
        environment={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        source="https://github.com/kyonofx/scdp",
        summary="eSCN network predicting atom-centred Gaussian-orbital coefficients",
    )
)
register_provider(
    DensityProvider(
        name=CUSTOM,
        title="custom bridge",
        bridge=None,
        models=(),
        default_model="custom",
        needs_repository=False,
        needs_checkpoint=False,
        summary="any script following the bridge protocol (ML_Density_Bridge)",
    )
)


__all__ = [
    "BRIDGES",
    "CUSTOM",
    "DensityProvider",
    "get_provider",
    "initial_density_methods",
    "provider_names",
    "register_provider",
]
