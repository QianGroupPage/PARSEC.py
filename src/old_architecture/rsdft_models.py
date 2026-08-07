"""Shared RSDFT data containers.

These dataclasses are used to pass structured information between the
input/setup/output/solver modules so the new entry point does not need to
rely on the large set of mutable globals used by the legacy ``main.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .rsdft_hartree import normalize_hartree_method


A0_ANG = 0.529177210903
RY_EV = 13.605698066


@dataclass
class SolverSettings:
    """Normalized solver controls shared across the run.

    The fields here correspond to the main numerical knobs used by the
    legacy script: SCF stopping criteria, diagonalization settings,
    preconditioning, wavefunction output, and CPU/GPU selection.
    """
    fd_order: int = 8
    maxits: int = 40
    tol: float = 2.0e-4
    fermi_temp: float = 500.0
    save_wfn: int = 0
    cg_prec: int = 0
    poldeg: int = 10
    diagmeth: int = 3
    adaptive_scheme: int = 0
    use_gpu: int = 0
    recenter_atoms: int = 1
    hartree_method: str = "split"

    @staticmethod
    def _parse_tol(value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid tolerance") from exc

        if parsed >= 1 and abs(parsed - round(parsed)) < 1e-12:
            return 10 ** (-int(round(parsed)))
        return parsed

    @staticmethod
    def _parse_bool(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return 1 if value else 0
        if isinstance(value, str):
            raw = value.strip().lower()
            if raw in {"1", "true", "yes", "y", "on"}:
                return 1
            if raw in {"0", "false", "no", "n", "off"}:
                return 0
        raise ValueError("invalid boolean value")

    def normalize(self) -> None:
        """Clamp/normalize values after reading defaults or user overrides."""
        self.fd_order = int(self.fd_order)
        self.maxits = max(1, int(self.maxits))
        self.tol = float(self.tol)
        self.fermi_temp = float(self.fermi_temp)
        self.save_wfn = 1 if int(self.save_wfn) else 0
        self.cg_prec = 1 if int(self.cg_prec) else 0
        self.poldeg = max(1, int(self.poldeg))
        self.diagmeth = int(self.diagmeth)
        self.adaptive_scheme = 1 if int(self.adaptive_scheme) else 0
        self.use_gpu = 1 if int(self.use_gpu) else 0
        self.recenter_atoms = 1 if int(self.recenter_atoms) else 0
        self.hartree_method = normalize_hartree_method(self.hartree_method)

        if not 0 <= self.diagmeth <= 3:
            self.diagmeth = 3

    def apply_overrides(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        """Apply a user override dictionary and return the values that changed."""
        if not overrides:
            return {}

        mapping: dict[str, tuple[str, Callable[[Any], Any]]] = {
            "fd_order": ("fd_order", int),
            "maxits": ("maxits", int),
            "tol": ("tol", self._parse_tol),
            "fermi_temp": ("fermi_temp", float),
            "save_wfn": ("save_wfn", self._parse_bool),
            "write_wfn": ("save_wfn", self._parse_bool),
            "cg_prec": ("cg_prec", int),
            "poldeg": ("poldeg", int),
            "diagmeth": ("diagmeth", int),
            "adaptivescheme": ("adaptive_scheme", int),
            "adaptive_scheme": ("adaptive_scheme", int),
            "use_gpu": ("use_gpu", self._parse_bool),
            "gpu": ("use_gpu", self._parse_bool),
            "recenter_atoms": ("recenter_atoms", self._parse_bool),
            "recenter": ("recenter_atoms", self._parse_bool),
            "hartree_method": ("hartree_method", normalize_hartree_method),
            "hartree_potential_method": ("hartree_method", normalize_hartree_method),
            "hartree": ("hartree_method", normalize_hartree_method),
        }

        applied: dict[str, Any] = {}
        for raw_key, raw_value in overrides.items():
            if raw_value is None:
                continue

            key = str(raw_key).strip()
            target = mapping.get(key) or mapping.get(key.lower())
            if target is None:
                continue

            attr_name, caster = target
            try:
                value = caster(raw_value)
            except (TypeError, ValueError):
                print(f"Warning: Could not parse value for {raw_key}, keeping current setting.")
                continue

            setattr(self, attr_name, value)
            applied[attr_name] = value

        self.normalize()
        return applied


@dataclass
class InputData:
    """Geometry and user-input state before domain construction.

    Important fields:
        atoms, n_atom, z_charge: physical system definition.
        density_method, ml_file_path: how the initial density is built.
        settings_overrides: extra run controls parsed from the input source.
        grid_mode/grid_data: optional externally supplied ML grid metadata.
    """
    atoms: list[dict[str, Any]]
    n_atom: list[int]
    z_charge: float
    density_method: str = "sad"
    ml_file_path: str | None = None
    settings_overrides: dict[str, Any] = field(default_factory=dict)
    input_file_path: str | None = None
    grid_npy_path: str | None = None
    grid_poscar_path: str | None = None
    unit_used: str = "bohr"
    grid_mode: str = "default"
    grid_data: dict[str, Any] | None = None


@dataclass
class RunPaths:
    """Output file paths derived once the system/domain is known."""
    output_file: str
    wfn_file: str
    initial_density_base: str
    converged_density_base: str


@dataclass
class PreparedSystem:
    """Fully prepared RSDFT problem passed into the numerical solver.

    This is the boundary between the setup stage and the calculation stage:
    geometry has already been converted/recentered, the domain has been
    chosen, and all output paths are known.
    """
    input_data: InputData
    settings: SolverSettings
    domain: dict[str, Any]
    h: float
    nev: int
    zelec: float
    ztest: float
    n_types: int
    paths: RunPaths


@dataclass
class SolverBackend:
    """Concrete CPU or GPU implementation bundle for solver stages."""
    label: str
    pseudo_diag: Callable[..., Any]
    pseudo_nl: Callable[..., Any]
    pseudo_diag_source: str
    pseudo_nl_source: str
    xc: Callable[..., Any]
    mixer: Callable[..., Any]
    reset_mixer: Callable[..., Any]
    first_filt: Callable[..., Any]
    chefsi1: Callable[..., Any]
    lanczos: Callable[..., Any]
    chsubsp: Callable[..., Any]
    occupations: Callable[..., Any]
    pcg: Callable[..., Any]
    array_module: Any
    cupy_module: Any | None = None

    def to_numpy_array(self, value: Any) -> Any:
        """Convert backend arrays to NumPy arrays when needed for I/O/reporting."""
        cp = self.cupy_module
        if cp is not None and isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
        return value

    def to_host_scalar(self, value: Any) -> Any:
        """Convert backend scalar-like values to plain Python/NumPy scalars."""
        cp = self.cupy_module
        if cp is not None and isinstance(value, cp.ndarray):
            return cp.asnumpy(value).item()
        if cp is not None and isinstance(value, cp.generic):
            return value.item()
        try:
            import numpy as np

            if isinstance(value, np.ndarray) and value.ndim == 0:
                return value.item()
            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass
        return value


@dataclass
class EnergyComponents:
    """Compact energy summary in Rydberg units."""
    eigen_sum_ry: float
    hartree_ry: float
    xc_ry: float
    ion_ry: float
    total_ry: float


@dataclass
class SCFDiagnostics:
    """Convergence diagnostics for one SCF iteration."""
    potential_relative: float
    potential_norm: float
    potential_abs_norm: float
    potential_rms: float
    density_relative: float
    energy_change_ry: float | None


@dataclass
class SCFResult:
    """Final outputs from the self-consistent-field loop."""
    rho: Any
    hpot: Any
    xc_potential: Any
    exc: float
    potential: Any
    wavefunctions: Any
    eigenvalues: Any
    occupations: Any
    iterations: int
    error: float
    converged: bool
    e_nuc0: float
    hpot0: Any
    n_atoms: int
