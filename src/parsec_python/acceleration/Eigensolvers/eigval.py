"""Stateful CuPy first-solver-to-SUBSPACE eigensolver policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import os
from time import perf_counter
from typing import Any, Literal

import numpy as np

from parsec_python.Eigensolvers.eigval import EigvalSettings

from ..backends.cupy import (
    CuPyHamiltonian,
    CuPyTimingStats,
    require_cupy,
    resolve_device_stages,
    synchronized_call,
)
from .chebdav import DeviceChebDavResult, run_chebdav
from .chebff import DeviceChebFFResult, run_chebff
from .subspace import (
    DeviceSubspaceResult,
    DeviceSubspaceState,
    run_subspace_filter,
)
from .spectral_bounds import LanczosBoundResult, lanczos_upper_bound


SolverPath = Literal["chebff", "chebdav", "subspace"]


@dataclass(frozen=True)
class CuPyEigvalDeviceState:
    """Complete device state retained privately between public solves."""

    operator_dimension: int
    requested_states: int
    working_states: int
    initial_method: str
    subspace: DeviceSubspaceState
    solves_completed: int


@dataclass(frozen=True)
class CuPyEigvalResult:
    """Host view of requested states returned to SCF/modular callers."""

    eigenvalues: np.ndarray
    vectors: Any
    residual_norms: np.ndarray | None
    state: CuPyEigvalDeviceState
    solver_path: SolverPath
    restarted: bool
    restart_reason: str | None
    requested_states: int
    working_states: int
    solve_seconds: float

    @property
    def wavefunctions(self) -> np.ndarray:
        """Alias matching the physical name used by the SCF result model."""

        return self.vectors


def _reject_unsupported_settings(settings: EigvalSettings) -> None:
    if settings.initial_method not in {"chebff", "chebdav"}:
        raise ValueError("CuPy initial_method must be 'chebff' or 'chebdav'")


class CuPyEigvalSolver:
    """GPU-resident first/later SCF eigensolver.

    Construct from host static operators, then call :meth:`solve` once per SCF
    iteration.  Between calls, :meth:`update_local_potential` uploads only the
    new length-``N`` local field.  The buffered ``N x (states + safety)`` Ritz
    subspace remains in :attr:`device_state`; only the requested eigenpairs are
    downloaded in each returned :class:`CuPyEigvalResult`.
    """

    def __init__(
        self,
        negative_laplacian: Any,
        effective_potential: Any | None = None,
        nonlocal_operator: Any | None = None,
        *,
        settings: EigvalSettings = EigvalSettings(),
        timing_stats: CuPyTimingStats | None = None,
        retain_vectors_on_device: bool = False,
        compute_subspace_residuals: bool = True,
    ) -> None:
        _reject_unsupported_settings(settings)
        self.settings = settings
        self.retain_vectors_on_device = bool(retain_vectors_on_device)
        self.compute_subspace_residuals = bool(compute_subspace_residuals)
        self.timing_stats = timing_stats or CuPyTimingStats()
        if isinstance(negative_laplacian, CuPyHamiltonian):
            if effective_potential is not None or nonlocal_operator is not None:
                raise ValueError(
                    "a prebuilt CuPyHamiltonian cannot be combined with host terms"
                )
            self.operator = negative_laplacian
            # Keep one shared timing record even for a caller-built operator.
            self.timing_stats = self.operator.timing_stats
        else:
            if effective_potential is None:
                raise ValueError("effective_potential is required with a host Laplacian")
            self.operator = CuPyHamiltonian(
                negative_laplacian,
                effective_potential,
                nonlocal_operator,
                timing_stats=self.timing_stats,
            )
        self._state: CuPyEigvalDeviceState | None = None

    @classmethod
    def from_reference_hamiltonian(
        cls,
        hamiltonian: Any,
        *,
        settings: EigvalSettings = EigvalSettings(),
        timing_stats: CuPyTimingStats | None = None,
    ) -> "CuPyEigvalSolver":
        """Upload a ``parsec_python.KohnShamHamiltonian`` once."""

        return cls(
            hamiltonian.negative_laplacian,
            hamiltonian.effective_potential,
            hamiltonian.nonlocal_operator,
            settings=settings,
            timing_stats=timing_stats,
        )

    @property
    def device_state(self) -> CuPyEigvalDeviceState | None:
        """Expose state for diagnostics without converting its CuPy arrays."""

        return self._state

    def reset(self) -> None:
        """Discard the reusable GPU eigensubspace."""

        self._state = None

    def offload_state_to_host(self) -> CuPyEigvalDeviceState | None:
        """Move the persistent Ritz state to exact FP64 host storage.

        This is a memory-capacity policy used by the symmetry scheduler for
        systems whose independent representation states do not all fit on one
        device.  The Rayleigh--Ritz workspace is scratch and is deliberately
        discarded; all mathematical state (vectors, eigenvalues, bounds, and
        filter counters) is retained without conversion loss.
        """

        state = self._state
        if state is None:
            return None
        cp, _ = require_cupy()
        saved = state.subspace
        if not isinstance(saved.vectors, cp.ndarray):
            return state
        host_subspace = replace(
            saved,
            eigenvalues=np.ascontiguousarray(
                cp.asnumpy(saved.eigenvalues), dtype=np.float64
            ),
            vectors=np.asfortranarray(
                cp.asnumpy(saved.vectors), dtype=np.float64
            ),
            ritz_workspace=None,
        )
        self._state = replace(state, subspace=host_subspace)
        return self._state

    def restore_state_to_device(self) -> CuPyEigvalDeviceState | None:
        """Upload a previously host-spilled Ritz state to the active CUDA device."""

        state = self._state
        if state is None:
            return None
        cp, _ = require_cupy()
        saved = state.subspace
        if isinstance(saved.vectors, cp.ndarray):
            return state
        device_subspace = replace(
            saved,
            eigenvalues=cp.asarray(saved.eigenvalues, dtype=cp.float64),
            vectors=cp.asarray(
                saved.vectors, dtype=cp.float64, order="F"
            ),
            ritz_workspace=None,
        )
        self._state = replace(state, subspace=device_subspace)
        return self._state

    def truncate_state(self, requested_states: int) -> CuPyEigvalDeviceState:
        """Reduce the active saved subspace without restarting the solver.

        PARSEC changes ``eigen%nn`` after globally sorting representation
        eigenvalues while retaining the already allocated state arrays.  A
        CuPy view of the leading Ritz columns provides the equivalent active
        state: later SUBSPACE work uses fewer vectors, and a future request
        to grow again is handled by the normal explicit restart policy.
        """

        state = self._state
        if state is None:
            raise ValueError("cannot truncate an empty eigensolver state")
        requested_states = int(requested_states)
        working_states = self._working_state_count(requested_states)
        if working_states > state.working_states:
            raise ValueError("truncate_state cannot grow the saved subspace")
        if (
            requested_states == state.requested_states
            and working_states == state.working_states
        ):
            return state
        saved = state.subspace
        subspace = replace(
            saved,
            working_states=working_states,
            eigenvalues=saved.eigenvalues[:working_states],
            vectors=saved.vectors[:, :working_states],
        )
        self._state = replace(
            state,
            requested_states=requested_states,
            working_states=working_states,
            subspace=subspace,
        )
        return self._state

    def update_settings(self, settings: EigvalSettings) -> None:
        """Replace settings; an initial-method change restarts on next solve."""

        _reject_unsupported_settings(settings)
        self.settings = settings

    def update_local_potential(self, effective_potential: Any) -> None:
        """Update ``V_ion,local + V_H + V_xc`` without moving eigensubspace data."""

        self.operator.update_local_potential(effective_potential)

    def _working_state_count(self, requested_states: int) -> int:
        dimension = int(self.operator.shape[0])
        requested_states = int(requested_states)
        if not 1 <= requested_states <= dimension:
            raise ValueError("requested_states is outside the operator dimension")
        return min(dimension, requested_states + self.settings.safety_buffer)

    def prepare_spectral_bound(
        self,
        requested_states: int,
        *,
        settings: EigvalSettings | None = None,
        reset: bool = False,
    ) -> LanczosBoundResult:
        """Run exactly the bound that the next :meth:`solve` would request.

        Symmetry-sector orchestration may execute these independent, narrow
        Lanczos recurrences on separate CUDA streams before serializing the
        bandwidth-heavy filters.  Seeds, steps, and saved-state policy are
        identical to the in-solver path.
        """

        if settings is not None:
            self.update_settings(settings)
        working_states = self._working_state_count(int(requested_states))
        reason = self._incompatibility(int(requested_states), working_states)
        first = reset or self._state is None or reason is not None
        if first:
            options = (
                self.settings.chebff
                if self.settings.initial_method == "chebff"
                else self.settings.chebdav
            )
            seed = options.random_seed
            steps = options.lanczos_steps
        else:
            assert self._state is not None
            options = self.settings.subspace
            seed = options.random_seed + self._state.subspace.filters_completed
            steps = options.lanczos_steps
        return lanczos_upper_bound(
            self.operator,
            steps=steps,
            rng=np.random.default_rng(seed),
        )

    def _incompatibility(
        self,
        requested_states: int,
        working_states: int,
    ) -> str | None:
        state = self._state
        if state is None:
            return None
        dimension = int(self.operator.shape[0])
        if state.operator_dimension != dimension:
            return "operator_dimension_changed"
        if state.requested_states != requested_states:
            return "requested_state_count_changed"
        if state.working_states != working_states:
            return "working_state_count_changed"
        if state.initial_method != self.settings.initial_method:
            return "initial_solver_changed"
        if state.solves_completed < 1:
            return "invalid_saved_solve_count"
        saved = state.subspace
        if saved.operator_dimension != dimension:
            return "saved_subspace_dimension_changed"
        if saved.working_states != working_states:
            return "saved_subspace_state_count_changed"
        if saved.eigenvalues.shape != (working_states,):
            return "saved_eigenvalue_shape_changed"
        if saved.vectors.shape != (dimension, working_states):
            return "saved_vector_shape_changed"
        return None

    @staticmethod
    def _state_from_first(
        first: DeviceChebFFResult | DeviceChebDavResult,
        requested_states: int,
        initial_method: str,
    ) -> CuPyEigvalDeviceState:
        subspace = DeviceSubspaceState(
            operator_dimension=first.state.operator_dimension,
            working_states=first.state.wanted_states,
            eigenvalues=first.state.eigenvalues,
            vectors=first.state.vectors,
            filter_lower_bound=None,
            first_filter=True,
            filters_completed=0,
        )
        return CuPyEigvalDeviceState(
            operator_dimension=first.state.operator_dimension,
            requested_states=requested_states,
            working_states=first.state.wanted_states,
            initial_method=initial_method,
            subspace=subspace,
            solves_completed=1,
        )

    def solve(
        self,
        requested_states: int,
        *,
        settings: EigvalSettings | None = None,
        spectral_bound: LanczosBoundResult | None = None,
    ) -> CuPyEigvalResult:
        """Run the selected first solver, then one SUBSPACE pass per later call."""

        if settings is not None:
            self.update_settings(settings)
        requested_states = int(requested_states)
        working_states = self._working_state_count(requested_states)
        previous_state = self._state
        reason = self._incompatibility(requested_states, working_states)
        restart = previous_state is not None and reason is not None
        fallback_reason = None

        if previous_state is None or reason is not None:
            if self.settings.initial_method == "chebff":
                first_solver = run_chebff
                first_settings = self.settings.chebff
            else:
                first_solver = run_chebdav
                first_settings = self.settings.chebdav
            first_options = {"settings": first_settings}
            if spectral_bound is not None:
                first_options["spectral_bound"] = spectral_bound
            try:
                first, solve_seconds = synchronized_call(
                    first_solver,
                    self.operator,
                    working_states,
                    **first_options,
                )
            except np.linalg.LinAlgError as error:
                fallback = os.environ.get(
                    "PARSEC_CUPY_CHEBDAV_FAILURE_FALLBACK", "off"
                ).strip().lower()
                if fallback not in {"off", "chebff"}:
                    raise ValueError(
                        "PARSEC_CUPY_CHEBDAV_FAILURE_FALLBACK must be off or chebff"
                    ) from error
                if self.settings.initial_method != "chebdav" or fallback != "chebff":
                    raise
                # This opt-in recovery changes only the eigensolver used for
                # the first invariant subspace.  It retains the requested
                # state count, Hamiltonian, precision, and all later PARSEC
                # SUBSPACE iterations.  Drop failed Davidson temporaries
                # before launching CHEBFF in the same CUDA context.
                cp, _ = require_cupy()
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                first_options = {"settings": self.settings.chebff}
                first, solve_seconds = synchronized_call(
                    run_chebff,
                    self.operator,
                    working_states,
                    **first_options,
                )
                first_solver = run_chebff
                fallback_reason = (
                    "chebdav orthogonalization failed; explicit chebff "
                    f"fallback: {error}"
                )
            resolve_device_stages(self.timing_stats)
            self._state = self._state_from_first(
                first,
                requested_states,
                self.settings.initial_method,
            )
            solver_path = (
                "chebff" if first_solver is run_chebff else "chebdav"
            )
            residual_device = (
                None
                if solver_path == "chebff"
                else first.residual_norms
            )
            self.timing_stats.first_solve_seconds += solve_seconds
            self.timing_stats.first_solve_calls += 1
        else:
            subspace_options = {"settings": self.settings.subspace}
            if spectral_bound is not None:
                subspace_options["spectral_bound"] = spectral_bound
            if not self.compute_subspace_residuals:
                subspace_options["compute_residuals"] = False
            later, solve_seconds = synchronized_call(
                run_subspace_filter,
                self.operator,
                previous_state.subspace,
                **subspace_options,
            )
            resolve_device_stages(self.timing_stats)
            self._state = CuPyEigvalDeviceState(
                operator_dimension=previous_state.operator_dimension,
                requested_states=requested_states,
                working_states=working_states,
                initial_method=previous_state.initial_method,
                subspace=later.state,
                solves_completed=previous_state.solves_completed + 1,
            )
            solver_path = "subspace"
            residual_device = later.residual_norms
            self.timing_stats.subspace_solve_seconds += solve_seconds
            self.timing_stats.subspace_solve_calls += 1

        self.timing_stats.solve_calls += 1
        assert self._state is not None
        cp, _ = require_cupy()

        def download():
            saved = self._state.subspace
            eigenvalues = cp.asnumpy(saved.eigenvalues[:requested_states])
            vectors = (
                saved.vectors[:, :requested_states]
                if self.retain_vectors_on_device
                else cp.asnumpy(saved.vectors[:, :requested_states])
            )
            residuals = (
                None
                if residual_device is None
                else cp.asnumpy(residual_device[:requested_states])
            )
            return eigenvalues, vectors, residuals

        # The first/subspace call above has already synchronized this solver's
        # stream.  cp.asnumpy performs the only synchronization needed by the
        # transfer itself, so wrapping it in synchronized_call would add two
        # empty stream barriers to every representation and every SCF step.
        download_started = perf_counter()
        downloaded = download()
        download_seconds = perf_counter() - download_started
        self.timing_stats.download_seconds += download_seconds
        eigenvalues, vectors, residuals = downloaded
        return CuPyEigvalResult(
            eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
            vectors=(
                vectors
                if self.retain_vectors_on_device
                else np.asarray(vectors, dtype=np.float64)
            ),
            residual_norms=(
                None
                if residuals is None
                else np.asarray(residuals, dtype=np.float64)
            ),
            state=self._state,
            solver_path=solver_path,
            restarted=restart,
            restart_reason=(reason if reason is not None else fallback_reason),
            requested_states=requested_states,
            working_states=working_states,
            solve_seconds=float(solve_seconds),
        )


__all__ = [
    "CuPyEigvalDeviceState",
    "CuPyEigvalResult",
    "CuPyEigvalSolver",
    "EigvalSettings",
]
