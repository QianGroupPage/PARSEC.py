"""Numerical RSDFT driver for the refactored code path.

The main public function here is ``run_rsdft_calculation()``. It performs the
same broad stages as the legacy ``main.py``:
1. Build the Laplacian and ionic terms.
2. Construct the initial density/potentials.
3. Build nonlocal and XC pieces.
4. Run the self-consistent-field loop.
5. Save final densities, energies, and optional wavefunctions.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .Eigensolvers.pcg import pcg as pcg_cpu
from .Laplacian.fd3d import fd3d
from .Laplacian.nuclear import nuclear
from .V_ion.nelectrons import nelectrons
from .V_ion.pseudoNL_original_ML4Den import pseudoNL_ML4Den

from .rsdft_diagnostics import (
    compute_orbital_stationarity_diagnostic,
    write_orbital_stationarity_diagnostic,
)
from .rsdft_hartree import solve_hartree_from_grid_density
from .rsdft_models import EnergyComponents, PreparedSystem, SCFDiagnostics, SCFResult, SolverBackend
from .rsdft_output import (
    TimingRecorder,
    print_eigenvalues_to_console,
    print_timing_analysis,
    print_total_energy_summary,
    save_density_variants,
    save_wavefunction,
    write_diag_info,
    write_initial_density_diagnostics,
    write_scf_iteration,
    write_timing_analysis,
    write_total_energy_summary,
)


def _scaled_poldeg(poldeg: int, degree_modifier: float) -> int:
    """Convert the adaptive polynomial scale factor into a safe integer degree."""
    return max(1, int(round(poldeg * degree_modifier)))


def _scaled_lanczos_steps(nev: int, modifier: float) -> int:
    """Convert the adaptive Lanczos subspace scaling into an integer step count."""
    return nev + max(1, int(round(500 * modifier)))


def _density_setup_needs_cpu_laplacian(density_method: str) -> bool:
    """Return True when the initial-density path still needs a CPU Laplacian."""
    return density_method not in {"sad", "sad_ml_grid"}


def _build_laplacians(problem: PreparedSystem, backend: SolverBackend):
    """Build the Laplacian matrices required by the selected backend."""
    h = problem.domain["h"]
    nx, ny, nz = problem.domain["nx"], problem.domain["ny"], problem.domain["nz"]
    scale = 1.0 / (h * h)

    if backend.label != "gpu":
        return scale * fd3d(nx, ny, nz, problem.settings.fd_order), None

    laplacian_cpu = None
    if _density_setup_needs_cpu_laplacian(problem.input_data.density_method):
        laplacian_cpu = scale * fd3d(nx, ny, nz, problem.settings.fd_order)

    try:
        from .Laplacian.fd3d_gpu import fd3d_gpu
    except ImportError as exc:
        raise SystemExit("GPU backend requested but fd3d_gpu could not be imported.") from exc

    laplacian_gpu = fd3d_gpu(nx, ny, nz, problem.settings.fd_order, scale=scale)
    return laplacian_cpu, laplacian_gpu


def _build_preconditioner(A, cg_prec: int, backend: SolverBackend, density_method: str):
    """Build the ILU preconditioner used by the CPU Hartree solve."""
    if not cg_prec:
        return []

    if backend.label == "gpu" and not _density_setup_needs_cpu_laplacian(density_method):
        print("Skipping CPU ILU build on GPU backend; the GPU Hartree path does not use it.")
        return []

    if A is None:
        return []

    print("Calling ilu0 ...")
    preconditioner = spla.spilu(A)
    print("done.")
    return preconditioner


def _random_lanczos_vector(backend: SolverBackend, n: int):
    """Create the initial random vector for Lanczos on the active backend."""
    if backend.label == "gpu":
        cp = backend.cupy_module
        return cp.random.randn(n, 1, dtype=cp.float32)
    return np.random.randn(n, 1)


def _compute_initial_density(
    problem: PreparedSystem,
    elem,
    n_elements: int,
    backend: SolverBackend,
    A,
    preconditioner,
    build_hpot: bool = True,
):
    """Build the initial density and diagonal ionic potentials.

    Output:
        ``(rho0, hpot0, ppot, diag_info)`` matching the legacy main script.
    """
    density_method = problem.input_data.density_method
    if density_method == "sad":
        print("Using SAD method...")
        return backend.pseudo_diag(
            problem.domain,
            problem.input_data.atoms,
            elem,
            n_elements,
            return_info=True,
            build_hpot=build_hpot,
        )

    if density_method == "sad_ml_grid":
        print(f"Using SAD density on ML grid from: {problem.input_data.grid_npy_path}")
        from .V_ion.pseudoDiag_MLgrid import pseudoDiag_MLgrid

        return pseudoDiag_MLgrid(
            problem.domain,
            problem.input_data.atoms,
            elem,
            n_elements,
            return_info=True,
            build_hpot=build_hpot,
        )

    print(f"Using ML grid/density file: {problem.input_data.ml_file_path}")
    from .V_ion.pseudoDiag_ML4Den_poisson import pseudoDiag_ML4Den

    return pseudoDiag_ML4Den(
        problem.domain,
        problem.input_data.atoms,
        elem,
        n_elements,
        problem.input_data.ml_file_path,
        A,
        problem.settings.cg_prec,
        preconditioner,
        return_info=True,
        build_hpot=build_hpot,
    )


def _compute_nonlocal_potential(problem: PreparedSystem, elem, n_elements: int, backend: SolverBackend):
    """Build the nonlocal pseudopotential operator for the chosen density mode."""
    density_method = problem.input_data.density_method
    if density_method == "sad":
        print("Using SAD method...")
        return backend.pseudo_nl(problem.domain, problem.input_data.atoms, elem, n_elements)

    if density_method == "sad_ml_grid":
        print(f"Using SAD density on ML grid from: {problem.input_data.grid_npy_path}")
        from .V_ion.pseudoNL_original_MLgrid import pseudoNL_MLgrid

        return pseudoNL_MLgrid(problem.domain, problem.input_data.atoms, elem, n_elements)

    print(f"Using ML grid/density file: {problem.input_data.ml_file_path}")
    return pseudoNL_ML4Den(problem.domain, problem.input_data.atoms, elem, n_elements)


def compute_energy_components(
    rho,
    hpot,
    hpot0,
    xc_potential,
    exc: float,
    lam_host,
    occup_host,
    nev: int,
    e_nuc0: float,
    h: float,
    backend: SolverBackend,
) -> EnergyComponents:
    """Package the current energy breakdown into an ``EnergyComponents`` object."""
    esum = float(np.sum(lam_host[:nev] * occup_host[:nev]))
    eigen_sum_ry = 4.0 * esum
    xp = backend.array_module
    hartree_ry = float(backend.to_host_scalar(xp.sum(rho * (hpot + hpot0)) * h**3))
    vxc_sum_ry = float(backend.to_host_scalar(xp.sum(rho * xc_potential) * h**3))
    xc_ry = float(backend.to_host_scalar(exc))
    total_ry = eigen_sum_ry - hartree_ry + xc_ry - vxc_sum_ry + float(e_nuc0)
    return EnergyComponents(
        eigen_sum_ry=eigen_sum_ry,
        hartree_ry=hartree_ry,
        xc_ry=xc_ry,
        ion_ry=float(e_nuc0),
        total_ry=total_ry,
    )


def _parsec_charge_weighted_sre(
    delta_potential,
    rho,
    h: float,
    backend: SolverBackend,
    nrep: float = 1.0,
) -> float:
    """Return PARSEC's default charge-weighted SRE for one spin channel.

    The refactored Python path is spin-unpolarized and does not use symmetry
    representations, so ``nrep`` is one unless that support is added later.
    """
    xp = backend.array_module
    delta = xp.asarray(delta_potential)
    rho_arr = xp.asarray(rho)
    hcub = h**3

    electron_count = xp.sum(rho_arr) * hcub * nrep
    electron_count_host = float(backend.to_host_scalar(electron_count))
    if electron_count_host <= 0.0:
        return float("inf")

    weighted_sre_sq = xp.sum(rho_arr * delta * delta) * hcub * nrep / electron_count
    weighted_sre_sq = xp.maximum(weighted_sre_sq, 0.0)
    return float(backend.to_host_scalar(xp.sqrt(weighted_sre_sq)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite relative value when both norms are zero."""
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _backend_norm(backend: SolverBackend, value) -> float:
    """Compute a vector norm and convert the result to a Python float."""
    return float(backend.to_host_scalar(backend.array_module.linalg.norm(value)))


def _synchronize_backend(backend: SolverBackend) -> None:
    """Synchronize GPU work before taking wall-clock timings."""
    if backend.label == "gpu":
        backend.cupy_module.cuda.get_current_stream().synchronize()


def run_rsdft_calculation(
    problem: PreparedSystem,
    elem,
    n_elements: int,
    backend: SolverBackend,
    run_start_time: float | None = None,
) -> SCFResult:
    """Execute the real-space DFT solve for one prepared system.

    Input:
        problem: fully prepared run data from ``rsdft_setup.prepare_system``.
        elem, n_elements: element lookup table and its size.
        backend: CPU/GPU implementation bundle selected earlier.

    Output:
        ``SCFResult`` containing the final density, potentials, eigenpairs,
        convergence flag, and other end-of-run quantities.
    """
    calculation_start_time = time.perf_counter()
    print(" ")
    print("******************")
    print("     OUTPUT       ")
    print("******************")
    print(" ")
    print(" Working.....constructing Laplacian matrix...")

    timings = TimingRecorder()
    h = problem.domain["h"]
    nx, ny, nz = problem.domain["nx"], problem.domain["ny"], problem.domain["nz"]

    # Stage 1: build the finite-difference Laplacian on the chosen domain.
    start_time = time.perf_counter()
    A, gpu_laplacian = _build_laplacians(problem, backend)
    _synchronize_backend(backend)
    laplacian_time = time.perf_counter() - start_time
    print(laplacian_time)
    timings.add("Laplacian construction", laplacian_time)

    if backend.label == "gpu":
        n = gpu_laplacian.shape[0]
    else:
        n = A.shape[0]
    Hpot = np.zeros(n)
    pot = Hpot.copy()
    err = 10.0 + problem.settings.tol

    # Stage 2: set up ionic terms and the initial charge density.
    print(" Working.....setting up ionic potential...")
    start_time = time.perf_counter()
    e_nuc0 = nuclear(problem.domain, problem.input_data.atoms, elem, n_elements)
    enuc_time = time.perf_counter() - start_time
    print(" Enuc time", enuc_time)
    timings.add("Ion-ion repulsion setup", enuc_time)

    preconditioner = _build_preconditioner(
        A,
        problem.settings.cg_prec,
        backend,
        problem.input_data.density_method,
    )

    print(" Working.....setting up diagonal part of ionic potential...")
    start_time = time.perf_counter()
    rho0, hpot0, ppot, diag_info = _compute_initial_density(
        problem,
        elem,
        n_elements,
        backend,
        A,
        preconditioner,
        build_hpot=False,
    )
    rho0 = np.asarray(backend.to_numpy_array(rho0)).reshape(-1)
    hpot0 = np.asarray(backend.to_numpy_array(hpot0)).reshape(-1)
    ppot = np.asarray(backend.to_numpy_array(ppot)).reshape(-1)
    _synchronize_backend(backend)
    pseudo_diag_time = time.perf_counter() - start_time
    print(" pseudoDiag time: ", pseudo_diag_time)
    timings.add("Diagonal ionic potential setup", pseudo_diag_time)
    write_diag_info(problem.paths.output_file, diag_info)

    if problem.input_data.z_charge != 0:
        scaling_factor = problem.ztest / problem.zelec
        rho0 *= scaling_factor

    print(" Working.....setting up Hartree reference from normalized density...")
    start_time = time.perf_counter()
    active_laplacian = gpu_laplacian if backend.label == "gpu" else A
    initial_hartree_backend, initial_hartree_iterations = solve_hartree_from_grid_density(
        active_laplacian,
        backend.array_module.asarray(rho0).reshape(-1),
        h,
        backend,
    )
    hpot0 = np.asarray(backend.to_numpy_array(initial_hartree_backend)).reshape(-1)
    Hpot = np.zeros_like(hpot0)
    _synchronize_backend(backend)
    initial_hartree_time = time.perf_counter() - start_time
    print(" initial Hartree reference time: ", initial_hartree_time)
    print(f" Initial Hartree Poisson iterations: {initial_hartree_iterations}")
    timings.add("Initial Hartree reference setup", initial_hartree_time)
    with open(problem.paths.output_file, "a", encoding="utf-8") as fid:
        fid.write(
            " Initial Hartree reference potential: solved once from normalized rho0 "
            "with the active Poisson solver.\n"
        )
        fid.write(f" Initial Hartree Poisson iterations: {initial_hartree_iterations}\n")

    save_density_variants(rho0, problem.domain, problem.paths.initial_density_base)

    rhoxc = np.transpose(rho0) / (h**3)
    hpsum0 = float(np.sum(rho0 * hpot0))
    hpsum0_ev = hpsum0 * 13.605698066

    print(" Working.....setting up nonlocal part of ionic potential...")
    start_time = time.perf_counter()
    vnl = _compute_nonlocal_potential(problem, elem, n_elements, backend)
    _synchronize_backend(backend)
    pseudo_nl_time = time.perf_counter() - start_time
    print(" pseudoNL time: ", pseudo_nl_time)
    timings.add("Nonlocal ionic potential setup", pseudo_nl_time)

    print(" Working.....setting up exchange and correlation potentials...")
    start_time = time.perf_counter()
    if backend.label == "gpu":
        cp = backend.cupy_module
        XCpot_backend, exc = backend.xc(problem.domain, cp.asarray(rhoxc, dtype=cp.float32))
    else:
        XCpot_backend, exc = backend.xc(problem.domain, rhoxc)
    _synchronize_backend(backend)
    exc_time = time.perf_counter() - start_time
    print(" exc time: ", exc_time)
    timings.add("Exchange-correlation setup", exc_time)

    write_initial_density_diagnostics(problem.paths.output_file, rhoxc, hpsum0_ev, exc)
    setup_timing_entries = list(timings.entries)
    timings.flush(problem.paths.output_file)

    nelec = nelectrons(problem.input_data.atoms, elem, n_elements)
    if problem.input_data.z_charge != 0:
        nelec -= problem.input_data.z_charge

    if backend.label == "gpu":
        cp = backend.cupy_module
        rho0_backend = cp.asarray(rho0, dtype=cp.float32)
        hpot0_backend = cp.asarray(hpot0, dtype=cp.float32)
        ppot_backend = cp.asarray(ppot, dtype=cp.float32)
        Hpot_backend = cp.asarray(Hpot, dtype=cp.float32)
        XCpot_backend = cp.asarray(XCpot_backend, dtype=cp.float32).reshape(-1)
        pot_backend = ppot_backend + hpot0_backend + 0.5 * XCpot_backend
        rho_backend = rho0_backend / h**3
    else:
        rho0_backend = rho0
        hpot0_backend = hpot0
        ppot_backend = ppot
        Hpot_backend = Hpot
        XCpot_backend = np.asarray(XCpot_backend).reshape(-1)
        pot_backend = ppot_backend + hpot0_backend + 0.5 * XCpot_backend
        rho_backend = rhoxc

    backend.reset_mixer()

    with open(problem.paths.output_file, "a", encoding="utf-8") as fid:
        fid.write("\n----------------------------------\n\n")
        fid.write("SCF convergence metric: PARSEC charge-weighted SRE\n")
        fid.write("SRE = sqrt(h^3 * sum_i rho_i * (V_new_i - V_old_i)^2 / N_e)\n\n")

    half_a_plus_vnl = None
    gpu_hamiltonian = None
    rho_rhs_scale = 4 * np.pi / h**3
    if backend.label == "gpu":
        from .Eigensolvers.gpu_linear_operator import ShiftedHamiltonianOperator, to_gpu_matrix

        gpu_base_hamiltonian = gpu_laplacian * np.float32(0.5) + to_gpu_matrix(vnl)
        gpu_hamiltonian = ShiftedHamiltonianOperator(gpu_base_hamiltonian, pot_backend)
        pot_backend = gpu_hamiltonian.diagonal
    else:
        half_a_plus_vnl = 0.5 * A + vnl

    if problem.settings.adaptive_scheme != 0 and sum(problem.input_data.n_atom) <= 2:
        degree_modifier = 0.75
        m_modifier = 0.95
    else:
        degree_modifier = 1.0
        m_modifier = 1.0

    W = []
    lam = []
    occup = []
    lam_host = np.array([])
    occup_host = np.array([])
    n_atoms = sum(atom["coord"].shape[0] for atom in problem.input_data.atoms)
    iterations = 0
    previous_rho_backend = backend.array_module.asarray(rho_backend).copy()
    previous_total_energy_ry: float | None = None
    scf_diag_time = 0.0
    scf_density_time = 0.0
    scf_hartree_time = 0.0
    scf_xc_time = 0.0
    scf_other_time = 0.0
    scf_loop_start_time = time.perf_counter()

    # Stage 3: self-consistent-field loop.
    while err > problem.settings.tol and iterations < problem.settings.maxits:
        iterations += 1
        iteration_start_time = time.perf_counter()
        print(f"  Working ... SCF iter # {iterations} ... ")

        if backend.label == "gpu":
            gpu_hamiltonian.update_diagonal(pot_backend)
            B = gpu_hamiltonian
        else:
            B = half_a_plus_vnl + sp.diags(pot_backend, 0, shape=(n, n))
        start_time = time.perf_counter()

        # Solve the current Hamiltonian using the selected eigensolver path.
        if problem.settings.diagmeth == 1 or (iterations == 1 and problem.settings.diagmeth == 0):
            print("Calling lanczos...")
            diag_label = f"lanczos_{backend.label} (diagmeth={problem.settings.diagmeth})"
            W, lam = backend.lanczos(
                B,
                problem.nev + 15,
                _random_lanczos_vector(backend, n),
                _scaled_lanczos_steps(problem.nev, m_modifier),
                1e-5,
            )
        elif iterations == 1 and problem.settings.diagmeth == 2:
            print("Calling chsubsp...")
            diag_label = f"chsubsp_{backend.label} (diagmeth={problem.settings.diagmeth})"
            W, lam = backend.chsubsp(_scaled_poldeg(problem.settings.poldeg, degree_modifier), problem.nev + 15, B)
        elif iterations == 1 and problem.settings.diagmeth == 3:
            print("Calling first_filt...")
            diag_label = f"first_filt_{backend.label} (diagmeth={problem.settings.diagmeth})"
            W, lam = backend.first_filt(problem.nev + 15, B, _scaled_poldeg(problem.settings.poldeg, degree_modifier))
        else:
            print("Calling chebsf...")
            diag_label = f"chebsf_{backend.label} (diagmeth={problem.settings.diagmeth})"
            W, lam = backend.chefsi1(
                W,
                lam,
                _scaled_poldeg(problem.settings.poldeg, degree_modifier),
                problem.nev,
                B,
            )

        _synchronize_backend(backend)
        diag_time = time.perf_counter() - start_time
        scf_diag_time += diag_time

        density_start_time = time.perf_counter()
        _, occup = backend.occupations(lam[: problem.nev], problem.settings.fermi_temp, nelec, 1e-6)
        lam_host = np.asarray(backend.to_numpy_array(lam)).reshape(-1)
        occup_host = np.asarray(backend.to_numpy_array(occup)).reshape(-1)

        # Update the density from the occupied states, then solve Poisson/XC.
        rho_grid_backend = (W[:, : problem.nev] ** 2) @ (2 * occup)
        if backend.label == "gpu":
            rho_grid_backend = backend.array_module.asarray(rho_grid_backend).reshape(-1)
            hrhs = rho_rhs_scale * (rho_grid_backend - rho0_backend)
            rho_backend = rho_grid_backend / h**3
        else:
            rho_grid_backend = np.asarray(backend.to_numpy_array(rho_grid_backend)).reshape(-1)
            hrhs = rho_rhs_scale * (rho_grid_backend - rho0_backend)
            rho_backend = rho_grid_backend / h**3

        _synchronize_backend(backend)
        density_time = time.perf_counter() - density_start_time
        scf_density_time += density_time

        start_time = time.perf_counter()
        hart_tol = 1e-5
        if backend.label == "gpu" and problem.settings.cg_prec:
            hart_prec_label = "gpu-no-prec (ILU unavailable on GPU path)"
        else:
            hart_prec_label = "precLU" if problem.settings.cg_prec else "no prec"

        if problem.settings.cg_prec:
            print(f"with CG_prec (Hartree CG tol = {hart_tol:.1e})")
            if backend.label == "cpu":
                Hpot_backend, _ = pcg_cpu(A, hrhs, Hpot_backend, 200, hart_tol, preconditioner, "precLU")
            else:
                Hpot_backend, _ = backend.pcg(gpu_laplacian, hrhs, Hpot_backend, 200, hart_tol, preconditioner, "precLU")
        else:
            print(f"no CG_prec (Hartree CG tol = {hart_tol:.1e})")
            if backend.label == "gpu":
                Hpot_backend, _ = backend.pcg(gpu_laplacian, hrhs, Hpot_backend, 200, hart_tol)
            else:
                Hpot_backend, _ = backend.pcg(A, hrhs, Hpot_backend, 200, hart_tol)
        _synchronize_backend(backend)
        hart_time = time.perf_counter() - start_time
        scf_hartree_time += hart_time

        xc_start_time = time.perf_counter()
        XCpot_backend, exc = backend.xc(problem.domain, rho_backend, problem.paths.output_file)
        if backend.label == "gpu":
            XCpot_backend = backend.array_module.asarray(XCpot_backend).reshape(-1)
            pot_new_backend = ppot_backend + 0.5 * XCpot_backend + Hpot_backend + hpot0_backend
            delta_backend = pot_new_backend - pot_backend
        else:
            XCpot_backend = np.asarray(XCpot_backend).reshape(-1)
            pot_new_backend = ppot_backend + 0.5 * XCpot_backend + Hpot_backend + hpot0_backend
            delta_backend = pot_new_backend - pot_backend

        _synchronize_backend(backend)
        xc_time = time.perf_counter() - xc_start_time
        scf_xc_time += xc_time

        potential_abs_norm = _backend_norm(backend, delta_backend)
        potential_new_norm = _backend_norm(backend, pot_new_backend)
        potential_rms = potential_abs_norm / np.sqrt(delta_backend.size)
        potential_relative = _safe_ratio(potential_abs_norm, potential_new_norm)

        density_delta_backend = rho_backend - previous_rho_backend
        density_abs_norm = _backend_norm(backend, density_delta_backend)
        density_norm = _backend_norm(backend, rho_backend)
        density_relative = _safe_ratio(density_abs_norm, density_norm)

        err_new = _parsec_charge_weighted_sre(delta_backend, rho_backend, h, backend)

        # Mild adaptive tuning of polynomial degree / Lanczos subspace size.
        if problem.settings.adaptive_scheme == 0 or err_new > 1 or err_new > 2 * err:
            degree_modifier = 1.0
            m_modifier = 1.0
        elif err_new > err:
            degree_modifier = min(1.1, degree_modifier + 0.2)
            m_modifier = min(1.1, degree_modifier + 0.05)
        elif 3 * err_new < err:
            degree_modifier = max(0.5, degree_modifier - 0.1)
            m_modifier = max(0.9, degree_modifier - 0.025)

        err = err_new
        iteration_energies = compute_energy_components(
            rho_backend,
            Hpot_backend,
            hpot0_backend,
            XCpot_backend,
            exc,
            lam_host,
            occup_host,
            problem.nev,
            e_nuc0,
            h,
            backend,
        )
        energy_change_ry = None
        if previous_total_energy_ry is not None:
            energy_change_ry = abs(iteration_energies.total_ry - previous_total_energy_ry)
        scf_diagnostics = SCFDiagnostics(
            potential_relative=potential_relative,
            potential_norm=potential_new_norm,
            potential_abs_norm=potential_abs_norm,
            potential_rms=potential_rms,
            density_relative=density_relative,
            energy_change_ry=energy_change_ry,
        )
        write_scf_iteration(
            problem.paths.output_file,
            iterations,
            diag_label,
            diag_time,
            problem.nev,
            lam_host,
            occup_host,
            hart_tol,
            hart_prec_label,
            hart_time,
            err,
            scf_diagnostics,
            iteration_energies,
            n_atoms,
        )
        print(f"   ... PARSEC charge-weighted SRE = {err:10.2e}\n")

        previous_rho_backend = backend.array_module.asarray(rho_backend).copy()
        previous_total_energy_ry = iteration_energies.total_ry

        pot_backend, _ = backend.mixer(pot_backend, delta_backend)
        _synchronize_backend(backend)
        iteration_time = time.perf_counter() - iteration_start_time
        scf_other_time += max(0.0, iteration_time - diag_time - density_time - hart_time - xc_time)

    # Stage 4: final reporting and optional wavefunction export.
    scf_loop_time = time.perf_counter() - scf_loop_start_time
    final_reporting_start_time = time.perf_counter()
    print("SCF loop completed.")
    converged = err <= problem.settings.tol
    if converged:
        print("          ")
        print("**************************")
        print(" CONVERGED SOLUTION!! ")
        print("**************************")
        print("         ")
    else:
        print("          ")
        print("**************************")
        print(" !!THE SYSTEM DID NOT CONVERGE!!")
        print("          ")
        print(" !!THESE ARE THE VALUES FROM THE LAST ITERATION!!")
        print("**************************")
        print("         ")

    rho_host = np.asarray(backend.to_numpy_array(rho_backend)).reshape(-1)
    Hpot_host = np.asarray(backend.to_numpy_array(Hpot_backend)).reshape(-1)
    XCpot_host = np.asarray(backend.to_numpy_array(XCpot_backend)).reshape(-1)
    pot_host = np.asarray(backend.to_numpy_array(pot_backend)).reshape(-1)
    hpot0_host = np.asarray(backend.to_numpy_array(hpot0_backend)).reshape(-1)

    save_density_variants(rho_host * (problem.domain["h"] ** 3), problem.domain, problem.paths.converged_density_base)
    print_eigenvalues_to_console(problem.nev, lam_host, occup_host)

    final_energies = compute_energy_components(
        rho_backend,
        Hpot_backend,
        hpot0_backend,
        XCpot_backend,
        exc,
        lam_host,
        occup_host,
        problem.nev,
        e_nuc0,
        h,
        backend,
    )
    write_total_energy_summary(problem.paths.output_file, final_energies, n_atoms)
    print_total_energy_summary(final_energies, n_atoms)

    if backend.label == "gpu" and gpu_hamiltonian is not None:
        density_potential_backend = ppot_backend + 0.5 * XCpot_backend + Hpot_backend + hpot0_backend
        gpu_hamiltonian.update_diagonal(density_potential_backend)
        stationarity_diagnostic = compute_orbital_stationarity_diagnostic(
            gpu_hamiltonian,
            W,
            occup,
            problem.nev,
            backend,
        )
        write_orbital_stationarity_diagnostic(problem.paths.output_file, stationarity_diagnostic)
        if stationarity_diagnostic.get("available"):
            print(
                " Orbital stationarity residual RMS = "
                f"{stationarity_diagnostic['residual_rms']:.3e}, "
                "relative Frobenius = "
                f"{stationarity_diagnostic['residual_relative_frob']:.3e}"
            )
        else:
            print(
                " Orbital stationarity diagnostic skipped: "
                f"{stationarity_diagnostic.get('reason', 'unknown reason')}"
            )

    backend.reset_mixer()
    if problem.settings.save_wfn:
        save_wavefunction(
            problem.paths.wfn_file,
            problem.domain,
            pot_host,
            rho_host,
            W,
            problem.nev,
            problem.n_types,
            problem.input_data.atoms,
            backend.to_numpy_array,
        )

    _synchronize_backend(backend)
    final_reporting_time = time.perf_counter() - final_reporting_start_time
    calculation_wall_time = time.perf_counter() - calculation_start_time
    driver_wall_time = time.perf_counter() - run_start_time if run_start_time is not None else None
    write_timing_analysis(
        problem.paths.output_file,
        setup_timing_entries,
        iterations,
        scf_diag_time,
        scf_density_time,
        scf_hartree_time,
        scf_xc_time,
        scf_other_time,
        scf_loop_time,
        final_reporting_time,
        calculation_wall_time,
        driver_wall_time,
    )
    print_timing_analysis(
        setup_timing_entries,
        iterations,
        scf_diag_time,
        scf_density_time,
        scf_hartree_time,
        scf_xc_time,
        scf_other_time,
        scf_loop_time,
        final_reporting_time,
        calculation_wall_time,
        driver_wall_time,
    )

    return SCFResult(
        rho=rho_host,
        hpot=Hpot_host,
        xc_potential=XCpot_host,
        exc=float(exc),
        potential=pot_host,
        wavefunctions=W,
        eigenvalues=lam_host,
        occupations=occup_host,
        iterations=iterations,
        error=err,
        converged=converged,
        e_nuc0=float(e_nuc0),
        hpot0=hpot0_host,
        n_atoms=n_atoms,
    )
