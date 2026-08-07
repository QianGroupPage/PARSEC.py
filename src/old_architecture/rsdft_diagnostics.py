"""Post-SCF numerical diagnostics."""

from __future__ import annotations

import math
from typing import Any

from .rsdft_models import RY_EV


def _host_scalar(value: Any, backend) -> float:
    return float(backend.to_host_scalar(value))


def _projected_residual_metrics(hamiltonian, psi_block, backend) -> dict[str, Any]:
    """Return complement-space residual metrics for one orbital block."""
    xp = backend.array_module
    n_block = int(psi_block.shape[1])

    hpsi = hamiltonian @ psi_block
    overlap = psi_block.conj().T @ psi_block
    gram = psi_block.conj().T @ hpsi
    eye = xp.eye(n_block, dtype=overlap.dtype)

    try:
        coeff = xp.linalg.solve(overlap, gram)
        projection = "generalized"
    except Exception:
        coeff = gram
        projection = "orthonormal_fallback"

    residual = hpsi - psi_block @ coeff
    residual_col_norms = xp.linalg.norm(residual, axis=0)
    residual_frob = _host_scalar(xp.linalg.norm(residual), backend)
    hpsi_frob = _host_scalar(xp.linalg.norm(hpsi), backend)
    overlap_error = overlap - eye

    return {
        "projection": projection,
        "hpsi": hpsi,
        "residual": residual,
        "residual_col_norms": residual_col_norms,
        "residual_rms": math.sqrt(
            max(_host_scalar(xp.mean(residual_col_norms**2), backend), 0.0)
        ),
        "residual_max": _host_scalar(xp.max(residual_col_norms), backend),
        "residual_frob": residual_frob,
        "residual_relative_frob": residual_frob / max(hpsi_frob, 1.0e-30),
        "overlap_error_frob": _host_scalar(xp.linalg.norm(overlap_error), backend),
        "overlap_error_max": _host_scalar(xp.max(xp.abs(overlap_error)), backend),
    }


def compute_orbital_stationarity_diagnostic(
    hamiltonian,
    wavefunctions,
    occupations,
    nev: int,
    backend,
    occ_tol: float = 1.0e-8,
) -> dict[str, Any]:
    """Measure the final occupied-subspace gradient without forming P explicitly.

    The primary residual is

        R_occ = H Psi_occ - Psi_occ (Psi_occ^T Psi_occ)^-1 Psi_occ^T H Psi_occ,

    where Psi_occ includes only orbitals with occupation above ``occ_tol``.
    For orthonormal orbitals this is (I - P_occ) H P_occ in column form.
    The diagnostic also reports a secondary residual over all states that have
    occupations available, so extra unoccupied Ritz-vector quality is visible
    without confusing it with occupied-subspace stationarity.
    """
    xp = backend.array_module

    if wavefunctions is None or len(wavefunctions) == 0:
        return {"available": False, "reason": "no wavefunctions were available"}

    psi = xp.asarray(wavefunctions)
    if psi.ndim != 2 or psi.shape[1] == 0:
        return {"available": False, "reason": "wavefunctions were not a 2D matrix"}

    occup = xp.asarray(occupations).reshape(-1)
    n_reported = min(int(nev), int(psi.shape[1]), int(occup.size))
    if n_reported <= 0:
        return {"available": False, "reason": "no states with occupations were available"}

    occup_reported = occup[:n_reported]
    occup_reported_host = backend.to_numpy_array(occup_reported)
    active_indices = [
        index
        for index, occ_value in enumerate(occup_reported_host)
        if float(occ_value) > occ_tol
    ]
    if not active_indices:
        return {"available": False, "reason": "no states had occupation above the threshold"}

    active_index_set = set(active_indices)
    inactive_reported_indices = [
        index for index in range(n_reported) if index not in active_index_set
    ]
    extra_indices = list(range(n_reported, int(psi.shape[1])))
    virtual_indices = inactive_reported_indices + extra_indices

    psi_occ = psi[:, active_indices]
    occup_occ = occup[active_indices]
    occ_metrics = _projected_residual_metrics(hamiltonian, psi_occ, backend)

    reported_metrics = _projected_residual_metrics(
        hamiltonian,
        psi[:, :n_reported],
        backend,
    )

    spin_occup = 2.0 * occup_occ
    positive_weight = xp.maximum(spin_occup, 0.0)
    weight_sum = _host_scalar(xp.sum(positive_weight), backend)
    if weight_sum > occ_tol:
        weighted_residual_rms = math.sqrt(
            max(
                _host_scalar(
                    xp.sum(positive_weight * occ_metrics["residual_col_norms"] ** 2),
                    backend,
                )
                / weight_sum,
                0.0,
            )
        )
    else:
        weighted_residual_rms = float("nan")

    virtual_count = len(virtual_indices)
    ov_frob = None
    ov_max = None
    if virtual_count:
        psi_virt = psi[:, virtual_indices]
        h_vo = psi_virt.conj().T @ occ_metrics["hpsi"]
        ov_frob = _host_scalar(xp.linalg.norm(h_vo), backend)
        ov_max = _host_scalar(xp.max(xp.abs(h_vo)), backend)

    return {
        "available": True,
        "projection": occ_metrics["projection"],
        "reported_state_count": n_reported,
        "active_occupied_count": len(active_indices),
        "occupied_count": len(active_indices),
        "occupation_active_count": len(active_indices),
        "occupation_min": _host_scalar(xp.min(occup_occ), backend),
        "occupation_max": _host_scalar(xp.max(occup_occ), backend),
        "reported_occupation_min": _host_scalar(xp.min(occup_reported), backend),
        "reported_occupation_max": _host_scalar(xp.max(occup_reported), backend),
        "virtual_count": virtual_count,
        "residual_rms": occ_metrics["residual_rms"],
        "residual_max": occ_metrics["residual_max"],
        "residual_frob": occ_metrics["residual_frob"],
        "residual_relative_frob": occ_metrics["residual_relative_frob"],
        "weighted_residual_rms": weighted_residual_rms,
        "commutator_frob_estimate": math.sqrt(2.0) * occ_metrics["residual_frob"],
        "available_ov_frob": ov_frob,
        "available_ov_max": ov_max,
        "overlap_error_frob": occ_metrics["overlap_error_frob"],
        "overlap_error_max": occ_metrics["overlap_error_max"],
        "reported_projection": reported_metrics["projection"],
        "reported_residual_rms": reported_metrics["residual_rms"],
        "reported_residual_max": reported_metrics["residual_max"],
        "reported_residual_relative_frob": reported_metrics["residual_relative_frob"],
        "reported_commutator_frob_estimate": math.sqrt(2.0) * reported_metrics["residual_frob"],
        "reported_overlap_error_frob": reported_metrics["overlap_error_frob"],
        "reported_overlap_error_max": reported_metrics["overlap_error_max"],
    }


def write_hartree_consistency_diagnostic(output_file: str, diagnostic: dict[str, Any]) -> None:
    """Append final Hartree/Poisson consistency checks to the run log."""
    with open(output_file, "a", encoding="utf-8") as fid:
        fid.write("\n\n Hartree consistency diagnostic\n")
        fid.write(" --------------------------------------------------\n")

        if not diagnostic.get("available"):
            fid.write(f" Diagnostic skipped: {diagnostic.get('reason', 'unknown reason')}\n")
            fid.write(" --------------------------------------------------\n")
            return

        fid.write(" Carried Hartree potential is Hpot + hpot0 from the SCF split.\n")
        fid.write(" Recomputed Hartree potential solves Poisson once from the final density.\n")
        fid.write(f" Poisson recompute iterations:       {diagnostic['recomputed_iterations']:10d}\n")
        fid.write(f" Poisson residual norm carried:      {diagnostic['carried_residual_norm']:10.5e}\n")
        fid.write(f" Poisson residual rel carried:       {diagnostic['carried_residual_relative']:10.3e}\n")
        fid.write(f" Poisson residual RMS carried:       {diagnostic['carried_residual_rms']:10.5e}\n")
        fid.write(f" Poisson residual norm recomputed:   {diagnostic['recomputed_residual_norm']:10.5e}\n")
        fid.write(f" Poisson residual rel recomputed:    {diagnostic['recomputed_residual_relative']:10.3e}\n")
        fid.write(f" Poisson residual RMS recomputed:    {diagnostic['recomputed_residual_rms']:10.5e}\n")
        fid.write(f" ||V_H carried - recomputed||:       {diagnostic['hartree_delta_norm']:10.5e}\n")
        fid.write(f" Relative Hartree delta norm:        {diagnostic['hartree_delta_relative']:10.3e}\n")
        fid.write(f" Hartree delta RMS:                  {diagnostic['hartree_delta_rms']:10.5e}\n")
        fid.write(f" Hartree delta max abs:              {diagnostic['hartree_delta_max_abs']:10.5e}\n")
        fid.write(f" Hartree delta mean:                 {diagnostic['hartree_delta_mean']:10.5e}\n")
        fid.write(f" Hartree delta centered RMS:         {diagnostic['hartree_delta_centered_rms']:10.5e}\n")
        fid.write(
            f" Hartree energy carried:             {diagnostic['carried_hartree_ry'] * RY_EV:10.5f}  eV   = "
            f"{diagnostic['carried_hartree_ry']:10.5f}  Ry\n"
        )
        fid.write(
            f" Hartree energy recomputed:          {diagnostic['recomputed_hartree_ry'] * RY_EV:10.5f}  eV   = "
            f"{diagnostic['recomputed_hartree_ry']:10.5f}  Ry\n"
        )
        fid.write(
            f" Hartree energy delta:               {diagnostic['hartree_energy_delta_ry'] * RY_EV:10.5e}  eV   = "
            f"{diagnostic['hartree_energy_delta_ry']:10.5e}  Ry\n"
        )
        fid.write(
            " Total energy with recomputed Hartree uses the same eigenvalues/XC, "
            "replacing only the Hartree term.\n"
        )
        fid.write(
            f" Total energy carried:               {diagnostic['carried_total_ry'] * RY_EV:10.5f}  eV   = "
            f"{diagnostic['carried_total_ry']:10.5f}  Ry\n"
        )
        fid.write(
            f" Total energy recomputed-Hartree:    {diagnostic['recomputed_total_ry'] * RY_EV:10.5f}  eV   = "
            f"{diagnostic['recomputed_total_ry']:10.5f}  Ry\n"
        )
        fid.write(
            f" Total energy delta:                 {diagnostic['total_energy_delta_ry'] * RY_EV:10.5e}  eV   = "
            f"{diagnostic['total_energy_delta_ry']:10.5e}  Ry\n"
        )
        fid.write(" --------------------------------------------------\n")


def write_orbital_stationarity_diagnostic(output_file: str, diagnostic: dict[str, Any]) -> None:
    """Append the orbital stationarity diagnostic to the run log."""
    with open(output_file, "a", encoding="utf-8") as fid:
        fid.write("\n\n Orbital stationarity diagnostic\n")
        fid.write(" --------------------------------------------------\n")
        fid.write(" Hamiltonian source: rebuilt from final density-derived potential\n")

        if not diagnostic.get("available"):
            fid.write(f" Diagnostic skipped: {diagnostic.get('reason', 'unknown reason')}\n")
            fid.write(" --------------------------------------------------\n")
            return

        fid.write(f" Active projection model:       {diagnostic['projection']}\n")
        fid.write(f" States with occupations:       {diagnostic['reported_state_count']:10d}\n")
        fid.write(f" Active occupied states:        {diagnostic['active_occupied_count']:10d}\n")
        fid.write(
            " Active occupation range:       "
            f"{diagnostic['occupation_min']:10.3e}  {diagnostic['occupation_max']:10.3e}\n"
        )
        fid.write(
            " Reported occupation range:     "
            f"{diagnostic['reported_occupation_min']:10.3e}  "
            f"{diagnostic['reported_occupation_max']:10.3e}\n"
        )
        fid.write(f" Available virtual states:      {diagnostic['virtual_count']:10d}\n")
        fid.write(f" Active residual RMS ||R_i||:   {diagnostic['residual_rms']:10.3e}\n")
        fid.write(f" Active residual max ||R_i||:   {diagnostic['residual_max']:10.3e}\n")
        fid.write(f" Active relative Frobenius:     {diagnostic['residual_relative_frob']:10.3e}\n")
        fid.write(f" Active occupation-weighted RMS:{diagnostic['weighted_residual_rms']:10.3e}\n")
        fid.write(f" Estimated active ||[H,P]||_F:  {diagnostic['commutator_frob_estimate']:10.3e}\n")

        if diagnostic["available_ov_frob"] is not None:
            fid.write(f" Physical OV block ||H_vo||_F:  {diagnostic['available_ov_frob']:10.3e}\n")
            fid.write(f" Physical OV block max abs:     {diagnostic['available_ov_max']:10.3e}\n")
        else:
            fid.write(" Physical OV block:             not available; no virtual states in W\n")

        fid.write(f" Active overlap error ||S-I||:  {diagnostic['overlap_error_frob']:10.3e}\n")
        fid.write(f" Active overlap max abs error:  {diagnostic['overlap_error_max']:10.3e}\n")
        fid.write(" -- all reported states, including unoccupied Ritz vectors --\n")
        fid.write(f" Reported residual RMS ||R_i||: {diagnostic['reported_residual_rms']:10.3e}\n")
        fid.write(f" Reported residual max ||R_i||: {diagnostic['reported_residual_max']:10.3e}\n")
        fid.write(
            f" Reported relative Frobenius:   {diagnostic['reported_residual_relative_frob']:10.3e}\n"
        )
        fid.write(
            f" Estimated reported ||[H,P]||:  {diagnostic['reported_commutator_frob_estimate']:10.3e}\n"
        )
        fid.write(
            f" Reported overlap error ||S-I||:{diagnostic['reported_overlap_error_frob']:10.3e}\n"
        )
        fid.write(
            f" Reported overlap max abs error:{diagnostic['reported_overlap_error_max']:10.3e}\n"
        )
        fid.write(" --------------------------------------------------\n")
