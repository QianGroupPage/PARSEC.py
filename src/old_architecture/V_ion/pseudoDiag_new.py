import logging

import numpy as np

from .preProcess import preProcess
from .splineData import splineData
from ..Splines.fspline import fspline


logging.basicConfig(
    level=logging.ERROR,
    filename="error_log.txt",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class MexFileDiscrepancyError(Exception):
    """
    Custom exception class.
    """

    def __init__(self, message, identifier=None, stack=None):
        super().__init__(message)
        self.message = message
        self.identifier = identifier if identifier is not None else []
        self.stack = stack if stack is not None else []


def logError(exception):
    """
    Custom logging function to log errors.
    """
    logging.error(f"Message: {exception.message}")
    if exception.identifier:
        logging.error(f"Identifier: {exception.identifier}")
    if exception.stack:
        logging.error(f"Stack trace: {exception.stack}")
    print(f"Error logged: {exception.message}")


def _evaluate_spline_array(z, c, d, xi, x_values):
    """
    Vectorized counterpart of fsplevalIO for a 1D array of query points.
    Values outside the tabulated interval are clamped to the nearest endpoint.
    """
    x_values = np.asarray(x_values, dtype=float)
    if x_values.size == 0:
        return np.empty_like(x_values)

    x_clamped = np.clip(x_values, xi[0], xi[-1])
    intervals = np.searchsorted(xi, x_clamped, side="right") - 1
    intervals = np.clip(intervals, 0, len(xi) - 2)

    x_left = xi[intervals]
    x_right = xi[intervals + 1]
    h = x_right - x_left
    t1 = x_right - x_clamped
    t2 = x_clamped - x_left

    return (
        t1 * (z[intervals] * t1 * t1 / h + c[intervals])
        + t2 * (z[intervals + 1] * t2 * t2 / h + d[intervals])
    )


def pseudoDiag(Domain, Atoms, elem, N_elements, return_info=False, build_hpot=True):
    """
    Set up initial screening and local ionic pseudopotential.

    This version keeps the original interface, but accelerates the hot path by
    evaluating each spline row in vectorized NumPy instead of calling fsplevalIO
    repeatedly for each x-point.
    """
    AtomFuncData, data_list = splineData()

    nx, ny, nz = Domain["nx"], Domain["ny"], Domain["nz"]
    h = Domain["h"]
    rad = Domain["radius"]

    ndim = nx * ny * nz
    pot = np.zeros(ndim)
    rho0 = np.zeros(ndim)
    hpot0 = np.zeros(ndim)

    N_types = len(Atoms)
    Z_sum = 0.0
    rrho_scale = (h ** 3) / (4 * np.pi)

    x_grid = np.arange(nx, dtype=float) * h - rad
    y_grid = np.arange(ny, dtype=float) * h - rad
    z_grid = np.arange(nz, dtype=float) * h - rad

    i_charge = data_list.index("charge")
    i_pot_s = data_list.index("pot_S")
    i_hartree = data_list.index("hartree")

    for at_typ in range(N_types):
        typ = Atoms[at_typ]["typ"]
        xyz = Atoms[at_typ]["coord"]
        natoms = xyz.shape[0]

        elem_index = None
        for i in range(N_elements):
            if typ == elem["Element"].iloc[i]:
                elem_index = i
                Z = elem["Z"].iloc[i]
                Z_sum += Z * natoms
                break

        if elem_index is None:
            raise ValueError(f"Element {typ} not found in elements table.")

        atom_data_index = 0
        for i in range(len(AtomFuncData)):
            if typ == AtomFuncData[i]["atom"]:
                atom_data_index = i
                break

        atom_data = AtomFuncData[atom_data_index]["data"]
        x_charg = atom_data[:, 0]
        y_charg = atom_data[:, i_charge]
        x_pot_s = atom_data[:, 0]
        y_pot_s = atom_data[:, i_pot_s]
        x_vhart = atom_data[:, 0]
        y_vhart = atom_data[:, i_hartree]

        I = preProcess(y_charg)
        x_charg, y_charg = x_charg[I], y_charg[I]

        I = preProcess(y_pot_s)
        x_pot_s, y_pot_s = x_pot_s[I], y_pot_s[I]

        I = preProcess(y_vhart)
        x_vhart, y_vhart = x_vhart[I], y_vhart[I]

        z_chg, c_chg, d_chg = fspline(x_charg, y_charg)
        z_p_s, c_p_s, d_p_s = fspline(x_pot_s, y_pot_s)
        z_vht, c_vht, d_vht = fspline(x_vhart, y_vhart)

        charge_support = float(x_charg[-1])

        for at in range(natoms):
            x_shift_sq = (x_grid - xyz[at, 0]) ** 2
            y_shift_sq = (y_grid - xyz[at, 1]) ** 2
            z_shift_sq = (z_grid - xyz[at, 2]) ** 2

            for k_idx in range(nz):
                z_term = z_shift_sq[k_idx]
                k_offset = nx * ny * k_idx

                for j_idx in range(ny):
                    row_offset = k_offset + nx * j_idx
                    r1 = np.sqrt(x_shift_sq + y_shift_sq[j_idx] + z_term)

                    ppot = _evaluate_spline_array(z_p_s, c_p_s, d_p_s, x_pot_s, r1)
                    rrho = _evaluate_spline_array(z_chg, c_chg, d_chg, x_charg, r1)
                    if build_hpot:
                        hpot00 = _evaluate_spline_array(z_vht, c_vht, d_vht, x_vhart, r1)

                    rrho[r1 > charge_support] = 0.0
                    rrho = np.maximum(0.0, rrho * rrho_scale)

                    row_end = row_offset + nx
                    pot[row_offset:row_end] += ppot
                    rho0[row_offset:row_end] += rrho
                    if build_hpot:
                        hpot0[row_offset:row_end] += hpot00

    print(
        "[atomic] rho0 sum =",
        float(rho0.sum()),
        " min =",
        float(rho0.min()),
        " max =",
        float(rho0.max()),
        " any_nonzero =",
        bool(np.any(rho0 != 0)),
    )

    rho0_sum = np.sum(rho0)
    electron_count_initial = float(rho0_sum)
    rho0 = Z_sum * rho0 / rho0_sum
    electron_count_norm = float(np.sum(rho0))
    print("[atomic] rho0 sum =", np.sum(rho0))

    diag_info = {
        "electron_count_initial": electron_count_initial,
        "electron_count_normalized": electron_count_norm,
        "electron_target": float(Z_sum),
    }

    if return_info:
        return rho0, hpot0, pot, diag_info
    return rho0, hpot0, pot
