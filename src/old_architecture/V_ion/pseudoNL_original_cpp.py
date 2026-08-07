"""Additive wrapper for the native C++/OpenMP pseudoNL path."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from ..Splines.fspline import fspline

from .preProcess import preProcess
from .splineData import splineData


def _load_native():
    try:
        import rsdft_native
    except ImportError as exc:
        raise ImportError(
            "The rsdft_native extension is not built yet. "
            "From the repo root, install the build dependencies and run "
            "`python -m pip install -v .`."
        ) from exc
    return rsdft_native


@lru_cache(maxsize=1)
def _nl_templates():
    atom_func_data, data_list = splineData()
    i_pot_p = data_list.index("pot_P")
    i_pot_s = data_list.index("pot_S")
    i_wfn_p = data_list.index("wfn_P")

    templates = {}
    for atom_entry in atom_func_data:
        atom_data = atom_entry["data"]
        xi = atom_data[:, 0]
        rows, cols = atom_data.shape

        if cols <= i_pot_p:
            pot_ps = np.zeros(rows, dtype=np.float64)
        else:
            pot_ps = np.asarray(atom_data[:, i_pot_p], dtype=np.float64).copy()
        if cols > i_pot_s:
            pot_ps -= np.asarray(atom_data[:, i_pot_s], dtype=np.float64)

        if cols <= i_wfn_p:
            wfn_p = np.zeros(rows, dtype=np.float64)
        else:
            wfn_p = np.asarray(atom_data[:, i_wfn_p], dtype=np.float64).copy()

        keep = preProcess(wfn_p)
        xi_wfn_p = np.ascontiguousarray(xi[keep], dtype=np.float64)
        wfn_p = np.ascontiguousarray(wfn_p[keep], dtype=np.float64)
        z_wav, c_wav, d_wav = fspline(xi_wfn_p, wfn_p)

        keep = preProcess(pot_ps)
        xi_pot_ps = np.ascontiguousarray(xi[keep], dtype=np.float64)
        pot_ps = np.ascontiguousarray(pot_ps[keep], dtype=np.float64)
        z_pot_ps, c_pot_ps, d_pot_ps = fspline(xi_pot_ps, pot_ps)

        templates[atom_entry["atom"]] = {
            "xi_wfn_p": xi_wfn_p,
            "z_wfn_p": np.ascontiguousarray(z_wav, dtype=np.float64),
            "c_wfn_p": np.ascontiguousarray(c_wav, dtype=np.float64),
            "d_wfn_p": np.ascontiguousarray(d_wav, dtype=np.float64),
            "xi_pot_ps": xi_pot_ps,
            "z_pot_ps": np.ascontiguousarray(z_pot_ps, dtype=np.float64),
            "c_pot_ps": np.ascontiguousarray(c_pot_ps, dtype=np.float64),
            "d_pot_ps": np.ascontiguousarray(d_pot_ps, dtype=np.float64),
        }

    return templates


def build_info():
    """Return basic information about the compiled native extension."""
    return _load_native().build_info()


def pseudoNL(Domain, Atoms, elem, N_elements):
    """Prepare spline data in Python, then run the hot loops in native code."""
    native = _load_native()
    templates = _nl_templates()
    elem_lookup = {
        str(elem["Element"].iloc[index]): {
            "Zvalue": float(elem["Zvalue"].iloc[index]),
            "R": float(elem["R"].iloc[index]),
        }
        for index in range(N_elements)
    }

    species = []
    for atom in Atoms:
        typ = atom["typ"]
        if typ not in templates:
            raise ValueError(f"No spline template found for atom type {typ}.")
        if typ not in elem_lookup:
            raise ValueError(f"Element {typ} not found in elements table.")

        coords = np.ascontiguousarray(np.asarray(atom["coord"], dtype=np.float64))
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Atom coordinates for {typ} must have shape (n, 3).")

        species.append(
            {
                "typ": typ,
                "coords": coords,
                "xint": elem_lookup[typ]["Zvalue"] / (float(Domain["h"]) ** 3),
                "rzero": elem_lookup[typ]["R"],
                **templates[typ],
            }
        )

    payload = native.pseudo_nl_omp(Domain, species)
    ndim = int(Domain["nx"]) * int(Domain["ny"]) * int(Domain["nz"])
    rows = np.asarray(payload["rows"], dtype=np.int64)
    cols = np.asarray(payload["cols"], dtype=np.int64)
    data = np.asarray(payload["data"], dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(ndim, ndim))
