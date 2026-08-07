"""Additive wrapper for the native C++/OpenMP pseudoDiag path."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

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
def _diag_templates():
    atom_func_data, data_list = splineData()
    i_charge = data_list.index("charge")
    i_pot_s = data_list.index("pot_S")
    i_hartree = data_list.index("hartree")

    templates = {}
    for atom_entry in atom_func_data:
        atom_data = atom_entry["data"]

        x_charg = atom_data[:, 0]
        y_charg = atom_data[:, i_charge]
        keep = preProcess(y_charg)
        x_charg = np.ascontiguousarray(x_charg[keep], dtype=np.float64)
        y_charg = np.ascontiguousarray(y_charg[keep], dtype=np.float64)
        z_chg, c_chg, d_chg = fspline(x_charg, y_charg)

        x_pot_s = atom_data[:, 0]
        y_pot_s = atom_data[:, i_pot_s]
        keep = preProcess(y_pot_s)
        x_pot_s = np.ascontiguousarray(x_pot_s[keep], dtype=np.float64)
        y_pot_s = np.ascontiguousarray(y_pot_s[keep], dtype=np.float64)
        z_p_s, c_p_s, d_p_s = fspline(x_pot_s, y_pot_s)

        x_vhart = atom_data[:, 0]
        y_vhart = atom_data[:, i_hartree]
        keep = preProcess(y_vhart)
        x_vhart = np.ascontiguousarray(x_vhart[keep], dtype=np.float64)
        y_vhart = np.ascontiguousarray(y_vhart[keep], dtype=np.float64)
        z_vht, c_vht, d_vht = fspline(x_vhart, y_vhart)

        templates[atom_entry["atom"]] = {
            "x_charge": x_charg,
            "z_charge": np.ascontiguousarray(z_chg, dtype=np.float64),
            "c_charge": np.ascontiguousarray(c_chg, dtype=np.float64),
            "d_charge": np.ascontiguousarray(d_chg, dtype=np.float64),
            "x_pot_s": x_pot_s,
            "z_pot_s": np.ascontiguousarray(z_p_s, dtype=np.float64),
            "c_pot_s": np.ascontiguousarray(c_p_s, dtype=np.float64),
            "d_pot_s": np.ascontiguousarray(d_p_s, dtype=np.float64),
            "x_hartree": x_vhart,
            "z_hartree": np.ascontiguousarray(z_vht, dtype=np.float64),
            "c_hartree": np.ascontiguousarray(c_vht, dtype=np.float64),
            "d_hartree": np.ascontiguousarray(d_vht, dtype=np.float64),
        }

    return templates


def build_info():
    """Return basic information about the compiled native extension."""
    return _load_native().build_info()


def pseudoDiag(Domain, Atoms, elem, N_elements, return_info=False, build_hpot=True):
    """Prepare spline data in Python, then run the hot loops in native code."""
    native = _load_native()
    templates = _diag_templates()
    elem_lookup = {
        str(elem["Element"].iloc[index]): float(elem["Z"].iloc[index])
        for index in range(N_elements)
    }

    species = []
    z_sum = 0.0
    for atom in Atoms:
        typ = atom["typ"]
        if typ not in templates:
            raise ValueError(f"No spline template found for atom type {typ}.")
        if typ not in elem_lookup:
            raise ValueError(f"Element {typ} not found in elements table.")

        coords = np.ascontiguousarray(np.asarray(atom["coord"], dtype=np.float64))
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Atom coordinates for {typ} must have shape (n, 3).")

        z_sum += elem_lookup[typ] * coords.shape[0]
        species.append(
            {
                "typ": typ,
                "coords": coords,
                **templates[typ],
            }
        )

    return native.pseudo_diag_omp(
        Domain,
        species,
        z_sum,
        return_info=return_info,
        build_hpot=build_hpot,
    )
