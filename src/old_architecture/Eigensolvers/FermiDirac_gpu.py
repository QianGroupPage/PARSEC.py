import cupy as cp

def FermiDirac(lam, EF, Temp, Nelec):
    """
    GPU-accelerated Fermi-Dirac distribution using CuPy.

    Parameters:
    lam (cp.ndarray): Eigenvalues (GPU array).
    EF (float): Fermi level.
    Temp (float): Temperature in energy units.
    Nelec (float): Total number of electrons.

    Returns:
    fe (float): Deviation from expected electron count.
    occup (cp.ndarray): Occupation numbers on GPU.
    """
    # Constants
    kT = Temp * 6.33327186e-06
    spin = 1  # spin factor

    # Fermi–Dirac distribution
    t = 1 + cp.exp((lam - EF) / kT)
    occup = spin / t

    # Electron count deviation
    fe = cp.sum(occup) - Nelec / 2

    return fe, occup
