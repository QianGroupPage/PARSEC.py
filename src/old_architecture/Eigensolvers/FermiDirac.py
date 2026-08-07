import numpy as np

def FermiDirac(lam, EF, Temp, Nelec):
    """
    Parameters:
    lam (ndarray): Array of eigenvalues.
    EF (float): Fermi level.
    Temp (float): Temperature in energy units.
    Nelec (float): Total number of electrons.

    Returns:
    fe (float): Error related to the sum of occupation numbers and Nelec.
    occup (ndarray): Occupation numbers for each eigenvalue.
    """
    # Constants
    kT = Temp * 6.33327186e-06  # Scaling temperature
    spin = 1  # Spin factor; adjust as needed

    # Calculate the occupation numbers using a stable logistic form to avoid overflow
    logits = (lam - EF) / kT
    exp_neg = np.exp(-np.abs(logits))
    occup = np.where(
        logits >= 0,
        spin * exp_neg / (1.0 + exp_neg),
        spin / (1.0 + exp_neg),
    )

    # Compute the deviation from the expected number of electrons
    fe = np.sum(occup) - Nelec / 2

    return fe, occup
