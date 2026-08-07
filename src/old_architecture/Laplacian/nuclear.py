import os
import numpy as np
import pandas as pd
from pathlib import Path


def load_elements():
    """
    Load the element data from 'elements_new.csv' assuming there is no header,
    and that the first column contains the element type.
    """
    # src/ directory (one level above Laplacian/)
    base_dir = Path(__file__).resolve().parents[1]
    csv_path = base_dir / "elements_new.csv"
    elem = pd.read_csv(csv_path, header=None)

    # Assign column names (assuming we know the structure)
    elem.columns = ['Element', 'AtomicNumber', 'Z', 'unknown', 'h', 'r'] + [f'Property{i}' for i in range(1, 6)]

    N_elements = elem.shape[0]  # This gives the number of elements (rows)
    return elem, N_elements

# Call the function to load the element data
elem, N_elements = load_elements()

def nuclear(Domain, Atoms, elem, N_elements):
    """
    Calculate the nuclear repulsion term between atoms based on their coordinates.

    Parameters:
    Domain (dict): Contains domain/grid properties.
    Atoms (list): List of atomic species with type and coordinates.

    Returns:
    E_nuc0 (float): The nuclear repulsion energy.
    """

    # Initialize variables
    indx = 0
    xx, yy, zz, zt = [], [], [], []  # To store coordinates and atomic numbers

    # Iterate over atomic types
    N_types = len(Atoms)
    for at_typ in range(N_types):
        typ = Atoms[at_typ]['typ']

        # Look for matching element data in the elem DataFrame
        Z = 0
        xyz = []
        natoms = 0
        for i in range(N_elements):
            if typ == elem['Element'].iloc[i]:
                Z = elem['Z'].iloc[i]  # Get the Z
                xyz = Atoms[at_typ]['coord']
                natoms = xyz.shape[0]  # Number of atoms of this type

        # Store coordinates and atomic numbers
        for at in range(natoms):
            indx += 1
            xx.append(xyz[at][0])  # x-coordinates
            yy.append(xyz[at][1])  # y-coordinates
            zz.append(xyz[at][2])  # z-coordinates
            zt.append(Z)  # Atomic number

    # Convert lists to numpy arrays for efficient calculations
    xx = np.array(xx)
    yy = np.array(yy)
    zz = np.array(zz)
    zt = np.array(zt)

    # Calculate nuclear repulsion energy
    E_nuc0 = 0
    for i in range(indx):
        for j in range(i + 1, indx):
            xx1, yy1, zz1, Z1 = xx[i], yy[i], zz[i], zt[i]
            xx2, yy2, zz2, Z2 = xx[j], yy[j], zz[j], zt[j]

            # Compute distances to all other atoms
            d2 = (xx1 - xx2) ** 2 + (yy1 - yy2) ** 2 + (zz1 - zz2) ** 2

            # # Exclude self-interactions (d2 == 0)
            # non_zero_mask = d2 != 0
            # Z2 = zt[non_zero_mask]
            # R2 = np.sqrt(d2[non_zero_mask])
            R2 = np.sqrt(d2)

            # Accumulate the repulsion energy
            E_nuc0 += Z1 * Z2 / R2
    E_nuc0 *= 2
    # Return the nuclear repulsion energy
    return E_nuc0

# Domain = {
#     'radius': 7.175,
#     'nx': 42,
#     'ny': 42,
#     'nz': 42,
#     'h': 0.35
# }
# Atoms = [
#     {'typ': 'H', 'coord': np.array([[0, 0, 0], [1, 0, 0]])},
#     {'typ': 'O', 'coord': np.array([[0.5, 0.5, 0]])}
# ]
#
# enuc = nuclear(Domain, Atoms, elem, N_elements)
# print("Nuclear repulsion energy:", enuc)
