import os
import scipy.io


def splineData():
    """
    Load spline data for atoms from the 'splineData.mat' file.

    Returns:
    AtomFuncData (list of dicts): List of dictionaries containing atom-specific data.
    data_list (list): List of data labels (e.g., charge, hartree, etc.).
    """
    # Load the data from splineData.mat (anchored to this file's directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mat_path = os.path.join(base_dir, 'splineData.mat')
    mat_data = scipy.io.loadmat(mat_path)

    # Initialize the list to store atomic data
    AtomFuncData = []

    # Define atoms and load their respective data
    atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
             'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']

    # Iterate over each atom and load its data from the .mat file
    for atom in atoms:
        data_key = 'data' + atom  # The key used in the .mat file for each atom's data
        if data_key in mat_data:
            atom_data = mat_data[data_key]
            AtomFuncData.append({'atom': atom, 'data': atom_data})
        else:
            raise ValueError(f"Data for atom {atom} not found in 'splineData.mat'")

    # List of data labels
    data_list = ['radius', 'charge', 'hartree', 'pot_P', 'pot_S', 'wfn_P', 'wfn_S']

    return AtomFuncData, data_list
