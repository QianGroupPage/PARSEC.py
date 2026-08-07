def nelectrons(Atoms, elem, N_elements):
    """
    Computes the total number of valence electrons for a given set of atoms.

    Parameters:
    Atoms (list): A list of atom dictionaries, each containing:
                  - 'typ': The atom type (string, e.g., 'H', 'C')
                  - 'coord': The coordinates of the atom (Nx3 array)

    Returns:
    Nelec (int): The total number of valence electrons.
    """
    Nelec = 0  # Initialize total number of valence electrons

    # Iterate over each atom type in Atoms to compute the total valence electrons
    for atom in Atoms:
        typ = atom['typ']
        for i in range(N_elements):
            if typ == elem['Element'].iloc[i]:
                val = elem['Z'].iloc[i]
                Nelec += len(atom['coord']) * val  # Multiply by number of atoms of this type

    return Nelec
