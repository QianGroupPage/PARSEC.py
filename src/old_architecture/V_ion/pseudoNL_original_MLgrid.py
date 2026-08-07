import numpy as np
import scipy.sparse as sp
from .splineData import splineData
from .preProcess import preProcess
from ..Splines.fspline import fspline
from ..Splines.fsplevalIO import fsplevalIO
from .spconvert import spconvert

def pseudoNL_MLgrid(Domain, Atoms, elem, N_elements):
    """
    Computes the non-local pseudopotential contribution to potential energy.

    Parameters:
    Domain (dict): Describes the spherical area where the nonlocal pseudopotential
                   should be analyzed (contains nx, ny, nz, h, radius).
    Atoms (list of dict): Contains information about the different atoms of interest.

    Returns:
    vnl (scipy.sparse.csr_matrix): Sparse matrix for nonlocal pseudopotential.
    """

    # Extract domain variables
    nx = Domain['nx']
    ny = Domain['ny']
    nz = Domain['nz']
    h = Domain['h']
    rad = Domain['radius']

    # Load spline data (Assuming we have a function that does this)
    AtomFuncData, data_list = splineData()

    # Dimension of the sparse matrix for nonlocal pseudopotential (vnl)
    ndim = nx * ny * nz

    # imax is the default maximum nonzero values that should be found in vnl
    # imax = optimalSize(Atoms, h)  # Custom function that calculates optimal size
    # Zeyi: no need to predetermine the number of elements in the sparse vnl

    # Initialize sparse matrix for the nonlocal pseudopotential
    vnl = sp.lil_matrix((ndim, ndim), dtype=np.float64)  # LIL format for efficient construction

    # Number of atomic species (types of atoms)
    N_types = len(Atoms)

    # Initialize variables for pot_P, pot_S, and wav_P
    pot_P = 0
    pot_S = 0
    wav_P = 0

    # Find out what columns pot_P, pot_S, and wav_P are stored in
    for i, data in enumerate(data_list):
        if data == 'pot_P':
            pot_P = i
        elif data == 'pot_S':
            pot_S = i
        elif data == 'wfn_P':
            wav_P = i

    # Iterate over all atom types
    for at_typ in range(N_types):
        typ = Atoms[at_typ]['typ']
        index = 0
        for i in range(len(AtomFuncData)):
            if AtomFuncData[i]['atom'] == typ:
                index = i
                break

        # Extract spline data for wave functions and potentials
        # We assume that the first column is for the radius
        xi = AtomFuncData[index]['data'][:, 0]
        # Now we need to find the index for the P wave function
        # Now we do the same thing for finding the P and S potentials
        rows, cols = AtomFuncData[index]['data'].shape

        # Initialize potPS based on the presence of pot_P
        if cols < pot_P:
            potPS = np.zeros((rows, 1))
        else:
            potPS = AtomFuncData[index]['data'][:, pot_P]

        # Adjust potPS if pot_S is available
        if cols >= pot_S:
            potPS = potPS - AtomFuncData[index]['data'][:, pot_S]

        # Initialize wfn_P based on the presence of wav_P
        if cols < wav_P:
            wfn_P = np.zeros((rows, 1))
        else:
            wfn_P = AtomFuncData[index]['data'][:, wav_P]

        # Pre-process the data
        I = preProcess(wfn_P)
        wfn_P = wfn_P[I]
        xi_wfn_P = xi[I]
        I = preProcess(potPS)
        potPS = potPS[I]
        xi_potPS = xi[I]

        zWav, cWav, dWav = fspline(xi_wfn_P, wfn_P)
        zPotPS, cPotPS, dPotPS = fspline(xi_potPS, potPS)
        # End generating the coefficients for splines

        # Find the element data index in elem
        for i in range(N_elements):
            if typ == elem['Element'].iloc[i]:
                index = i
                break

        xint = elem['Zvalue'].iloc[index] / (h ** 3)
        xyz = Atoms[at_typ]['coord']
        natoms = xyz.shape[0]
        Rzero = elem['R'].iloc[index]

        # This for loop iterates through all the atoms of at_typ
        for at in range(natoms):
            # xxa, yya, zza, store x,y,z positions of atom
            xxa = xyz[at, 0]
            yya = xyz[at, 1]
            zza = xyz[at, 2]

            # Loop over grid points near atom
            i0 = int(round((xxa) / h + 1))
            j0 = int(round((yya) / h + 1))
            k0 = int(round((zza) / h + 1))

            # Determine the maximum distance from initial (x, y, z)
            span = int(round(Rzero / h))

            # Clip to valid grid indices (1..n) to avoid out-of-domain linear indices
            i_min = max(1, i0 - span)
            i_max = min(nx, i0 + span)
            j_min = max(1, j0 - span)
            j_max = min(ny, j0 + span)
            k_min = max(1, k0 - span)
            k_max = min(nz, k0 + span)

            if i_min > i_max or j_min > j_max or k_min > k_max:
                continue

            indx = 0
            nn, xx, yy, zz, dd, vspp, wavpp = [], [], [], [], [], [], []

            for k in range(k_min, k_max + 1):
                zzz = (k - 1) * h - zza
                for j in range(j_min, j_max + 1):
                    yyy = (j - 1) * h - yya
                    j_p_ps = 1
                    j_wfn = 1
                    for i in range(i_min, i_max + 1):
                        xxx = (i - 1) * h - xxa
                        # calculate distance from (xxx, yyy, zzz) to atom
                        dd1 = np.sqrt(xxx ** 2 + yyy ** 2 + zzz ** 2)

                        if 0 < dd1 < Rzero:
                            indx += 1
                            nn.append(i + nx * ((j - 1) + (k - 1) * ny))
                            xx.append(xxx)
                            yy.append(yyy)
                            zz.append(zzz)
                            dd.append(dd1)

                            vspp_val, j_p_ps = fsplevalIO(zPotPS, cPotPS, dPotPS, xi_potPS, dd1, j_p_ps)
                            wavpp_val, j_wfn = fsplevalIO(zWav, cWav, dWav, xi_wfn_P, dd1, j_wfn)

                            vspp.append(vspp_val)
                            wavpp.append(wavpp_val)

            # Optimization via pre-allocation
            # Here we're creating an imx x 3 matrix of zeros.
            # The natoms * part was removed because, in practice, the code never needs more than
            # indx^2 + 1 space. The ?indx+1? might have been a question in the original code.

            if indx == 0:
                continue

            vnll = np.zeros((indx ** 2 + 1, 3))

            # The following calculations have been moved outside of loops to optimize performance.
            # Explanation of the operations below:
            # - { xx, yy, zz } / dd calculates px, py, and pz respectively, element by element.
            # - wavpp * vspp calculates the 'fac' for ulmspx2, ulmspy2, and ulmspz2.
            # - The operation { px, py, pz } * fac results in the values for ulmspx2, ulmspy2, and ulmspz2.
            # - The difference between ulmspx1 and ulmspx2 is a factor of / xint.
            # - Therefore, ulmspx2 is computed first, then ulmspx1 is calculated by dividing ulmspx2 by / xint.
            # This approach reduces the number of individual calculations and performs them more efficiently.

            ulmspx = np.array(xx) / np.array(dd) * np.array(wavpp) * np.array(vspp)
            ulmspy = np.array(yy) / np.array(dd) * np.array(wavpp) * np.array(vspp)
            ulmspz = np.array(zz) / np.array(dd) * np.array(wavpp) * np.array(vspp)

            # Well, should all be real here
            xmatrix = np.outer(ulmspx, ulmspx) / xint
            ymatrix = np.outer(ulmspy, ulmspy) / xint
            zmatrix = np.outer(ulmspz, ulmspz) / xint

            total = xmatrix + ymatrix + zmatrix

            for i in range(indx):
                vnll[i * indx:(i + 1) * indx, 0] = nn[i]
                vnll[i * indx:(i + 1) * indx, 1] = nn[:indx]
                vnll[i * indx:(i + 1) * indx, 2] = total[i, :indx]

            vnll[indx ** 2, :] = [ndim, ndim, 0]
            #vnl += sp.coo_matrix((vnll[:, 2], (vnll[:, 0].astype(int), vnll[:, 1].astype(int))), shape=(ndim, ndim))

            vnl += spconvert(vnll)

    return vnl
