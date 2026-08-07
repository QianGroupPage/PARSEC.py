import numpy as np
import scipy.sparse as sp
from .splineData import splineData
from .preProcess import preProcess
from ..Splines.fspline import fspline
from ..Splines.fsplevalIO import fsplevalIO

def pseudoNL(Domain, Atoms, elem, N_elements):
    """
    Optimized version of pseudoNL for better performance and memory usage.
    """
    # Extract domain variables
    nx = Domain['nx']
    ny = Domain['ny']
    nz = Domain['nz']
    h = Domain['h']
    rad = Domain['radius']

    print('Atoms', Atoms)
    print('Domain', Domain)

    # Load spline data
    AtomFuncData, data_list = splineData()

    # Dimension of the sparse matrix
    ndim = nx * ny * nz

    # Find column indices for pot_P, pot_S, and wav_P
    pot_P = pot_S = wav_P = 0
    for i, data in enumerate(data_list):
        if data == 'pot_P':
            pot_P = i
        elif data == 'pot_S':
            pot_S = i
        elif data == 'wfn_P':
            wav_P = i

    # Collect all sparse matrix data in lists (much faster than building incrementally)
    all_rows = []
    all_cols = []
    all_data = []

    N_types = len(Atoms)

    # Iterate over all atom types
    for at_typ in range(N_types):
        typ = Atoms[at_typ]['typ']
        
        # Find atom data index
        index = next((i for i, atom_data in enumerate(AtomFuncData) 
                     if atom_data['atom'] == typ), 0)

        # Extract spline data
        xi = AtomFuncData[index]['data'][:, 0]
        rows, cols = AtomFuncData[index]['data'].shape

        # Get potential and wavefunction data
        if cols >= pot_P:
            potPS = AtomFuncData[index]['data'][:, pot_P].copy()
        else:
            potPS = np.zeros(rows)

        if cols >= pot_S:
            potPS -= AtomFuncData[index]['data'][:, pot_S]

        if cols >= wav_P:
            wfn_P = AtomFuncData[index]['data'][:, wav_P].copy()
        else:
            wfn_P = np.zeros(rows)

        # Pre-process data
        I = preProcess(wfn_P)
        wfn_P = wfn_P[I]
        xi_wfn_P = xi[I]
        
        I = preProcess(potPS)
        potPS = potPS[I]
        xi_potPS = xi[I]

        # Generate spline coefficients
        zWav, cWav, dWav = fspline(xi_wfn_P, wfn_P)
        zPotPS, cPotPS, dPotPS = fspline(xi_potPS, potPS)

        # Find element data index
        elem_index = next((i for i in range(N_elements) 
                          if typ == elem['Element'].iloc[i]), 0)

        xint = elem['Zvalue'].iloc[elem_index] / (h ** 3)
        xyz = Atoms[at_typ]['coord']
        natoms = xyz.shape[0]
        Rzero = elem['R'].iloc[elem_index]
        span = int(round(Rzero / h))

        # Process each atom of this type
        for at in range(natoms):
            xxa, yya, zza = xyz[at]

            # Calculate grid boundaries
            i0 = int(round((xxa + rad) / h + 1))
            j0 = int(round((yya + rad) / h + 1))
            k0 = int(round((zza + rad) / h + 1))

            # Pre-allocate arrays for maximum possible points
            max_points = (2 * span + 1) ** 3
            nn = np.zeros(max_points, dtype=np.int32)
            xx = np.zeros(max_points)
            yy = np.zeros(max_points)
            zz = np.zeros(max_points)
            dd = np.zeros(max_points)
            vspp = np.zeros(max_points)
            wavpp = np.zeros(max_points)

            indx = 0
            # Vectorized grid point collection
            for k in range(k0 - span, k0 + span + 1):
                zzz = (k - 1) * h - rad - zza
                for j in range(j0 - span, j0 + span + 1):
                    yyy = (j - 1) * h - rad - yya
                    # Reset interval guesses for each y-slice
                    j_p_ps = 1
                    j_wfn = 1
                    for i in range(i0 - span, i0 + span + 1):
                        xxx = (i - 1) * h - rad - xxa
                        dd1 = np.sqrt(xxx*xxx + yyy*yyy + zzz*zzz)

                        if 0 < dd1 < Rzero:
                            nn[indx] = i + nx * ((j - 1) + (k - 1) * ny)
                            xx[indx] = xxx
                            yy[indx] = yyy
                            zz[indx] = zzz
                            dd[indx] = dd1

                            vspp_val, j_p_ps = fsplevalIO(zPotPS, cPotPS, dPotPS, 
                                                         xi_potPS, dd1, j_p_ps)
                            wavpp_val, j_wfn = fsplevalIO(zWav, cWav, dWav, 
                                                         xi_wfn_P, dd1, j_wfn)

                            vspp[indx] = vspp_val
                            wavpp[indx] = wavpp_val
                            indx += 1

            if indx == 0:
                continue

            # Trim arrays to actual size
            nn = nn[:indx]
            xx = xx[:indx]
            yy = yy[:indx]
            zz = zz[:indx]
            dd = dd[:indx]
            vspp = vspp[:indx]
            wavpp = wavpp[:indx]

            # Vectorized calculations
            fac = wavpp * vspp
            ulmspx = xx / dd * fac
            ulmspy = yy / dd * fac
            ulmspz = zz / dd * fac

            # Compute matrix elements more efficiently
            # Instead of creating full matrices, compute elements directly
            for i in range(indx):
                for j in range(indx):
                    value = (ulmspx[i] * ulmspx[j] + 
                            ulmspy[i] * ulmspy[j] + 
                            ulmspz[i] * ulmspz[j]) / xint
                    
                    if abs(value) > 1e-12:  # Only store significant values
                        all_rows.append(nn[i])
                        all_cols.append(nn[j])
                        all_data.append(value)

    # Create sparse matrix once from collected data
    if all_data:
        vnl = sp.csr_matrix((all_data, (all_rows, all_cols)), shape=(ndim, ndim))
    else:
        vnl = sp.csr_matrix((ndim, ndim))

    return vnl


def pseudoNL_vectorized(Domain, Atoms, elem, N_elements):
    """
    Highly optimized vectorized version using advanced NumPy operations.
    """
    # Extract domain variables
    nx = Domain['nx']
    ny = Domain['ny'] 
    nz = Domain['nz']
    h = Domain['h']
    rad = Domain['radius']

    # Load spline data
    AtomFuncData, data_list = splineData()
    ndim = nx * ny * nz

    # Find column indices
    pot_P = pot_S = wav_P = 0
    for i, data in enumerate(data_list):
        if data == 'pot_P':
            pot_P = i
        elif data == 'pot_S':
            pot_S = i
        elif data == 'wfn_P':
            wav_P = i

    # Use lists to collect sparse data
    rows_list = []
    cols_list = []
    data_list = []

    N_types = len(Atoms)

    for at_typ in range(N_types):
        typ = Atoms[at_typ]['typ']
        
        # Find atom data
        index = next((i for i, atom_data in enumerate(AtomFuncData) 
                     if atom_data['atom'] == typ), 0)

        # Process spline data (same as before)
        xi = AtomFuncData[index]['data'][:, 0]
        rows, cols = AtomFuncData[index]['data'].shape

        if cols >= pot_P:
            potPS = AtomFuncData[index]['data'][:, pot_P].copy()
        else:
            potPS = np.zeros(rows)

        if cols >= pot_S:
            potPS -= AtomFuncData[index]['data'][:, pot_S]

        if cols >= wav_P:
            wfn_P = AtomFuncData[index]['data'][:, wav_P].copy()
        else:
            wfn_P = np.zeros(rows)

        # Pre-process and generate splines
        I = preProcess(wfn_P)
        wfn_P = wfn_P[I]
        xi_wfn_P = xi[I]
        
        I = preProcess(potPS)
        potPS = potPS[I]
        xi_potPS = xi[I]

        zWav, cWav, dWav = fspline(xi_wfn_P, wfn_P)
        zPotPS, cPotPS, dPotPS = fspline(xi_potPS, potPS)

        # Find element data
        elem_index = next((i for i in range(N_elements) 
                          if typ == elem['Element'].iloc[i]), 0)

        xint = elem['Zvalue'].iloc[elem_index] / (h ** 3)
        xyz = Atoms[at_typ]['coord']
        Rzero = elem['R'].iloc[elem_index]
        span = int(round(Rzero / h))

        # Process each atom
        for at in range(xyz.shape[0]):
            atom_pos = xyz[at]
            
            # Create grid ranges
            i_range = np.arange(-span, span + 1)
            j_range = np.arange(-span, span + 1)
            k_range = np.arange(-span, span + 1)
            
            # Use meshgrid for vectorized grid generation
            I, J, K = np.meshgrid(i_range, j_range, k_range, indexing='ij')
            
            # Calculate positions
            i0 = int(round((atom_pos[0] + rad) / h + 1))
            j0 = int(round((atom_pos[1] + rad) / h + 1))
            k0 = int(round((atom_pos[2] + rad) / h + 1))
            
            i_coords = I + i0
            j_coords = J + j0
            k_coords = K + k0
            
            # Calculate relative positions
            xxx = (i_coords - 1) * h - rad - atom_pos[0]
            yyy = (j_coords - 1) * h - rad - atom_pos[1]
            zzz = (k_coords - 1) * h - rad - atom_pos[2]
            
            # Calculate distances
            distances = np.sqrt(xxx**2 + yyy**2 + zzz**2)
            
            # Create mask for valid points
            mask = (distances > 0) & (distances < Rzero)
            
            if not np.any(mask):
                continue
            
            # Extract valid points
            xxx_valid = xxx[mask]
            yyy_valid = yyy[mask]
            zzz_valid = zzz[mask]
            distances_valid = distances[mask]
            
            # Calculate grid indices
            nn = (i_coords[mask] + nx * ((j_coords[mask] - 1) + 
                                        (k_coords[mask] - 1) * ny))
            
            # Evaluate splines for all points at once (if possible)
            # Note: This might need modification based on fsplevalIO implementation
            vspp_vals = np.zeros(len(distances_valid))
            wavpp_vals = np.zeros(len(distances_valid))
            
            j_p_ps = j_wfn = 1
            for idx, d in enumerate(distances_valid):
                vspp_vals[idx], j_p_ps = fsplevalIO(zPotPS, cPotPS, dPotPS, 
                                                   xi_potPS, d, j_p_ps)
                wavpp_vals[idx], j_wfn = fsplevalIO(zWav, cWav, dWav, 
                                                   xi_wfn_P, d, j_wfn)
            
            # Vectorized calculation of matrix elements
            fac = wavpp_vals * vspp_vals
            ulmspx = xxx_valid / distances_valid * fac
            ulmspy = yyy_valid / distances_valid * fac
            ulmspz = zzz_valid / distances_valid * fac
            
            # Use broadcasting to compute outer products efficiently
            # This is memory-efficient for moderate-sized arrays
            n_points = len(nn)
            if n_points > 0:
                # Compute all pairwise products at once
                ulms_outer_x = np.outer(ulmspx, ulmspx)
                ulms_outer_y = np.outer(ulmspy, ulmspy)
                ulms_outer_z = np.outer(ulmspz, ulmspz)
                
                total_matrix = (ulms_outer_x + ulms_outer_y + ulms_outer_z) / xint
                
                # Create row and column indices
                row_indices = np.repeat(nn, n_points)
                col_indices = np.tile(nn, n_points)
                matrix_data = total_matrix.flatten()
                
                # Filter out small values
                significant_mask = np.abs(matrix_data) > 1e-12
                
                rows_list.extend(row_indices[significant_mask])
                cols_list.extend(col_indices[significant_mask])
                data_list.extend(matrix_data[significant_mask])

    # Create final sparse matrix
    if data_list:
        vnl = sp.csr_matrix((data_list, (rows_list, cols_list)), shape=(ndim, ndim))
    else:
        vnl = sp.csr_matrix((ndim, ndim))

    return vnl
