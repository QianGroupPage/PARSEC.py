import numpy as np
from .splineData import splineData
from .preProcess import preProcess
from ..Splines.fspline import fspline
from ..Splines.fsplevalIO import fsplevalIO

def pseudoDiag_MLgrid(Domain, Atoms, elem, N_elements, return_info=False, build_hpot=True):
    """
    Set up initial screening and local ionic pseudopotential using SAD atomic superposition
    on the provided grid (e.g. ML-derived grid).
    Args:
        Domain: Grid domain information
        Atoms: Atomic coordinates and types
        elem: Element information
        N_elements: Number of elements

    Returns:
        rho0: SAD initial charge density (flattened)
        hpot0: Hartree potential calculated from SAD density
        pot: Pseudopotential from atomic superposition
        diag_info (dict, optional): Electron count diagnostics when return_info=True.
    """
    OPTIMIZATIONLEVEL = 0
    enableMexFilesTest = 0

    # print("Atoms", Atoms)
    # print("Domain", Domain)

    # Extract domain information
    nx, ny, nz = Domain['nx'], Domain['ny'], Domain['nz']
    h = Domain['h']
    rad = Domain['radius']
    n = nx * ny * nz # Total number of grid points

    # Initialize output arrays
    rho0 = np.zeros(n) # Flattened for compatibility with SCF loop
    pot = np.zeros(n)
    hpot0 = np.zeros(n)

    # ==========================================
    # 1. PSEUDOPOTENTIAL CALCULATION (UNCHANGED FROM ATOMIC APPROACH)
    # ==========================================

    print("Calculating pseudopotential from atomic superposition...")
    # Get atomic functional data
    AtomFuncData, data_list = splineData()
    N_types = len(Atoms)
    Z_sum = 0.0

    # Atom-type loop
    for at_typ in range(N_types):
        typ = Atoms[at_typ]['typ']
        xyz = Atoms[at_typ]['coord']
        natoms = xyz.shape[0]
        # Look for matching element data in the elem DataFrame
        for i in range(N_elements):
            if typ == elem['Element'].iloc[i]:
                Z = elem['Z'].iloc[i]
                Z_sum += Z * natoms

        # Search for the atom's data in AtomFuncData
        index = 0
        for i in range(len(AtomFuncData)):
            if typ == AtomFuncData[i]['atom']:
                index = i
                break

        # Localizing variables and initializing arrays
        i_charge = data_list.index('charge')
        i_pot_S = data_list.index('pot_S')
        i_hartree = data_list.index('hartree')

        atom_data = AtomFuncData[index]['data']
        x_charg = atom_data[:, 0]
        y_charg = atom_data[:, i_charge]
        x_pot_s = atom_data[:, 0]
        y_pot_s = atom_data[:, i_pot_S]
        x_vhart = atom_data[:, 0]
        y_vhart = atom_data[:, i_hartree]

        
        # ---------- check raw charge data start ----------
        # # x_charg: shape (M,), non-uniform radii in Bohr
        # # y_charg: shape (M,), assumed f(r) = 4πρ(r) in Bohr units

        # r = x_charg.astype(float)
        # f = y_charg.astype(float)

        # # sanity: ensure strictly increasing
        # # (preProcess usually fixes this, but double-check)
        # order = np.argsort(r)
        # r = r[order]
        # f = f[order]
        # assert np.all(np.diff(r) > 0), "x_charg must be strictly increasing"

        # dr = np.diff(r)
        # fi = f[:-1]
        # fj = f[1:]
        # ri2 = r[:-1] ** 2
        # rj2 = r[1:] ** 2

        # # trapezoid on non-uniform grid for N = ∫ f(r) r^2 dr
        # N_est = np.sum(0.5 * (fi * ri2 + fj * rj2) * dr)
        # print("[radial CSV] N_est (should match valence) =", float(N_est))

        # # plt.figure(figsize=(7, 4))
        # # plt.plot(x_charg, y_charg, 'o-', markersize=1, label=f'Atom type {typ} (charge)')
        # # plt.xlabel("r (Bohr)")
        # # plt.ylabel("charge density (raw units)")
        # # plt.title("Raw charge vs radius from .mat")
        # # plt.legend()
        # # plt.tight_layout()
        # # plt.show()

        # print("[atomic] y_charge sum =", float(y_charg.sum()),
        #       " min =", float(y_charg.min()),
        #       " max =", float(y_charg.max()))
        # ---------- check raw charge data end ----------


        # Pre-processing the data
        I = preProcess(y_charg)
        x_charg, y_charg = x_charg[I], y_charg[I]

        I = preProcess(y_pot_s)
        x_pot_s, y_pot_s = x_pot_s[I], y_pot_s[I]

        I = preProcess(y_vhart)
        x_vhart, y_vhart = x_vhart[I], y_vhart[I]

        # Calculating the splines
        z_chg, c_chg, d_chg = fspline(x_charg, y_charg)
        z_p_s, c_p_s, d_p_s = fspline(x_pot_s, y_pot_s)
        z_vht, c_vht, d_vht = fspline(x_vhart, y_vhart)

        
        # ---------- raw data plot start ----------
        # # 6 Å in Bohr
        # rmax_bohr = 6.3 * 1.889726

        # # dense evaluation from 0 → 6 Å
        # r_plot = np.linspace(0.0, rmax_bohr, 800)  # 800 points is plenty

        # # evaluate spline
        # y_spline = np.empty_like(r_plot)
        # j = 0
        # for t, rp in enumerate(r_plot):
        #     # clamp into valid spline domain
        #     if rp < x_charg[0]:
        #         rp = x_charg[0]
        #     elif rp > x_charg[-1]:
        #         rp = x_charg[-1]
        #     y_spline[t], j = fsplevalIO(z_chg, c_chg, d_chg, x_charg, rp, j)

        # # raw data within 6 Å only
        # mask = x_charg <= rmax_bohr

        # plt.figure(figsize=(7.2, 4.2))
        # plt.plot(x_charg[mask] / 1.889726, y_charg[mask], 'o', ms=2.0, label=f'{typ}: raw')
        # plt.plot(r_plot / 1.889726, y_spline, '-', lw=1.5, label=f'{typ}: spline')
        # plt.xlabel('r (Å)')
        # plt.ylabel('f(r) = 4πρ(r)')
        # plt.title('Charge radial data (0–6 Å)')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        # ---------- raw data plot end ----------

        # Atom-specific computations (loop through the grid points)
        for at in range(natoms):
            indx = 0

            # Adjust coordinates to the grid
            k = np.arange(nz)
            dz = (k * h - xyz[at, 2]) ** 2

            for k_idx in range(nz):
                j = np.arange(ny)
                dy = dz[k_idx] + (j * h - xyz[at, 1]) ** 2

                for j_idx in range(ny):
                    i = np.arange(nx)
                    r1 = np.sqrt(dy[j_idx] + (i * h - xyz[at, 0]) ** 2)

                    # Initialize arrays for potentials and charge
                    ppot = np.zeros(nx)
                    rrho = np.zeros(nx)
                    hpot00 = np.zeros(nx) if build_hpot else None

                    # Initialization of intervals
                    j_ch = 0
                    j_p_s = 0
                    j_vht = 0

                    # Check if optimization is enabled
                    if OPTIMIZATIONLEVEL != 0:
                        # TODO: switch to C++ code to compare the performance?
                        # Call the C function (PsuedoDiagLoops), or a Python placeholder
                        # ppot, rrho, hpot00 = PsuedoDiagLoops(
                        #     z_p_s, c_p_s, d_p_s, z_chg, c_chg, d_chg,
                        #     z_vht, c_vht, d_vht, x_pot_s, x_charg, x_vhart,
                        #     r1, j_p_s, j_ch, j_vht, nx
                        # )
                        ValueError("case not implemented: PsuedoDiagLoops")
                    else:
                        for i_idx in range(0, nx, 2):
                            iPlusOne = i_idx + 1
                            # Evaluate the splines
                            ppot[i_idx], j_p_s = fsplevalIO(z_p_s, c_p_s, d_p_s, x_pot_s, r1[i_idx], j_p_s)
                            rrho[i_idx], j_ch = fsplevalIO(z_chg, c_chg, d_chg, x_charg, r1[i_idx], j_ch)
                            if build_hpot:
                                hpot00[i_idx], j_vht = fsplevalIO(
                                    z_vht, c_vht, d_vht, x_vhart, r1[i_idx], j_vht
                                )

                            ppot[iPlusOne], j_p_s = fsplevalIO(z_p_s, c_p_s, d_p_s, x_pot_s, r1[iPlusOne], j_p_s)
                            rrho[iPlusOne], j_ch = fsplevalIO(z_chg, c_chg, d_chg, x_charg, r1[iPlusOne], j_ch)
                            if build_hpot:
                                hpot00[iPlusOne], j_vht = fsplevalIO(
                                    z_vht, c_vht, d_vht, x_vhart, r1[iPlusOne], j_vht
                                )
                        # done atom-specific calculations - now compute potentials, charge.
                    if enableMexFilesTest == 1:
                        ValueError("case not implemented: PsuedoDiagLoops")
                        # TODO: switch to C++ code to compare the performance?
                        # ppot2, rrho2, hpot002 = PsuedoDiagLoops(
                        #     z_p_s, c_p_s, d_p_s, z_chg, c_chg, d_chg,
                        #     z_vht, c_vht, d_vht, x_pot_s, x_charg, x_vhart,
                        #     r1, j_p_s, j_ch, j_vht, nx
                        # )
                        #
                        # # Check for discrepancies
                        # if (np.any(np.abs(ppot2 - ppot) > 1e-6) or
                        #         np.any(np.abs(rrho2 - rrho) > 1e-6) or
                        #         np.any(np.abs(hpot002 - hpot00) > 1e-6)):
                        #     # Prepare the stack trace
                        #     current_frame = inspect.currentframe()
                        #     stack_trace = inspect.getouterframes(current_frame)
                        #
                        #     # Raise custom exception with details
                        #     exception = MexFileDiscrepancyError(
                        #         message="Mex file discrepancy for PsuedoDiagLoops",
                        #         identifier=None,  # You can add an identifier if necessary
                        #         stack=[frame.function for frame in stack_trace]  # Extract function names from the stack
                        #     )
                        #     logError(exception)
                        #     raise exception  # Optionally raise the error after logging

                    rrho = np.maximum(0, rrho * (h ** 3) / (4 * np.pi))
                    indx_end = indx + nx

                    pot[indx:indx_end] += ppot
                    rho0[indx:indx_end] += rrho
                    if build_hpot:
                        hpot0[indx:indx_end] += hpot00
                    indx = indx_end

    # --- Zero check: atomic rho0 ---
    print("[atomic] rho0 sum =", float(rho0.sum()),
          " min =", float(rho0.min()),
          " max =", float(rho0.max()),
          " any_nonzero =", bool(np.any(rho0 != 0)))

    # Normalize rho0 to match total valence electron count
    rho0_sum = np.sum(rho0)
    electron_count_initial = float(rho0_sum)
    rho0 = Z_sum * rho0 / rho0_sum
    electron_count_norm = float(np.sum(rho0))
    print("[atomic] rho0 sum =", np.sum(rho0))

    # --- Plot atomic-superposition density along x (mid y,z) using rho0, then reset rho0 ---
    # # Plot e/grid
    # rho0_3d = rho0.reshape(nx, ny, nz, order='F')
    # hpot0_3d = hpot0.reshape(nx, ny, nz, order='F')
    # pot_3d = pot.reshape(nx, ny, nz, order='F')

    # mid-plane line
    # mid_x = nx // 2
    # mid_y = ny // 2
    # mid_z = nz // 2
    # line_hpot0_x = hpot0_3d[:, mid_y, mid_z]
    # line_hpot0_y = hpot0_3d[:, mid_y, mid_z]
    # line_hpot0_z = hpot0_3d[:, mid_y, mid_z]
    # line_pot_x = pot_3d[:, mid_y, mid_z]
    # line_pot_y = pot_3d[:, mid_y, mid_z]
    # line_pot_z = pot_3d[:, mid_y, mid_z]

    # # real x coordinates on the target grid (Bohr), origin at 0
    # x = np.linspace(0, 2 * rad, nx)
    # y = np.linspace(0, 2 * rad, ny)
    # z = np.linspace(0, 2 * rad, nz)

    # plot hpot0
    # plt.figure(figsize=(7.5, 4))
    # plt.plot(x, line_hpot0_x, label='Atomic superposition (mid y,z)')
    # plt.xlabel('x (Bohr)')
    # plt.ylabel('atomic unit')
    # plt.title('Atomic-superposition hartree pot along x (before ML)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(7.5, 4))
    # plt.plot(y, line_hpot0_y, label='Atomic superposition (mid x,z)')
    # plt.xlabel('y (Bohr)')
    # plt.ylabel('atomic unit')
    # plt.title('Atomic-superposition hartree pot along y (before ML)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(7.5, 4))
    # plt.plot(z, line_hpot0_z, label='Atomic superposition (mid x,y)')
    # plt.xlabel('z (Bohr)')
    # plt.ylabel('atomic unit')
    # plt.title('Atomic-superposition hartree pot along z (before ML)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(7.5, 4))
    # plt.plot(x, line_pot_x, label='Atomic superposition (mid y,z)')
    # plt.xlabel('x (Bohr)')
    # plt.ylabel('atomic unit')
    # plt.title('Atomic-superposition pot along x (before ML)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(7.5, 4))
    # plt.plot(y, line_pot_y, label='Atomic superposition (mid x,z)')
    # plt.xlabel('y (Bohr)')
    # plt.ylabel('atomic unit')
    # plt.title('Atomic-superposition pot along y (before ML)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(7.5, 4))
    # plt.plot(z, line_pot_z, label='Atomic superposition (mid x,y)')
    # plt.xlabel('z (Bohr)')
    # plt.ylabel('atomic unit')
    # plt.title('Atomic-superposition pot along z (before ML)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # Plot e/bohr^3
    # rho_bohr3 = rho0 / (h**3)
    # rho_bohr3_3d = rho_bohr3.reshape(nx, ny, nz, order='F')

    # # mid-plane line
    # line_rho0_x = rho_bohr3_3d[:, mid_y, mid_z]  # e/Bohr^3 (after rrho scaling)
    # line_rho0_y = rho_bohr3_3d[mid_x, :, mid_z]
    # line_rho0_z = rho_bohr3_3d[mid_x, mid_y, :]

    # plt.figure(figsize=(7.5, 4))
    # plt.plot(x, line_rho0_x, label='Atomic superposition (mid y,z)')
    # plt.xlabel('x (Bohr)')
    # plt.ylabel('density (e/bohr^3)')
    # plt.title('Atomic-superposition density along x (before ML)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(7.5, 4))
    # plt.plot(y, line_rho0_y, label='Atomic superposition (mid x,z)')
    # plt.xlabel('y (Bohr)')
    # plt.ylabel('density (e/bohr^3)')
    # plt.title('Atomic-superposition density along y (before ML)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(7.5, 4))
    # plt.plot(z, line_rho0_z, label='Atomic superposition (mid x,y)')
    # plt.xlabel('z (Bohr)')
    # plt.ylabel('density (e/bohr^3)')
    # plt.title('Atomic-superposition density along z (before ML)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    print("pseudoDiag_MLgrid completed successfully")

    diag_info = {
        "electron_count_initial": electron_count_initial,
        "electron_count_normalized": electron_count_norm,
        "electron_target": float(Z_sum),
    }

    if return_info:
        return rho0, hpot0, pot, diag_info
    return rho0, hpot0, pot
